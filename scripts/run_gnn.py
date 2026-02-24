"""Heterogeneous GNN for drug-disease link classification.

Architecture:
  - Learnable node embeddings (one table per node type, 64-dim)
  - 2-layer HeteroConv with SAGEConv per edge type
  - MLP link prediction head on concatenated drug + disease embeddings

Evaluation (leak-proof):
  5-fold GroupKFold grouped by disease. No disease appears in both
  train and test within a fold. Comparable to v3's LODO but 5 folds
  instead of 50 single-disease splits — necessary because each fold
  requires training the GNN from scratch.

Comparison target: v3 baseline AUC=0.738 ± 0.322 (RF + within-fold Node2Vec, LODO 50 diseases).

Run from project root: python scripts/run_gnn.py
"""
import os, time, warnings, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

os.makedirs('results', exist_ok=True)

# Mac MPS if available, otherwise CPU
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'

EMB_DIM    = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
LR         = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS     = 50
DROPOUT    = 0.3

# Edge types excluded from message-passing (these are the prediction targets)
TARGET_RELS = {'contraindication', 'indication', 'off-label use'}

print("=" * 60)
print("GNN Experiment: Heterogeneous GNN on PrimeKG")
print("=" * 60)
print(f"  Device: {DEVICE} | EMB: {EMB_DIM} | Hidden: {HIDDEN_DIM} | "
      f"Layers: {NUM_LAYERS} | Epochs: {EPOCHS}")

# ============================================================
# [1/5] Load PrimeKG + build dataset (identical to v3)
# ============================================================
t0 = time.time()
print("\n[1/5] Loading PrimeKG + building dataset...")
df = pd.read_csv('data/kg.csv', low_memory=False)
print(f"  Loaded: {df.shape[0]:,} edges in {time.time()-t0:.1f}s")

dd_mask = (((df['x_type']=='drug') & (df['y_type']=='disease')) |
           ((df['x_type']=='disease') & (df['y_type']=='drug')))
dd_df = df[dd_mask].copy()
is_drug_x = dd_df['x_type'] == 'drug'
dd_norm = pd.DataFrame({
    'drug_id':      np.where(is_drug_x, dd_df['x_id'].values,   dd_df['y_id'].values),
    'drug_name':    np.where(is_drug_x, dd_df['x_name'].values, dd_df['y_name'].values),
    'disease_id':   np.where(is_drug_x, dd_df['y_id'].values,   dd_df['x_id'].values),
    'disease_name': np.where(is_drug_x, dd_df['y_name'].values, dd_df['x_name'].values),
    'relation':     dd_df['relation'].values,
})
contra = dd_norm[dd_norm['relation'] == 'contraindication']
indica = dd_norm[dd_norm['relation'] == 'indication']

both     = set(contra['disease_id']) & set(indica['disease_id'])
dc = contra[contra['disease_id'].isin(both)].groupby(['disease_id','disease_name']).size().reset_index(name='contra_count')
di = indica[indica['disease_id'].isin(both)].groupby(['disease_id','disease_name']).size().reset_index(name='indica_count')
dcounts = dc.merge(di[['disease_id','indica_count']], on='disease_id')
dcounts['total'] = dcounts['contra_count'] + dcounts['indica_count']
dcounts['bal']   = dcounts[['contra_count','indica_count']].min(axis=1) / dcounts[['contra_count','indica_count']].max(axis=1)
sel_dis     = dcounts[dcounts['bal'] >= 0.2].sort_values('total', ascending=False).head(328)
sel_dis_ids = sel_dis['disease_id'].tolist()

sc = contra[contra['disease_id'].isin(sel_dis_ids)]
si = indica[indica['disease_id'].isin(sel_dis_ids)]
drc = sc.groupby(['drug_id','drug_name']).size().reset_index(name='c')
dri = si.groupby(['drug_id','drug_name']).size().reset_index(name='i')
drug_c = drc.merge(dri[['drug_id','i']], on='drug_id', how='outer').fillna(0)
drug_c['total'] = drug_c['c'] + drug_c['i']
sel_drug_ids = drug_c.sort_values('total', ascending=False).head(1000)['drug_id'].tolist()

pos = contra[(contra['disease_id'].isin(sel_dis_ids)) & (contra['drug_id'].isin(sel_drug_ids))].copy()
neg = indica[(indica['disease_id'].isin(sel_dis_ids)) & (indica['drug_id'].isin(sel_drug_ids))].copy()
pos['label'] = 1; neg['label'] = 0
df_dataset = pd.concat([pos, neg], ignore_index=True)
y = df_dataset['label'].values
print(f"  {len(sel_dis_ids)} diseases, {len(sel_drug_ids)} drugs")
print(f"  Dataset: {len(df_dataset)} samples ({len(pos)} contra, {len(neg)} indica)")

# ============================================================
# [2/5] Build 1-hop subgraph (no target edges)
# ============================================================
print("\n[2/5] Building 1-hop KG subgraph (target relations excluded)...")
t1 = time.time()
seeds = set(df_dataset['drug_id'].astype(str)) | set(df_dataset['disease_id'].astype(str))
x_s = df['x_id'].astype(str); y_s = df['y_id'].astype(str)
hop1 = df[x_s.isin(seeds) | y_s.isin(seeds)].copy()
hop1 = hop1[~hop1['relation'].isin(TARGET_RELS)].reset_index(drop=True)
hop1['x_id'] = hop1['x_id'].astype(str)
hop1['y_id'] = hop1['y_id'].astype(str)
print(f"  Edges after removing target relations: {len(hop1):,}  ({time.time()-t1:.1f}s)")

# ============================================================
# [3/5] Build PyG HeteroData
# ============================================================
print("\n[3/5] Building PyG heterogeneous graph...")
t2 = time.time()

def sanitize(s):
    """Make a valid Python identifier for use as a node/edge type key."""
    return s.replace('/', '_').replace('-', '_').replace(' ', '_')

# Collect unique nodes per sanitized type
node_sets = defaultdict(set)
hop1_x_types = hop1['x_type'].apply(sanitize).values
hop1_y_types = hop1['y_type'].apply(sanitize).values
for i in range(len(hop1)):
    node_sets[hop1_x_types[i]].add(hop1['x_id'].values[i])
    node_sets[hop1_y_types[i]].add(hop1['y_id'].values[i])

# Build per-type index mappings (node_id → local int index)
type_to_id_map = {}
node_type_sizes = {}
for ntype, nset in node_sets.items():
    sorted_nodes = sorted(nset)
    type_to_id_map[ntype] = {nid: i for i, nid in enumerate(sorted_nodes)}
    node_type_sizes[ntype] = len(nset)
    print(f"  {ntype}: {len(nset):,} nodes")

# Build edge_index tensors grouped by (src_type, relation, dst_type)
hop1_rels = hop1['relation'].apply(sanitize).values
data = HeteroData()
for ntype, count in node_type_sizes.items():
    data[ntype].num_nodes = count

grouped = pd.DataFrame({
    'x_type': hop1_x_types,
    'rel':    hop1_rels,
    'y_type': hop1_y_types,
    'x_id':   hop1['x_id'].values,
    'y_id':   hop1['y_id'].values,
}).groupby(['x_type', 'rel', 'y_type'])

n_edge_types = 0
for (x_type, rel, y_type), grp in grouped:
    x_map = type_to_id_map[x_type]
    y_map = type_to_id_map[y_type]
    valid = grp['x_id'].isin(x_map) & grp['y_id'].isin(y_map)
    grp   = grp[valid]
    if len(grp) == 0:
        continue
    src = grp['x_id'].map(x_map).astype(int).values
    dst = grp['y_id'].map(y_map).astype(int).values
    ei  = torch.tensor([src, dst], dtype=torch.long)
    data[x_type, rel, y_type].edge_index = ei
    # Reverse edges for undirected message passing
    data[y_type, rel + '_rev', x_type].edge_index = torch.tensor([dst, src], dtype=torch.long)
    n_edge_types += 1

print(f"\n  {len(node_type_sizes)} node types, {n_edge_types * 2} edge types (incl. reverse)")
print(f"  Total nodes: {sum(node_type_sizes.values()):,}")
total_edges = sum(et.edge_index.shape[1] for et in data.edge_stores)
print(f"  Total edges: {total_edges:,} (incl. reverse)  ({time.time()-t2:.1f}s)")

# Map dataset drug/disease pairs → local PyG indices
drug_local = df_dataset['drug_id'].astype(str).map(type_to_id_map['drug']).fillna(-1).astype(int).values
dis_local  = df_dataset['disease_id'].astype(str).map(type_to_id_map['disease']).fillna(-1).astype(int).values
n_missing  = (drug_local == -1).sum() + (dis_local == -1).sum()
print(f"  Pairs with missing node mapping: {n_missing} (should be 0)")

# ============================================================
# [4/5] GNN Model
# ============================================================

class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN with learnable node embeddings and SAGEConv per edge type.

    Message passing uses the KG structure (biological edges only — target
    drug-disease interaction edges are excluded from the graph).
    The link prediction head classifies (drug, disease) pairs using the
    learned node representations.
    """
    def __init__(self, node_type_sizes, edge_types, emb_dim, hidden_dim, dropout):
        super().__init__()

        # One embedding table per node type
        self.embeddings = nn.ModuleDict({
            nt: nn.Embedding(n, emb_dim)
            for nt, n in node_type_sizes.items()
        })

        # Two HeteroConv layers (SAGEConv per edge type)
        self.convs = nn.ModuleList([
            HeteroConv({
                et: SAGEConv((-1, -1), hidden_dim)
                for et in edge_types
            }, aggr='mean')
            for _ in range(NUM_LAYERS)
        ])

        self.dropout = nn.Dropout(dropout)

        # Link prediction head: [drug_emb || disease_emb] → scalar logit
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x_dict = {nt: self.embeddings[nt].weight for nt in self.embeddings}
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
        return x_dict

    def predict_links(self, x_dict, drug_idx, disease_idx):
        drug_emb    = x_dict['drug'][drug_idx]
        disease_emb = x_dict['disease'][disease_idx]
        return self.link_mlp(torch.cat([drug_emb, disease_emb], dim=-1)).squeeze(-1)


# ============================================================
# [5/5] 5-Fold GroupKFold (grouped by disease)
# ============================================================
print("\n[5/5] 5-fold GroupKFold evaluation (no disease in both train and test)...")
print(f"  Baseline: v3 LODO AUC=0.738 ± 0.322 (RF+within-fold Node2Vec)\n")

edge_types = list(data.edge_index_dict.keys())
groups     = df_dataset['disease_id'].values
gkf        = GroupKFold(n_splits=5)
data_dev   = data.to(DEVICE)

fold_results = []
for fold, (train_idx, test_idx) in enumerate(gkf.split(drug_local, y, groups=groups)):
    t_fold = time.time()
    n_test_diseases = len(set(groups[test_idx]))
    print(f"  Fold {fold+1}/5 | train={len(train_idx):,} pairs | "
          f"test={len(test_idx):,} pairs across {n_test_diseases} held-out diseases")

    # Build fresh model for each fold
    model     = HeteroGNN(node_type_sizes, edge_types, EMB_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    tr_drug = torch.tensor(drug_local[train_idx], dtype=torch.long, device=DEVICE)
    tr_dis  = torch.tensor(dis_local[train_idx],  dtype=torch.long, device=DEVICE)
    tr_y    = torch.tensor(y[train_idx],           dtype=torch.float, device=DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        x_dict = model(data_dev)
        logits = model.predict_links(x_dict, tr_drug, tr_dis)
        loss   = criterion(logits, tr_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:2d}/{EPOCHS}  loss={loss.item():.4f}", flush=True)

    # Evaluate on held-out disease pairs
    te_drug = torch.tensor(drug_local[test_idx], dtype=torch.long, device=DEVICE)
    te_dis  = torch.tensor(dis_local[test_idx],  dtype=torch.long, device=DEVICE)
    te_y    = y[test_idx]

    model.eval()
    with torch.no_grad():
        x_dict    = model(data_dev)
        te_probs  = torch.sigmoid(model.predict_links(x_dict, te_drug, te_dis)).cpu().numpy()

    auc = roc_auc_score(te_y, te_probs)
    pred = (te_probs >= 0.5).astype(int)
    acc  = accuracy_score(te_y, pred)
    f1   = f1_score(te_y, pred)

    elapsed = time.time() - t_fold
    print(f"  → Fold {fold+1}: AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  ({elapsed:.0f}s)\n")
    fold_results.append({
        'fold': fold + 1, 'auc': auc, 'acc': acc, 'f1': f1,
        'n_test': len(test_idx), 'n_test_diseases': n_test_diseases,
    })

# ============================================================
# Results
# ============================================================
print("=" * 60)
print("RESULTS")
print("=" * 60)
aucs = [r['auc'] for r in fold_results]
accs = [r['acc'] for r in fold_results]
f1s  = [r['f1']  for r in fold_results]

for r in fold_results:
    print(f"  Fold {r['fold']}: AUC={r['auc']:.3f}  Acc={r['acc']:.3f}  F1={r['f1']:.3f}")

print(f"\n  GNN (GroupKFold disease):  AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"  Baseline v3 (LODO RF+N2V): AUC=0.738 ± 0.322")
print(f"  Δ vs baseline: {np.mean(aucs) - 0.738:+.3f}")

results = {
    'fold_results':  fold_results,
    'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
    'mean_acc': np.mean(accs), 'std_acc': np.std(accs),
    'mean_f1':  np.mean(f1s),  'std_f1':  np.std(f1s),
    'model_config': {
        'emb_dim': EMB_DIM, 'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS, 'lr': LR, 'epochs': EPOCHS,
        'dropout': DROPOUT,
    },
}
with open('results/gnn_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n  Saved: results/gnn_results.pkl")
print(f"  Total runtime: {(time.time()-t0)/60:.1f} min")
print("=" * 60)
print("DONE")
