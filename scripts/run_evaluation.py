"""Rigorous Evaluation for MVP v2: GroupKFold (disease-grouped)."""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from gensim.models import Word2Vec
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, time, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('results', exist_ok=True)

print("=" * 70)
print("Rigorous Evaluation: GroupKFold (disease-grouped)")
print("=" * 70)

# ============================================================
# Section 1: Load PrimeKG (reused from run_mvp_v2.py)
# ============================================================
t0 = time.time()
print("\n[1/6] Loading PrimeKG...")
df = pd.read_csv('data/kg.csv', low_memory=False)
print(f"  Loaded: {df.shape[0]:,} edges in {time.time()-t0:.1f}s")

dd_mask = ((df['x_type']=='drug')&(df['y_type']=='disease'))|((df['x_type']=='disease')&(df['y_type']=='drug'))
dd_df = df[dd_mask].copy()
is_drug_x = dd_df['x_type'] == 'drug'
dd_norm = pd.DataFrame({
    'drug_id': np.where(is_drug_x, dd_df['x_id'].values, dd_df['y_id'].values),
    'drug_name': np.where(is_drug_x, dd_df['x_name'].values, dd_df['y_name'].values),
    'disease_id': np.where(is_drug_x, dd_df['y_id'].values, dd_df['x_id'].values),
    'disease_name': np.where(is_drug_x, dd_df['y_name'].values, dd_df['x_name'].values),
    'relation': dd_df['relation'].values
})

contra = dd_norm[dd_norm['relation']=='contraindication']
indica = dd_norm[dd_norm['relation']=='indication']
off_label = dd_norm[dd_norm['relation']=='off-label use']
print(f"  Contra: {len(contra):,}, Indica: {len(indica):,}, Off-label: {len(off_label):,}")

# ============================================================
# Section 2: Select diseases/drugs + build dataset
# ============================================================
print("\n[2/6] Selecting diseases & drugs...")
both = set(contra['disease_id']) & set(indica['disease_id'])
dc = contra[contra['disease_id'].isin(both)].groupby(['disease_id','disease_name']).size().reset_index(name='contra_count')
di = indica[indica['disease_id'].isin(both)].groupby(['disease_id','disease_name']).size().reset_index(name='indica_count')
dcounts = dc.merge(di[['disease_id','indica_count']], on='disease_id')
dcounts['total'] = dcounts['contra_count'] + dcounts['indica_count']
dcounts['bal'] = dcounts[['contra_count','indica_count']].min(axis=1)/dcounts[['contra_count','indica_count']].max(axis=1)
sel_dis = dcounts[dcounts['bal']>=0.2].sort_values('total', ascending=False).head(328)
sel_dis_ids = sel_dis['disease_id'].tolist()

sc = contra[contra['disease_id'].isin(sel_dis_ids)]
si = indica[indica['disease_id'].isin(sel_dis_ids)]
drc = sc.groupby(['drug_id','drug_name']).size().reset_index(name='c')
dri = si.groupby(['drug_id','drug_name']).size().reset_index(name='i')
drug_c = drc.merge(dri[['drug_id','i']], on='drug_id', how='outer').fillna(0)
drug_c['total'] = drug_c['c'] + drug_c['i']
sel_drugs = drug_c.sort_values('total', ascending=False).head(1000)
sel_drug_ids = sel_drugs['drug_id'].tolist()

pos = contra[(contra['disease_id'].isin(sel_dis_ids))&(contra['drug_id'].isin(sel_drug_ids))].copy()
neg = indica[(indica['disease_id'].isin(sel_dis_ids))&(indica['drug_id'].isin(sel_drug_ids))].copy()
pos['label'] = 1; neg['label'] = 0
df_dataset = pd.concat([pos, neg], ignore_index=True)
y = df_dataset['label'].values
disease_ids = df_dataset['disease_id'].values
N_DISEASES = len(sel_dis_ids); N_DRUGS = len(sel_drug_ids)
print(f"  {N_DISEASES} diseases, {N_DRUGS} drugs")
print(f"  Dataset: {len(df_dataset)} samples ({len(pos)} contra, {len(neg)} indica)")

target_edges = set()
for d, dis in zip(df_dataset['drug_id'].astype(str), df_dataset['disease_id'].astype(str)):
    target_edges.add((d, dis)); target_edges.add((dis, d))

# ============================================================
# Section 3: Build three graphs
# ============================================================
print("\n[3/6] Building graphs...")

# Graph A: bipartite
t1 = time.time()
dd_rels = df[df['relation'].isin(['contraindication','indication','off-label use'])]
xv = dd_rels['x_id'].astype(str).values; yv = dd_rels['y_id'].astype(str).values
ea = [(xv[i], yv[i]) for i in range(len(dd_rels)) if (xv[i], yv[i]) not in target_edges]
G_A = nx.Graph(); G_A.add_edges_from(ea)
G_A.remove_nodes_from(list(nx.isolates(G_A)))
print(f"  Graph A: {G_A.number_of_nodes()} nodes, {G_A.number_of_edges()} edges ({time.time()-t1:.1f}s)")

# Graph B: 1-hop PrimeKG subgraph
t2 = time.time()
seeds = set(df_dataset['drug_id'].astype(str)) | set(df_dataset['disease_id'].astype(str))
x_s = df['x_id'].astype(str); y_s = df['y_id'].astype(str)
hop1 = df[x_s.isin(seeds)|y_s.isin(seeds)]
xv2 = hop1['x_id'].astype(str).values; yv2 = hop1['y_id'].astype(str).values; rv2 = hop1['relation'].values
eb = [(xv2[i],yv2[i]) for i in range(len(hop1)) if not (rv2[i] in ('contraindication','indication') and (xv2[i],yv2[i]) in target_edges)]
G_B = nx.Graph(); G_B.add_edges_from(eb)
print(f"  Graph B: {G_B.number_of_nodes():,} nodes, {G_B.number_of_edges():,} edges ({time.time()-t2:.1f}s)")

# DISEASES database (Jensen Lab) integration for Graph C
# Replaces DisGeNET which now requires authentication
print("\n  Loading DISEASES database for Graph C...")
import requests
DISGENET_PATH = "data/diseases_knowledge.tsv"
disgenet_edges_added = 0

if not os.path.exists(DISGENET_PATH):
    print("  Downloading DISEASES database (Jensen Lab)...")
    try:
        resp = requests.get("https://download.jensenlab.org/human_disease_knowledge_filtered.tsv", timeout=60)
        resp.raise_for_status()
        with open(DISGENET_PATH, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        print(f"  Saved to {DISGENET_PATH}")
    except Exception as e:
        print(f"  Download failed: {e}")
        DISGENET_PATH = None
else:
    print(f"  Already have: {DISGENET_PATH}")

G_C = G_B.copy()
if DISGENET_PATH and os.path.exists(DISGENET_PATH):
    # DISEASES db has no header; cols: protein_id, gene_name, disease_id, disease_name, source, type, score
    dgn = pd.read_csv(DISGENET_PATH, sep='\t', header=None, low_memory=False,
                      names=['protein_id', 'gene_name', 'disease_id', 'disease_name', 'source', 'type', 'score'])
    primekg_disease_names = {}
    for nid, nname in zip(df[df['x_type']=='disease']['x_id'].astype(str), df[df['x_type']=='disease']['x_name'].str.lower().str.strip()):
        primekg_disease_names[nname] = nid
    for nid, nname in zip(df[df['y_type']=='disease']['y_id'].astype(str), df[df['y_type']=='disease']['y_name'].str.lower().str.strip()):
        primekg_disease_names[nname] = nid
    # Map gene symbol (lowercase) -> PrimeKG node ID
    primekg_gene_names = {}
    for gname, gid in zip(df[df['x_type']=='gene/protein']['x_name'].str.lower().str.strip(), df[df['x_type']=='gene/protein']['x_id'].astype(str)):
        primekg_gene_names[gname] = gid
    for gname, gid in zip(df[df['y_type']=='gene/protein']['y_name'].str.lower().str.strip(), df[df['y_type']=='gene/protein']['y_id'].astype(str)):
        primekg_gene_names[gname] = gid
    existing = set(G_B.edges())
    dnames = dgn['disease_name'].astype(str).str.lower().str.strip().values
    gnames = dgn['gene_name'].astype(str).str.lower().str.strip().values
    new_e = []
    for i in range(len(dgn)):
        did = primekg_disease_names.get(dnames[i])
        pgid = primekg_gene_names.get(gnames[i])
        if did and pgid and (did,pgid) not in existing and (pgid,did) not in existing:
            new_e.append((did, pgid))
            existing.add((did, pgid))
    G_C.add_edges_from(new_e)
    disgenet_edges_added = len(new_e)
    print(f"  DISEASES edges added: {disgenet_edges_added:,}")

print(f"  Graph C: {G_C.number_of_nodes():,} nodes, {G_C.number_of_edges():,} edges")

graphs = {'A': G_A, 'B': G_B, 'C': G_C}

# ============================================================
# Section 4: Compute features (heuristics + Node2Vec)
# ============================================================
print("\n[4/6] Computing heuristic scores...")

def compute_heuristic_scores(G, drug_id, disease_id):
    u, v = str(drug_id), str(disease_id)
    if u not in G or v not in G:
        return 0, 0.0, 0.0, 0, 0.0
    u_n = set(G.neighbors(u)); v_n = set(G.neighbors(v))
    common = u_n & v_n
    cn = len(common)
    aa = sum(1.0/np.log(max(G.degree(w),2)) for w in common)
    union = u_n | v_n
    jc = cn/len(union) if union else 0.0
    pa = G.degree(u)*G.degree(v)
    ra = sum(1.0/max(G.degree(w),1) for w in common)
    return cn, aa, jc, pa, ra

def compute_all_heuristics(G, df_dataset, graph_name):
    drug_ids = df_dataset['drug_id'].astype(str).values
    disease_ids = df_dataset['disease_id'].astype(str).values
    n = len(df_dataset)
    cn_a = np.zeros(n); aa_a = np.zeros(n); jc_a = np.zeros(n); pa_a = np.zeros(n); ra_a = np.zeros(n)
    for i in tqdm(range(n), desc=f"Heur {graph_name}"):
        cn_a[i], aa_a[i], jc_a[i], pa_a[i], ra_a[i] = compute_heuristic_scores(G, drug_ids[i], disease_ids[i])
    result = pd.DataFrame({f'CN_{graph_name}':cn_a, f'AA_{graph_name}':aa_a, f'JC_{graph_name}':jc_a, f'PA_{graph_name}':pa_a, f'RA_{graph_name}':ra_a})
    print(f"  Non-zero CN: {(cn_a>0).sum()}/{n}")
    return result

sys.stdout.flush()
heuristics = {}
for gn in ['A', 'B', 'C']:
    heuristics[gn] = compute_all_heuristics(graphs[gn], df_dataset, gn)
    sys.stdout.flush()

print("\n[5/6] Training Node2Vec embeddings...")

def deepwalk_random_walks(G, num_walks=10, walk_length=20):
    nodes = list(G.nodes())
    walks = []
    for _ in tqdm(range(num_walks), desc="Walks"):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks.append(walk)
    return walks

def train_n2v(G, name, dim=128):
    n_nodes = G.number_of_nodes()
    if n_nodes > 30000:
        nw, wl = 10, 15
    elif n_nodes > 10000:
        nw, wl = 15, 20
    else:
        nw, wl = 30, 30
    print(f"  Graph {name}: {n_nodes:,} nodes, walks={nw}, length={wl}")
    sys.stdout.flush()
    t = time.time()
    walks = deepwalk_random_walks(G, num_walks=nw, walk_length=wl)
    print(f"  Walks generated: {len(walks):,} in {time.time()-t:.1f}s")
    sys.stdout.flush()
    model = Word2Vec(walks, vector_size=dim, window=10, min_count=1, sg=1, workers=4, epochs=1)
    print(f"  Embeddings: {len(model.wv)} nodes in {time.time()-t:.1f}s total")
    return model

def compute_emb_features(model, df_dataset, name, dim=128):
    drug_ids = df_dataset['drug_id'].astype(str).values
    disease_ids = df_dataset['disease_id'].astype(str).values
    n = len(df_dataset)
    had = np.zeros((n, dim)); cos = np.zeros(n); l2 = np.zeros(n)
    found = 0
    for i in range(n):
        if drug_ids[i] in model.wv and disease_ids[i] in model.wv:
            de = model.wv[drug_ids[i]]; dise = model.wv[disease_ids[i]]
            had[i] = de*dise
            np_d = np.linalg.norm(de)*np.linalg.norm(dise)
            cos[i] = np.dot(de,dise)/np_d if np_d>0 else 0
            l2[i] = np.linalg.norm(de-dise)
            found += 1
    cols = [f'had_{j}_{name}' for j in range(dim)]
    result = pd.DataFrame(had, columns=cols)
    result[f'cos_sim_{name}'] = cos; result[f'l2_dist_{name}'] = l2
    print(f"  Pairs embedded: {found}/{n}")
    return result

sys.stdout.flush()
embeddings = {}
n2v_models = {}
for gn in ['A', 'B', 'C']:
    n2v_models[gn] = train_n2v(graphs[gn], gn)
    embeddings[gn] = compute_emb_features(n2v_models[gn], df_dataset, gn)
    sys.stdout.flush()

# Build combined feature matrices for each graph
X_combined = {}
X_heuristic = {}
X_embedding = {}
for gn in ['A', 'B', 'C']:
    X_heuristic[gn] = heuristics[gn].values
    X_embedding[gn] = embeddings[gn].values
    X_combined[gn] = np.hstack([X_heuristic[gn], X_embedding[gn]])
    print(f"  Graph {gn} combined features: {X_combined[gn].shape}")


# ============================================================
# Section 5: GroupKFold + Standard CV
# ============================================================
print("\n" + "=" * 70)
print("[6/6] EVALUATION: GroupKFold (disease-grouped) vs Standard CV")
print("=" * 70)

def group_kfold_cv(X, y, disease_ids, name, n_splits=5):
    """GroupKFold with disease as group - no disease in both train and test."""
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []; accs = []; f1s = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=disease_ids)):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        yp = rf.predict_proba(X[test_idx])[:, 1]
        ypred = rf.predict(X[test_idx])
        yt = y[test_idx]
        aucs.append(roc_auc_score(yt, yp))
        accs.append(accuracy_score(yt, ypred))
        f1s.append(f1_score(yt, ypred))
        print(f"    Fold {fold+1}: AUC={aucs[-1]:.3f}, Acc={accs[-1]:.3f}, F1={f1s[-1]:.3f} (test={len(test_idx)})")
    return {
        'name': name,
        'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs),
        'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
    }

# GroupKFold: all 3 graphs x all 3 methods
methods = ['Heuristics', 'Node2Vec', 'Combined']
gkf_results = []
for gn in ['A', 'B', 'C']:
    print(f"\n  --- Graph {gn} (GroupKFold) ---")
    for method_name, X_method in [('Heuristics', X_heuristic[gn]), ('Node2Vec', X_embedding[gn]), ('Combined', X_combined[gn])]:
        print(f"  {method_name}:")
        r = group_kfold_cv(X_method, y, disease_ids, f"Graph {gn} - {method_name}")
        gkf_results.append(r)
        print(f"    => AUC={r['roc_auc_mean']:.3f}+-{r['roc_auc_std']:.3f}")
    sys.stdout.flush()

gkf_df = pd.DataFrame(gkf_results).set_index('name')

# Standard CV for comparison
print("\n  Running standard 5-fold CV for comparison...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
standard_cv_results = []
for gn in ['A', 'B', 'C']:
    for method_name, X_method in [('Heuristics', X_heuristic[gn]), ('Node2Vec', X_embedding[gn]), ('Combined', X_combined[gn])]:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        roc = cross_val_score(rf, X_method, y, cv=cv, scoring='roc_auc')
        acc = cross_val_score(rf, X_method, y, cv=cv, scoring='accuracy')
        f1 = cross_val_score(rf, X_method, y, cv=cv, scoring='f1')
        standard_cv_results.append({
            'name': f'Graph {gn} - {method_name}',
            'roc_auc_mean': roc.mean(), 'roc_auc_std': roc.std(),
            'accuracy_mean': acc.mean(), 'accuracy_std': acc.std(),
            'f1_mean': f1.mean(), 'f1_std': f1.std(),
        })
std_cv_df = pd.DataFrame(standard_cv_results).set_index('name')


# ============================================================
# Visualization
# ============================================================
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Bar chart - Standard CV vs GroupKFold (Combined method)
ax = axes[0, 0]
graph_labels = ['A', 'B', 'C']
bar_width = 0.3
x = np.arange(len(graph_labels))
std_vals = [std_cv_df.loc[f'Graph {g} - Combined', 'roc_auc_mean'] for g in graph_labels]
std_errs = [std_cv_df.loc[f'Graph {g} - Combined', 'roc_auc_std'] for g in graph_labels]
gkf_vals = [gkf_df.loc[f'Graph {g} - Combined', 'roc_auc_mean'] for g in graph_labels]
gkf_errs = [gkf_df.loc[f'Graph {g} - Combined', 'roc_auc_std'] for g in graph_labels]
bars1 = ax.bar(x - bar_width/2, std_vals, bar_width, yerr=std_errs, label='Standard CV', color='#2196F3', capsize=5, alpha=0.85)
bars2 = ax.bar(x + bar_width/2, gkf_vals, bar_width, yerr=gkf_errs, label='GroupKFold', color='#FF9800', capsize=5, alpha=0.85)
ax.set_ylabel('ROC-AUC'); ax.set_title('Standard CV vs GroupKFold (Combined Method)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'Graph {g}' for g in graph_labels])
ax.legend(); ax.set_ylim([0.4, 1.05])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3); ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

# Plot 2: GroupKFold heatmap (all methods)
ax = axes[0, 1]
hm = np.zeros((3, 3))
for i, g in enumerate(graph_labels):
    for j, m in enumerate(methods):
        k = f'Graph {g} - {m}'
        hm[i, j] = gkf_df.loc[k, 'roc_auc_mean'] if k in gkf_df.index else 0
sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=methods, yticklabels=[f'Graph {g}' for g in graph_labels],
            ax=ax, vmin=0.5, vmax=1.0)
ax.set_title('GroupKFold ROC-AUC (Disease-Grouped)', fontweight='bold')

# Plot 3: Gap analysis - how much does standard CV overestimate?
ax = axes[1, 0]
gaps = {}
for m in methods:
    gaps[m] = [std_cv_df.loc[f'Graph {g} - {m}', 'roc_auc_mean'] - gkf_df.loc[f'Graph {g} - {m}', 'roc_auc_mean'] for g in graph_labels]
bar_width = 0.25
colors = ['#2196F3', '#FF9800', '#4CAF50']
for i, m in enumerate(methods):
    ax.bar(x + i*bar_width, gaps[m], bar_width, label=m, color=colors[i], alpha=0.85)
ax.set_ylabel('AUC Gap (Standard - GroupKFold)'); ax.set_title('Standard CV Overestimation', fontweight='bold')
ax.set_xticks(x + bar_width); ax.set_xticklabels([f'Graph {g}' for g in graph_labels])
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 4: All methods comparison (GroupKFold)
ax = axes[1, 1]
bar_width = 0.25
for i, m in enumerate(methods):
    vals = [gkf_df.loc[f'Graph {g} - {m}', 'roc_auc_mean'] for g in graph_labels]
    errs = [gkf_df.loc[f'Graph {g} - {m}', 'roc_auc_std'] for g in graph_labels]
    ax.bar(x + i*bar_width, vals, bar_width, yerr=errs, label=m, color=colors[i], capsize=3, alpha=0.85)
ax.set_ylabel('ROC-AUC'); ax.set_title('GroupKFold: All Methods', fontweight='bold')
ax.set_xticks(x + bar_width); ax.set_xticklabels([f'Graph {g}' for g in graph_labels])
ax.legend(); ax.set_ylim([0.4, 1.05])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/mvp_v2_evaluation.png', dpi=150, bbox_inches='tight')
print("Saved: results/mvp_v2_evaluation.png")


# ============================================================
# Summary Table
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF EVALUATION RESULTS")
print("=" * 70)

print(f"\n{'Protocol':<25} {'Graph':<10} {'Method':<15} {'AUC':>8} {'Std':>8} {'Acc':>8} {'F1':>8}")
print("-" * 80)

for gn in ['A', 'B', 'C']:
    for m in methods:
        k = f'Graph {gn} - {m}'
        r = std_cv_df.loc[k]
        print(f"{'Standard CV':<25} {'Graph '+gn:<10} {m:<15} {r['roc_auc_mean']:>8.3f} {r['roc_auc_std']:>8.3f} {r['accuracy_mean']:>8.3f} {r['f1_mean']:>8.3f}")

print("-" * 80)

for gn in ['A', 'B', 'C']:
    for m in methods:
        k = f'Graph {gn} - {m}'
        r = gkf_df.loc[k]
        print(f"{'GroupKFold (disease)':<25} {'Graph '+gn:<10} {m:<15} {r['roc_auc_mean']:>8.3f} {r['roc_auc_std']:>8.3f} {r['accuracy_mean']:>8.3f} {r['f1_mean']:>8.3f}")

print("-" * 80)

print("\nKEY FINDINGS:")
for gn in ['A', 'B', 'C']:
    std_auc = std_cv_df.loc[f'Graph {gn} - Combined', 'roc_auc_mean']
    gkf_auc = gkf_df.loc[f'Graph {gn} - Combined', 'roc_auc_mean']
    gap = std_auc - gkf_auc
    print(f"  Graph {gn}: Standard CV={std_auc:.3f}, GroupKFold={gkf_auc:.3f}  (inflation={gap:+.3f})")

best_gkf = gkf_df['roc_auc_mean'].idxmax()
print(f"\n  Best GroupKFold result: {best_gkf} (AUC={gkf_df.loc[best_gkf, 'roc_auc_mean']:.3f})")

print(f"\nTotal runtime: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
print("=" * 70)
print("DONE")
