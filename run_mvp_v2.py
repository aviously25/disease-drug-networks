"""MVP v2: Link Prediction - Run as script for real-time output."""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from gensim.models import Word2Vec
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, time, warnings, pickle, json
warnings.filterwarnings('ignore')
np.random.seed(42)

RESULTS = {}  # store all results for notebook injection later

print("=" * 60)
print("MVP v2: Link Prediction for Drug-Disease Prediction")
print("=" * 60)

# ============================================================
# Section 2: Load PrimeKG
# ============================================================
t0 = time.time()
print("\n[1/11] Loading PrimeKG...")
df = pd.read_csv('data/kg.csv', low_memory=False)
print(f"  Loaded: {df.shape[0]:,} edges in {time.time()-t0:.1f}s")

# Normalize drug-disease edges (vectorized)
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
# Section 3: Select diseases/drugs + build dataset
# ============================================================
print("\n[2/11] Selecting diseases & drugs...")
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
N_DISEASES = len(sel_dis_ids); N_DRUGS = len(sel_drug_ids)
print(f"  {N_DISEASES} diseases, {N_DRUGS} drugs")
print(f"  Dataset: {len(df_dataset)} samples ({len(pos)} contra, {len(neg)} indica)")

# Target edges to exclude
target_edges = set()
for d, dis in zip(df_dataset['drug_id'].astype(str), df_dataset['disease_id'].astype(str)):
    target_edges.add((d, dis)); target_edges.add((dis, d))

# ============================================================
# Section 4: Build three graphs
# ============================================================
print("\n[3/11] Building graphs...")

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
print("\n[4/11] DISEASES database integration...")
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
    print(f"  DISEASES rows: {len(dgn):,}, cols: {list(dgn.columns)[:5]}")

    # Map disease names (lowercase) -> PrimeKG node ID
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
else:
    print("  No DISEASES data. Graph C = Graph B.")

print(f"  Graph C: {G_C.number_of_nodes():,} nodes, {G_C.number_of_edges():,} edges")
print(f"  Extra edges vs B: {G_C.number_of_edges()-G_B.number_of_edges():,}")

graphs = {'A (no bio edges)': G_A, 'B (full PrimeKG)': G_B, 'C (PrimeKG+DisGeNET)': G_C}
RESULTS['graphs'] = {n: (G.number_of_nodes(), G.number_of_edges()) for n, G in graphs.items()}
print(f"\nGraph summary:")
for name, G in graphs.items():
    avg_deg = 2*G.number_of_edges()/max(G.number_of_nodes(),1)
    print(f"  {name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, avg_deg={avg_deg:.1f}")

# ============================================================
# Section 5: Heuristic scores
# ============================================================
print("\n[5/11] Computing heuristic scores...")

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
heuristics_A = compute_all_heuristics(G_A, df_dataset, 'A'); sys.stdout.flush()
heuristics_B = compute_all_heuristics(G_B, df_dataset, 'B'); sys.stdout.flush()
heuristics_C = compute_all_heuristics(G_C, df_dataset, 'C'); sys.stdout.flush()

# ============================================================
# Section 6: Node2Vec Embeddings
# ============================================================
print("\n[6/11] Training Node2Vec embeddings...")

def deepwalk_random_walks(G, num_walks=10, walk_length=20):
    """Generate random walks (DeepWalk style - no transition probabilities)."""
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
n2v_A = train_n2v(G_A, 'A'); sys.stdout.flush()
emb_A = compute_emb_features(n2v_A, df_dataset, 'A'); sys.stdout.flush()

n2v_B = train_n2v(G_B, 'B'); sys.stdout.flush()
emb_B = compute_emb_features(n2v_B, df_dataset, 'B'); sys.stdout.flush()

n2v_C = train_n2v(G_C, 'C'); sys.stdout.flush()
emb_C = compute_emb_features(n2v_C, df_dataset, 'C'); sys.stdout.flush()

# ============================================================
# Section 7: Train Models (3 graphs x 3 methods)
# ============================================================
print("\n[7/11] Training models (5-fold CV)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_cv(X, y, cv, name):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    roc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    acc = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(rf, X, y, cv=cv, scoring='f1')
    return {'name':name, 'roc_auc_mean':roc.mean(), 'roc_auc_std':roc.std(),
            'accuracy_mean':acc.mean(), 'accuracy_std':acc.std(),
            'f1_mean':f1.mean(), 'f1_std':f1.std()}

all_results = []
for gn, hdf, edf in [('A',heuristics_A,emb_A),('B',heuristics_B,emb_B),('C',heuristics_C,emb_C)]:
    Xh = hdf.values; Xe = edf.values; Xc = np.hstack([Xh, Xe])
    print(f"\n  --- Graph {gn} ---")
    r1 = evaluate_cv(Xh, y, cv, f'Graph {gn} - Heuristics')
    print(f"    Heuristics: AUC={r1['roc_auc_mean']:.3f}+-{r1['roc_auc_std']:.3f}")
    all_results.append(r1); sys.stdout.flush()
    r2 = evaluate_cv(Xe, y, cv, f'Graph {gn} - Node2Vec')
    print(f"    Node2Vec:   AUC={r2['roc_auc_mean']:.3f}+-{r2['roc_auc_std']:.3f}")
    all_results.append(r2); sys.stdout.flush()
    r3 = evaluate_cv(Xc, y, cv, f'Graph {gn} - Combined')
    print(f"    Combined:   AUC={r3['roc_auc_mean']:.3f}+-{r3['roc_auc_std']:.3f}")
    all_results.append(r3); sys.stdout.flush()

results_df = pd.DataFrame(all_results).set_index('name').sort_values('roc_auc_mean', ascending=False)
print("\nAll results sorted by AUC:")
for idx, row in results_df.iterrows():
    print(f"  {idx:<35} AUC={row['roc_auc_mean']:.3f} Acc={row['accuracy_mean']:.3f} F1={row['f1_mean']:.3f}")
RESULTS['cv_results'] = results_df.to_dict()

# ============================================================
# Section 8: LODO CV (sampled 50 diseases)
# ============================================================
print("\n[8/11] Disease-level CV (LODO, 50 sampled diseases)...")
disease_ids = df_dataset['disease_id'].values

def lodo_cv(X, y, disease_ids, name, max_d=50):
    unique_d = sorted(set(disease_ids))
    if len(unique_d) > max_d:
        rng = np.random.RandomState(42)
        unique_d = sorted(rng.choice(unique_d, size=max_d, replace=False))
    results = []
    for hd in tqdm(unique_d, desc=f"LODO {name}"):
        tm = np.array([d==hd for d in disease_ids])
        if tm.sum()<5: continue
        yt = y[tm]
        if len(np.unique(yt))<2: continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[~tm], y[~tm])
        yp = rf.predict_proba(X[tm])[:,1]
        ypred = rf.predict(X[tm])
        results.append({'disease_id':hd, 'roc_auc':roc_auc_score(yt,yp), 'accuracy':accuracy_score(yt,ypred), 'f1':f1_score(yt,ypred)})
    return pd.DataFrame(results)

lodo_results = {}
for gn, hdf, edf in [('A',heuristics_A,emb_A),('B',heuristics_B,emb_B),('C',heuristics_C,emb_C)]:
    Xc = np.hstack([hdf.values, edf.values])
    lr = lodo_cv(Xc, y, disease_ids, f'Graph {gn}')
    lodo_results[gn] = lr
    print(f"  Graph {gn}: AUC={lr['roc_auc'].mean():.3f}+-{lr['roc_auc'].std():.3f} ({len(lr)} diseases)")
    sys.stdout.flush()

RESULTS['lodo'] = {g: {'mean': r['roc_auc'].mean(), 'std': r['roc_auc'].std(), 'n': len(r)} for g, r in lodo_results.items()}

# ============================================================
# Section 9: Comparison with MVP v1
# ============================================================
print("\n[9/11] Comparison with MVP v1...")
print(f"\n{'Model':<40} {'ROC-AUC':>10} {'Accuracy':>10} {'F1':>10} {'LODO AUC':>10}")
print("-"*80)
print("--- MVP v1 (Feature Engineering) ---")
print(f"{'RF_Combined (v1)':<40} {'0.985':>10} {'0.947':>10} {'0.959':>10} {'0.831':>10}")
print("\n--- MVP v2 (Link Prediction) ---")
for idx, row in results_df.iterrows():
    gn = idx.split(' ')[1]
    la = lodo_results.get(gn, pd.DataFrame())
    ls = f"{la['roc_auc'].mean():.3f}" if len(la)>0 else 'N/A'
    print(f"{idx:<40} {row['roc_auc_mean']:>10.3f} {row['accuracy_mean']:>10.3f} {row['f1_mean']:>10.3f} {ls:>10}")

# ============================================================
# Section 10: Visualization
# ============================================================
print("\n[10/11] Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Bar chart
ax = axes[0,0]
graph_labels = ['A (no bio)','B (full PrimeKG)','C (+DisGeNET)']
methods = ['Heuristics','Node2Vec','Combined']
colors = ['#2196F3','#FF9800','#4CAF50']
x = np.arange(3); width = 0.25
for i, m in enumerate(methods):
    vals = [results_df.loc[f'Graph {g} - {m}','roc_auc_mean'] if f'Graph {g} - {m}' in results_df.index else 0 for g in 'ABC']
    errs = [results_df.loc[f'Graph {g} - {m}','roc_auc_std'] if f'Graph {g} - {m}' in results_df.index else 0 for g in 'ABC']
    ax.bar(x+i*width, vals, width, yerr=errs, label=m, color=colors[i], capsize=3, alpha=0.85)
ax.set_ylabel('ROC-AUC'); ax.set_title('ROC-AUC by Graph Config & Method', fontweight='bold')
ax.set_xticks(x+width); ax.set_xticklabels(graph_labels); ax.legend(); ax.set_ylim([0.4,1.0])
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3); ax.grid(axis='y', alpha=0.3)

# Heatmap
ax = axes[0,1]
hm = np.zeros((3,3))
for i, g in enumerate('ABC'):
    for j, m in enumerate(methods):
        k = f'Graph {g} - {m}'
        hm[i,j] = results_df.loc[k,'roc_auc_mean'] if k in results_df.index else 0
sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlOrRd', xticklabels=methods, yticklabels=graph_labels, ax=ax, vmin=0.5, vmax=1.0)
ax.set_title('ROC-AUC Heatmap', fontweight='bold')

# LODO
ax = axes[1,0]
lm = [lodo_results[g]['roc_auc'].mean() if g in lodo_results else 0 for g in 'ABC']
ls = [lodo_results[g]['roc_auc'].std() if g in lodo_results else 0 for g in 'ABC']
ax.bar(graph_labels, lm, yerr=ls, color=colors, capsize=5, alpha=0.85)
ax.axhline(y=0.831, color='red', linestyle='--', alpha=0.7, label='MVP v1 LODO (0.831)')
ax.set_ylabel('ROC-AUC'); ax.set_title('Disease-Level CV (LODO)', fontweight='bold')
ax.legend(); ax.set_ylim([0.4,1.0]); ax.grid(axis='y', alpha=0.3)

# ROC curves
ax = axes[1,1]
for gn, hdf, edf, col in [('A',heuristics_A,emb_A,'#2196F3'),('B',heuristics_B,emb_B,'#FF9800'),('C',heuristics_C,emb_C,'#4CAF50')]:
    Xc = np.hstack([hdf.values, edf.values])
    Xtr,Xte,ytr,yte = train_test_split(Xc, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    yp = rf.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, yp)
    ax.plot(fpr, tpr, color=col, lw=2, label=f'Graph {gn} (AUC={auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves - Combined', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mvp_v2_results.png', dpi=150, bbox_inches='tight')
print("  Saved: mvp_v2_results.png")

# ============================================================
# Section 11: Summary
# ============================================================
print("\n[11/11] Summary")
print("="*70)
print(f"DATASET: {len(df_dataset)} samples ({N_DISEASES} diseases, {N_DRUGS} drugs)")
print(f"\nGRAPH CONFIGS:")
for name, G in graphs.items():
    print(f"  {name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  DisGeNET edges added: {disgenet_edges_added:,}")

print(f"\n5-FOLD CV RESULTS:")
for idx, row in results_df.iterrows():
    print(f"  {idx:<35} AUC={row['roc_auc_mean']:.3f}+-{row['roc_auc_std']:.3f}")

print(f"\nLODO RESULTS (Combined, 50 diseases):")
for g in 'ABC':
    if g in lodo_results and len(lodo_results[g])>0:
        lr = lodo_results[g]
        print(f"  Graph {g}: AUC={lr['roc_auc'].mean():.3f}+-{lr['roc_auc'].std():.3f}")

best_v2 = results_df.iloc[0]
print(f"\nBest MVP v1: RF_Combined AUC=0.985, LODO=0.831")
print(f"Best MVP v2: {results_df.index[0]} AUC={best_v2['roc_auc_mean']:.3f}")

# Key takeaways
best_bg = {g: results_df[results_df.index.str.contains(f'Graph {g}')]['roc_auc_mean'].max() for g in 'ABC'}
bg = max(best_bg, key=best_bg.get)
print(f"\nKEY TAKEAWAYS:")
print(f"  1. Best graph: Graph {bg} (AUC={best_bg[bg]:.3f})")

dgn_diff = best_bg.get('C',0) - best_bg.get('B',0)
if dgn_diff > 0.005: print(f"  2. DisGeNET HELPS: +{dgn_diff:.3f}")
elif dgn_diff < -0.005: print(f"  2. DisGeNET HURTS: {dgn_diff:.3f}")
else: print(f"  2. DisGeNET minimal effect: {dgn_diff:+.3f}")

v_diff = best_v2['roc_auc_mean'] - 0.985
if v_diff > 0.005: print(f"  3. Link prediction OUTPERFORMS feature eng: {v_diff:+.3f}")
elif v_diff > -0.005: print(f"  3. Link prediction COMPARABLE to feature eng: {v_diff:+.3f}")
else: print(f"  3. Feature eng outperforms link prediction: {v_diff:+.3f}")

method_best = {m: results_df[results_df.index.str.contains(m)]['roc_auc_mean'].max() for m in methods}
bm = max(method_best, key=method_best.get)
print(f"  4. Best method: {bm} (AUC={method_best[bm]:.3f})")

# Save results for notebook injection
RESULTS['best_v2'] = best_v2['roc_auc_mean']
RESULTS['disgenet_added'] = disgenet_edges_added
with open('mvp_v2_results.pkl', 'wb') as f:
    pickle.dump(RESULTS, f)

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("="*70)
print("DONE")
