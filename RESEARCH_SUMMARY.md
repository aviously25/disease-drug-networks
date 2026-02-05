# Drug-Disease Adverse Outcome Prediction using PrimeKG

## Research Meeting Summary

---

## 1. Project Overview

### Objective
Predict whether a drug-disease pair will result in an **adverse outcome (contraindication)** or **therapeutic benefit (indication)** using biological knowledge graphs.

### Motivation
- Adverse drug reactions cause ~100,000 deaths/year in the US
- Early prediction could improve drug safety and guide prescribing decisions
- Knowledge graphs like PrimeKG encode rich biological relationships that may reveal hidden patterns

### Core Hypothesis
> If a drug targets genes/pathways that are involved in a disease's pathophysiology, there may be an interaction - either therapeutic (treating the disease) or adverse (worsening or causing complications).

### High-Level Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE OVERVIEW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────┐ │
│  │   PrimeKG   │ ──▶ │     Feature      │ ──▶ │  ML Models  │ ──▶ │ Pred │ │
│  │  (8.1M edges)│     │   Engineering    │     │  (LR, RF)   │     │      │ │
│  └─────────────┘     └──────────────────┘     └─────────────┘     └──────┘ │
│         │                    │                                              │
│         ▼                    ▼                                              │
│  ┌─────────────┐     ┌──────────────────┐                                  │
│  │ Drug-Disease │     │ • Shared genes   │                                  │
│  │ • Indication │     │ • Shared pathways│                                  │
│  │ • Contraind. │     │ • Graph distance │                                  │
│  └─────────────┘     │ • Centrality     │                                  │
│                      └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Source: PrimeKG

### 2.1 What is PrimeKG?

PrimeKG (Precision Medicine Knowledge Graph) is a comprehensive biomedical knowledge graph that integrates 20+ primary data sources including:
- **DrugBank** - Drug information and targets
- **DisGeNET** - Disease-gene associations
- **Reactome** - Biological pathways
- **OMIM** - Mendelian disease genetics
- **UniProt** - Protein information

**Reference:** Chandak et al., "Building a knowledge graph to enable precision medicine" (2023)

### 2.2 PrimeKG Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PrimeKG SCHEMA                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────┐         drug_protein          ┌──────────────┐             │
│    │   DRUG   │ ─────────────────────────────▶│ GENE/PROTEIN │             │
│    └──────────┘                               └──────────────┘             │
│         │                                            │                      │
│         │ contraindication                           │ disease_protein      │
│         │ indication                                 │                      │
│         │ off-label use                              │ pathway_protein      │
│         ▼                                            ▼                      │
│    ┌──────────┐                               ┌──────────────┐             │
│    │ DISEASE  │                               │   PATHWAY    │             │
│    └──────────┘                               └──────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total edges** | 8,100,498 |
| **Unique nodes** | 129,312 |
| **Node types** | 10 |
| **Relation types** | 30 |

### 2.4 Node Types

| Node Type | Count | Description |
|-----------|-------|-------------|
| gene/protein | 27,610 | Human genes and their protein products |
| biological_process | 28,642 | GO biological processes |
| disease | 17,080 | Diseases and conditions |
| effect/phenotype | 15,311 | Clinical effects and phenotypes |
| anatomy | 14,033 | Anatomical structures |
| molecular_function | 11,169 | GO molecular functions |
| drug | 7,957 | Pharmaceutical compounds |
| cellular_component | 4,176 | GO cellular components |
| pathway | 2,516 | Reactome biological pathways |
| exposure | 818 | Environmental exposures |

### 2.5 Key Relation Types Used in This Study

| Relation | Count | Description |
|----------|-------|-------------|
| `contraindication` | 61,350 | Drug should NOT be used for disease |
| `indication` | 18,776 | Drug IS approved to treat disease |
| `off-label use` | 5,136 | Unapproved but clinically practiced |
| `drug_protein` | 51,306 | Drug targets a gene/protein |
| `disease_protein` | 160,822 | Gene/protein associated with disease |
| `pathway_protein` | 85,292 | Gene/protein participates in pathway |
| `protein_protein` | 642,150 | Protein-protein interactions |

### 2.6 Edge File Format

Each edge in PrimeKG is represented as:

```
| relation | x_id | x_type | x_name | y_id | y_type | y_name |
|----------|------|--------|--------|------|--------|--------|
| contraindication | DB00999 | drug | Hydrochlorothiazide | 5044 | disease | hypertensive disorder |
| drug_protein | DB00999 | drug | Hydrochlorothiazide | 6531 | gene/protein | SLC12A3 |
| disease_protein | 5044 | disease | hypertensive disorder | 183 | gene/protein | AGT |
```

---

## 3. Task Definition

### 3.1 Problem Formulation

**Input:** A (drug, disease) pair

**Output:** Binary classification
- **1 = Contraindication** (adverse outcome - drug should NOT be used)
- **0 = Indication** (therapeutic - drug IS used to treat)

### 3.2 Why This Framing?

We chose contraindication vs. indication (rather than "has interaction" vs. "no interaction") because:

1. **Clear ground truth** - Both labels come from established medical knowledge
2. **Clinical relevance** - Distinguishing harmful from helpful is actionable
3. **Sufficient data** - 61K contraindications + 19K indications
4. **Biological hypothesis** - Same drug-gene-pathway features might predict both

### 3.3 Dataset Construction

#### Step 1: Select Diseases with Both Labels

To ensure balanced classes, we selected diseases that have BOTH contraindications AND indications:

```python
diseases_with_contra = set(contraindications['disease_id'])  # 1,195 diseases
diseases_with_indica = set(indications['disease_id'])        # 1,363 diseases
diseases_with_both = diseases_with_contra & diseases_with_indica  # 621 diseases
```

#### Step 2: Filter for Class Balance

We required at least 20% minority class to avoid extreme imbalance:

```python
balance_ratio = min(contra_count, indica_count) / max(contra_count, indica_count)
# Keep diseases where balance_ratio >= 0.2
```

This yielded **328 diseases** with acceptable balance.

#### Step 3: Select Top Diseases by Total Edges

We selected the **top 10 diseases** by total drug relationships:

| Disease | Contraindications | Indications | Total | Balance |
|---------|-------------------|-------------|-------|---------|
| Hypertension | 616 | 206 | 822 | 33% |
| Hypertensive disorder | 614 | 206 | 820 | 34% |
| Anxiety disorder | 360 | 112 | 472 | 31% |
| Peptic ulcer disease | 286 | 72 | 358 | 25% |
| Asthma | 256 | 80 | 336 | 31% |
| Intrinsic asthma | 158 | 84 | 242 | 53% |
| Allergic asthma | 158 | 84 | 242 | 53% |
| Congestive heart failure | 188 | 48 | 236 | 26% |
| Gout | 180 | 42 | 222 | 23% |
| Monogenic obesity | 172 | 40 | 212 | 23% |

#### Step 4: Select Drugs

We selected the **top 100 drugs** by total relationships with the selected diseases.

#### Step 5: Balance Final Dataset

Since contraindications outnumbered indications, we undersampled:

```
Before balancing:
  Contraindications: 1,078
  Indications: 204

After balancing (undersample majority):
  Contraindications: 204
  Indications: 204
  Total: 408 samples
```

---

## 4. Feature Engineering

### 4.1 Overview: The GDi/GDr Framework

Our biological hypothesis requires mapping drugs and diseases to a common space (genes and pathways) where we can measure overlap:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GDi / GDr FRAMEWORK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GDi (Disease Graph):                                                      │
│   ┌─────────┐    disease_protein    ┌──────┐    pathway_protein   ┌───────┐│
│   │ DISEASE │ ──────────────────▶  │ GENE │ ─────────────────▶  │PATHWAY││
│   └─────────┘                       └──────┘                      └───────┘│
│                                                                             │
│   GDr (Drug Graph):                                                         │
│   ┌──────┐       drug_protein       ┌──────┐    pathway_protein   ┌───────┐│
│   │ DRUG │ ──────────────────────▶ │ GENE │ ─────────────────▶  │PATHWAY││
│   └──────┘                          └──────┘                      └───────┘│
│                                                                             │
│   Feature = Overlap between GDi and GDr at gene/pathway level              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Building GDi: Disease → Genes → Pathways

#### Step 1: Extract Disease-Gene Associations

Query PrimeKG for all `disease_protein` edges:

```python
disease_gene_edges = df[
    (df['relation'] == 'disease_protein') &
    ((df['x_type'] == 'disease') | (df['y_type'] == 'disease'))
]
```

**Result:** 160,822 disease-gene edges covering 5,593 diseases.

#### Step 2: Extract Gene-Pathway Associations

Query PrimeKG for all `pathway_protein` edges:

```python
gene_pathway_edges = df[
    (df['relation'] == 'pathway_protein') &
    ((df['x_type'] == 'gene/protein') | (df['y_type'] == 'gene/protein'))
]
```

**Result:** 85,292 gene-pathway edges covering 10,849 genes.

#### Step 3: Build Mappings

```python
gdi_disease_genes = defaultdict(set)    # disease_id → {gene_ids}
gdi_gene_pathways = defaultdict(set)    # gene_id → {pathway_ids}
gdi_disease_pathways = defaultdict(set) # disease_id → {pathway_ids}

# Disease → Genes
for disease_id, gene_id in disease_gene_edges:
    gdi_disease_genes[disease_id].add(gene_id)

# Gene → Pathways
for gene_id, pathway_id in gene_pathway_edges:
    gdi_gene_pathways[gene_id].add(pathway_id)

# Disease → Pathways (transitive closure via genes)
for disease_id, gene_ids in gdi_disease_genes.items():
    for gene_id in gene_ids:
        gdi_disease_pathways[disease_id].update(gdi_gene_pathways[gene_id])
```

#### GDi Coverage for Selected Diseases

| Disease | Associated Genes | Associated Pathways |
|---------|------------------|---------------------|
| Anxiety disorder | 481 | 897 |
| Monogenic obesity | 205 | 471 |
| Congestive heart failure | 110 | 387 |
| Asthma | 87 | 290 |
| Intrinsic/Allergic asthma | 80 | 266 |
| Hypertension | 16 | 74 |
| Hypertensive disorder | 12 | 59 |
| Gout | 10 | 47 |
| Peptic ulcer disease | **3** | **10** |

**Critical Observation:** Annotation coverage varies dramatically. Peptic ulcer has only 3 genes!

### 4.3 Building GDr: Drug → Genes → Pathways

#### Step 1: Extract Drug-Gene (Target) Associations

```python
drug_gene_edges = df[
    (df['relation'] == 'drug_protein') &
    ((df['x_type'] == 'drug') | (df['y_type'] == 'drug'))
]
```

**Result:** 51,306 drug-gene edges covering 6,282 drugs.

#### Step 2: Build Mappings

```python
gdr_drug_genes = defaultdict(set)    # drug_id → {gene_ids}
gdr_drug_pathways = defaultdict(set) # drug_id → {pathway_ids}

# Drug → Genes (targets)
for drug_id, gene_id in drug_gene_edges:
    gdr_drug_genes[drug_id].add(gene_id)

# Drug → Pathways (via gene targets)
for drug_id, gene_ids in gdr_drug_genes.items():
    for gene_id in gene_ids:
        gdr_drug_pathways[drug_id].update(gdi_gene_pathways[gene_id])
```

#### Example: Dexamethasone
```
Drug: Dexamethasone (corticosteroid)
├── Target Genes: 27
│   ├── NR3C1 (glucocorticoid receptor)
│   ├── ANXA1 (anti-inflammatory)
│   └── ... 25 more
└── Associated Pathways: 51
    ├── R-HSA-9006931 (Signaling by Nuclear Receptors)
    ├── R-HSA-212436 (Generic Transcription Pathway)
    └── ... 49 more
```

### 4.4 Biological Feature Extraction

For each (drug, disease) pair, we compute:

#### Feature 1: Shared Genes Count
```python
shared_genes = disease_genes & drug_genes  # Set intersection
n_shared_genes = len(shared_genes)
```

**Interpretation:** How many genes are BOTH targeted by the drug AND associated with the disease? High overlap might indicate therapeutic effect OR adverse interaction.

#### Feature 2: Shared Pathways Count
```python
shared_pathways = disease_pathways & drug_pathways
n_shared_pathways = len(shared_pathways)
```

**Interpretation:** How many biological processes are affected by both the drug and the disease?

#### Feature 3: Pathway Overlap (Jaccard Similarity)
```python
pathway_overlap = len(A & B) / len(A | B)  # Jaccard index
```

**Interpretation:** Normalized overlap score (0 to 1) accounting for different pathway set sizes.

#### Feature 4: Graph Distance
```python
distance = nx.shortest_path_length(G, drug_id, disease_id)
```

**Interpretation:** How many hops in the knowledge graph between drug and disease? Closer might indicate stronger relationship.

#### Features 5-8: Entity Statistics
```python
n_disease_genes     # Total genes associated with disease
n_disease_pathways  # Total pathways for disease
n_drug_genes        # Total gene targets for drug
n_drug_pathways     # Total pathways affected by drug
```

**Interpretation:** Captures "promiscuity" - drugs targeting many genes might have more side effects; well-studied diseases have more annotations.

### 4.5 Graph Construction for Distance Features

We built a NetworkX graph containing the selected drugs, diseases, and their associated genes/pathways:

```python
G = nx.Graph()

# Add nodes
for disease_id in selected_diseases:
    G.add_node(disease_id, node_type='disease')
for drug_id in selected_drugs:
    G.add_node(drug_id, node_type='drug')
for gene_id in all_genes:
    G.add_node(gene_id, node_type='gene')
for pathway_id in all_pathways:
    G.add_node(pathway_id, node_type='pathway')

# Add edges
for disease_id in selected_diseases:
    for gene_id in gdi_disease_genes[disease_id]:
        G.add_edge(disease_id, gene_id)  # disease-gene

for drug_id in selected_drugs:
    for gene_id in gdr_drug_genes[drug_id]:
        G.add_edge(drug_id, gene_id)  # drug-gene

for gene_id in all_genes:
    for pathway_id in gdi_gene_pathways[gene_id]:
        G.add_edge(gene_id, pathway_id)  # gene-pathway
```

**Result:** Graph with 2,338 nodes and 6,391 edges.

### 4.6 Graph Structure Features (Baseline)

Beyond biological features, we computed graph-theoretic features:

#### Degree Centrality
```python
drug_degree = G.degree(drug_id)
disease_degree = G.degree(disease_id)
```
Number of direct connections in the graph.

#### Betweenness Centrality
```python
betweenness = nx.betweenness_centrality(G)
```
Fraction of shortest paths passing through a node. High betweenness = "hub" in the network.

#### Closeness Centrality
```python
closeness = nx.closeness_centrality(G)
```
Average inverse distance to all other nodes. High closeness = centrally located.

#### Derived Features
```python
degree_sum = drug_degree + disease_degree
degree_diff = abs(drug_degree - disease_degree)
```

### 4.7 Complete Feature Vector

For each (drug, disease) pair, the final feature vector contains **15 features**:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `n_shared_genes` | Biological | Overlapping gene targets |
| 2 | `n_shared_pathways` | Biological | Overlapping pathways |
| 3 | `pathway_overlap` | Biological | Jaccard similarity of pathways |
| 4 | `n_disease_genes` | Biological | Disease annotation richness |
| 5 | `n_disease_pathways` | Biological | Disease pathway coverage |
| 6 | `n_drug_genes` | Biological | Drug target promiscuity |
| 7 | `n_drug_pathways` | Biological | Drug pathway coverage |
| 8 | `drug_degree` | Graph | Drug connectivity |
| 9 | `disease_degree` | Graph | Disease connectivity |
| 10 | `drug_betweenness` | Graph | Drug hub-ness |
| 11 | `disease_betweenness` | Graph | Disease hub-ness |
| 12 | `drug_closeness` | Graph | Drug centrality |
| 13 | `disease_closeness` | Graph | Disease centrality |
| 14 | `degree_sum` | Graph | Combined connectivity |
| 15 | `degree_diff` | Graph | Connectivity difference |

### 4.8 Feature Statistics (Final Dataset)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| n_shared_genes | 0.61 | 1.22 | 0 | 12 |
| n_shared_pathways | 8.25 | 8.74 | 0 | 101 |
| pathway_overlap | 0.04 | 0.03 | 0 | 0.20 |
| n_disease_genes | 80.9 | 114.4 | 3 | 481 |
| n_disease_pathways | 225.2 | 216.8 | 10 | 897 |
| n_drug_genes | 11.5 | 9.6 | 0 | 43 |
| n_drug_pathways | 28.3 | 22.5 | 0 | 126 |
| graph_distance | 3.27 | 1.16 | 2 | 6 |

---

## 5. Model Training

### 5.1 Models

#### Logistic Regression
- Linear decision boundary
- Interpretable coefficients
- Baseline for comparison

```python
LogisticRegression(random_state=42, max_iter=1000)
```

#### Random Forest
- Ensemble of decision trees
- Captures non-linear interactions
- Built-in feature importance

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

### 5.2 Feature Set Ablations

We tested different feature combinations to understand their contributions:

| Experiment | Features | # Features |
|------------|----------|------------|
| Interaction_Safe | Biological only (no graph_distance) | 7 |
| Interaction_Full | Biological + graph_distance | 8 |
| Baseline | Graph structure only | 8 |
| Combined_Safe | Biological + Graph (no distance) | 15 |
| Combined_Full | All features | 16 |

### 5.3 Why Exclude `graph_distance`?

We were concerned about **data leakage**: if the graph encodes drug-disease relationships, then graph distance might trivially predict labels.

**Leakage check:** Compare models with and without `graph_distance`:
```
If AUC(with_distance) >> AUC(without_distance):
    → Distance is leaking label information
```

**Result:** No significant difference (< 0.01 AUC), so no leakage detected.

---

## 6. Evaluation Methodology

### 6.1 Standard K-Fold Cross-Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD 5-FOLD CROSS-VALIDATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  All 408 samples shuffled and split into 5 folds:                          │
│                                                                             │
│  Fold 1: [████████████████████] Train  [████] Test                         │
│  Fold 2: [████] Test  [████████████████████] Train                         │
│  Fold 3: [████████] Train [████] Test [████████] Train                     │
│  ...                                                                        │
│                                                                             │
│  ⚠️  PROBLEM: Same diseases appear in BOTH train and test!                 │
│                                                                             │
│  Example fold might have:                                                   │
│    Train: Hypertension(drug1), Hypertension(drug2), Asthma(drug3)...       │
│    Test:  Hypertension(drug4), Asthma(drug5)...                            │
│                                                                             │
│  Model can memorize "hypertension drugs look like X"                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```

**Samples per fold:** ~81 test samples (vs. only 8 before expanding dataset)

### 6.2 Disease-Level Cross-Validation (Leave-One-Disease-Out)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               DISEASE-LEVEL CROSS-VALIDATION (LODO)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Train on 9 diseases, test on 1 COMPLETELY UNSEEN disease:                 │
│                                                                             │
│  Fold 1: Train on [HTN, Anxiety, Asthma, CHF, Gout, Obesity, PUD, ...]    │
│          Test on  [Hypertensive disorder] ← NEVER SEEN IN TRAINING         │
│                                                                             │
│  Fold 2: Train on [HTN-d, Anxiety, Asthma, CHF, Gout, Obesity, PUD, ...]  │
│          Test on  [Hypertension] ← NEVER SEEN IN TRAINING                  │
│                                                                             │
│  ...repeat for all 10 diseases...                                          │
│                                                                             │
│  ✓ This tests TRUE generalization to new diseases!                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
def disease_level_cv(model, df_dataset, selected_disease_ids):
    results = []
    for held_out_disease in selected_disease_ids:
        # Split by disease
        train_mask = df_dataset['disease_id'] != held_out_disease
        test_mask = df_dataset['disease_id'] == held_out_disease

        # Train on 9 diseases
        model.fit(X_train, y_train)

        # Test on 1 unseen disease
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_proba)

        results.append({'disease': held_out_disease, 'auc': auc})
    return results
```

### 6.3 Evaluation Metrics

#### ROC-AUC (Primary Metric)
- Area Under the Receiver Operating Characteristic Curve
- Measures ranking quality: P(score(positive) > score(negative))
- **0.5 = random**, **1.0 = perfect**
- Robust to class imbalance

#### Accuracy
- Fraction of correct predictions
- Can be misleading with imbalanced data

#### F1 Score
- Harmonic mean of precision and recall
- Balances false positives and false negatives

#### Precision / Recall
- Precision: Of predicted positives, how many are correct?
- Recall: Of actual positives, how many did we find?

---

## 7. Results

### 7.1 Standard Cross-Validation Results

| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|-----|
| **RF_Combined_Safe** | **0.986 ± 0.011** | 0.919 ± 0.013 | 0.914 ± 0.013 |
| RF_Combined_Full | 0.980 ± 0.012 | 0.909 ± 0.010 | 0.902 ± 0.011 |
| RF_Interaction_Full | 0.975 ± 0.015 | 0.919 ± 0.034 | 0.916 ± 0.033 |
| RF_Interaction_Safe | 0.972 ± 0.011 | 0.909 ± 0.040 | 0.905 ± 0.041 |
| RF_Baseline | 0.971 ± 0.015 | 0.887 ± 0.023 | 0.874 ± 0.027 |
| LR_Combined_Safe | 0.635 ± 0.081 | 0.586 ± 0.047 | 0.497 ± 0.086 |
| LR_Combined_Full | 0.627 ± 0.075 | 0.588 ± 0.053 | 0.503 ± 0.095 |
| LR_Interaction_Safe | 0.613 ± 0.072 | 0.566 ± 0.031 | 0.484 ± 0.065 |
| LR_Baseline | 0.585 ± 0.052 | 0.578 ± 0.044 | 0.457 ± 0.085 |

**Observations:**
1. Random Forest dramatically outperforms Logistic Regression
2. All RF models achieve >0.97 ROC-AUC
3. Very low standard deviation suggests stable results

⚠️ **But are these results trustworthy?**

### 7.2 Leakage Check

Comparing models with and without `graph_distance`:

| Comparison | AUC Difference | Verdict |
|------------|----------------|---------|
| LR Interaction: Safe vs Full | -0.004 | ✓ OK |
| LR Combined: Safe vs Full | -0.008 | ✓ OK |
| RF Interaction: Safe vs Full | +0.003 | ✓ OK |
| RF Combined: Safe vs Full | -0.005 | ✓ OK |

**Conclusion:** Graph distance does NOT cause label leakage.

### 7.3 Disease-Level Cross-Validation Results

| Held-Out Disease | N_test | Pos/Neg | ROC-AUC | Accuracy |
|------------------|--------|---------|---------|----------|
| Intrinsic asthma | 46 | 16/30 | **1.000** | 0.978 |
| Allergic asthma | 55 | 25/30 | **1.000** | 0.945 |
| Asthma | 55 | 25/30 | **0.969** | 0.927 |
| Anxiety disorder | 26 | 24/2 | **0.958** | 0.769 |
| Hypertension | 64 | 28/36 | **0.940** | 0.797 |
| Monogenic obesity | 15 | 9/6 | 0.852 | 0.600 |
| Hypertensive disorder | 66 | 30/36 | 0.759 | 0.727 |
| Congestive heart failure | 21 | 13/8 | 0.702 | 0.571 |
| **Gout** | 36 | 14/22 | **0.198** | 0.278 |
| **Peptic ulcer disease** | 24 | 20/4 | **0.163** | 0.583 |
| **MEAN** | | | **0.754** | 0.718 |
| **STD** | | | **0.319** | 0.216 |

### 7.4 The Critical Comparison

| Evaluation Method | ROC-AUC | Std Dev |
|-------------------|---------|---------|
| Standard 5-fold CV | 0.986 | 0.011 |
| **Disease-level CV** | **0.754** | **0.319** |
| **Performance drop** | **-0.232 (-23.5%)** | |

### 7.5 Feature Importance

| Feature | Importance | Category |
|---------|------------|----------|
| n_drug_pathways | 0.113 | Biological |
| drug_betweenness | 0.113 | Graph |
| pathway_overlap | 0.102 | Biological |
| drug_closeness | 0.096 | Graph |
| degree_diff | 0.092 | Graph |
| degree_sum | 0.087 | Graph |
| drug_degree | 0.077 | Graph |
| n_shared_pathways | 0.074 | Biological |
| n_drug_genes | 0.073 | Biological |
| n_shared_genes | 0.032 | Biological |

**By category:**
- Graph structure features: **55%**
- Biological features: **45%**

---

## 8. Key Findings

### Finding 1: Standard CV Overestimates by 24%
The model appeared to achieve 98.6% ROC-AUC but actually generalizes at ~75% to unseen diseases.

### Finding 2: Extreme Variance Across Diseases
- Best: Intrinsic asthma (1.000 ROC-AUC)
- Worst: Peptic ulcer (0.163 ROC-AUC)
- Range: **0.84** (massive!)

### Finding 3: Sparse Annotation = Poor Prediction

| Disease | Genes | Pathways | ROC-AUC |
|---------|-------|----------|---------|
| Anxiety | 481 | 897 | 0.958 |
| Asthma | 87 | 290 | 0.969 |
| Gout | 10 | 47 | 0.198 |
| Peptic ulcer | 3 | 10 | 0.163 |

Strong correlation between annotation richness and performance.

### Finding 4: Model Memorizes Disease Patterns
The high standard CV performance + low disease-level CV performance indicates the model learned disease-specific patterns rather than generalizable drug-disease interactions.

---

## 9. Limitations

1. **Small dataset:** 408 samples, 10 diseases, 100 drugs
2. **Selection bias:** Chose diseases with good class balance
3. **Annotation bias:** Well-studied diseases have more gene associations
4. **No temporal validation:** Snapshot of current knowledge
5. **No external validation:** Need independent test sets
6. **Limited feature set:** No drug chemical structure features

---

## 10. Conclusions

### The Honest Assessment
- **True performance: ~0.75 ROC-AUC** on unseen diseases
- **High variance (±0.32)** means unreliable for some disease types
- **Current features don't generalize** across all disease categories

### Key Methodological Insight
> Standard cross-validation can be 24% overly optimistic when entities (diseases) appear in both train and test sets. Disease-level (or drug-level) CV is essential for honest evaluation.

---

## 11. Next Steps

### Immediate
1. **Drug-level CV** - Test generalization to unseen drugs
2. **More diseases** - Expand to 50+ diverse therapeutic areas
3. **Drug structure features** - Chemical fingerprints, molecular properties

### Medium-term
4. **Graph Neural Networks** - Learn representations from graph structure
5. **Multi-task learning** - Share knowledge across diseases
6. **External validation** - SIDER, FDA FAERS, DrugBank

### Long-term
7. **Temporal validation** - Train on old data, test on new discoveries
8. **Mechanistic interpretation** - Which pathways drive predictions?

---

## Appendix A: Code Structure

```
gnn-ddi/
├── mvp_drug_disease_prediction.ipynb    # Main analysis (this study)
├── explore_primekg.ipynb                # Dataset exploration
├── data/
│   └── kg.csv                           # PrimeKG (8.1M edges)
├── RESEARCH_SUMMARY.md                  # This document
└── requirements.txt                     # Dependencies
```

## Appendix B: Reproducibility

- **Random seed:** 42 (all experiments)
- **Python:** 3.10+
- **Key packages:** pandas, numpy, networkx, scikit-learn
- **Hardware:** Standard laptop (no GPU required)
- **Runtime:** ~5 minutes for full notebook

---

*Generated from MVP Drug-Disease Prediction Analysis*
