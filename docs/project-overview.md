## 1. Original Mandate / Research Question

**High-level question:** Given a pre-existing condition or genetic disease, can we predict a drug outcome — specifically, whether there is concern for a worse or adverse outcome when a patient takes a drug?

**Technical formulation:** Can we predict adverse or altered drug outcomes by modeling the interaction between drug-affected biological pathways and disease-perturbed pathways using graph representations?

The framing distinguishes this from standard drug repurposing work (e.g., TxGNN) by treating disease and drug not as static KG entities but as *interacting biological perturbation systems*:

- **GDi** — disease-perturbed pathway graph: how a genetic disease distorts normal biology (nodes: genes/proteins, pathways; edges: PPI, regulatory, signaling; attributes: mutation sites, expression dysregulation)
- **GDr** — drug-affected pathway graph: how a drug perturbs biology when administered (nodes: target proteins, downstream proteins; edges: drug-target binding, downstream signaling; attributes: binding affinity, agonist vs antagonist)

The hypothesis is that adverse outcome risk emerges from the *interaction* of these two systems — shared subgraphs, overlapping pathway perturbations, cross-graph signal — rather than from simple graph proximity between a drug node and a disease node.

**Differentiation from prior work:**
- *TxGNN*: uses PrimeKG for zero-shot drug repurposing; treats diseases as static entities, does not model pathway-level perturbation interaction
- *HyperADR*: models drug-gene-ADR associations via hypergraph, but does not represent pre-existing genetic disease as a pathway-level perturbation

---

## 2. Initial Experiments and Research Question Validation

A series of MVPs was run to validate whether the research question was tractable and to establish honest baselines before committing to GNN architectures.

### MVP 1 — Feature Engineering on PrimeKG
- **Setup:** 328 diseases, 1000 drugs; binary classification (contraindication=1, indication/no-relation=0); built GDi and GDr subgraphs from PrimeKG; extracted interaction features (shared genes, shared pathways, pathway overlap, graph distance); trained logistic regression and random forest
- **Results:** 5-fold CV AUC=0.984 (RF combined); LODO AUC=0.831
- **Problems identified:**
  - Severe data leakage — embeddings were trained on target edges; test drug/disease nodes leaked into training graph
  - Graph structure features dominated over biological interaction features (contradicting the core hypothesis)
  - PrimeKG interaction features were not additive over the baseline (same underlying data)
  - High baseline inflated by drug frequency (contraindication rate per drug)

### MVP 2 — Link Prediction with Leakage Controls
- **Goals:** eliminate leakage, formally separate experiments by graph type, lower baseline
- **Setup:** Three graphs — Graph A (bipartite drug-disease only, expected ~0.5), Graph B (1-hop PrimeKG subgraph), Graph C (PrimeKG + DisGeNET); Node2Vec retrained inside each fold
- **Key finding:** Graph A (no biology) outperformed Graph B and C under strict GroupKFold — the biological edges were not improving over the pure graph structure baseline
- **Methodological fixes established:** GroupKFold grouped by disease; drug frequency normalization (balanced pos/neg per drug); target edges fully excluded from message-passing graphs
- **Final honest LODO numbers:** ~0.724 for new drug + new disease

### MVP 3 — External Knowledge Graph Augmentation
- **Goals:** determine whether external sources (DisGeNET, Reactome, PharmGKB) add meaningful signal beyond PrimeKG
- **Setup:** 5 graph variants — PrimeKG 1-hop base (A), + DISEASES db (B), + Reactome (C), + PharmGKB (D), + all combined (E); Node2Vec retrained fresh inside every fold and every LODO split
- **Results (LODO-50):**
  - Graph A (baseline): AUC=0.738 ± 0.322
  - Graph C (+ Reactome): AUC=0.698 — *worse* than baseline
  - Graph D (+ PharmGKB): AUC=0.745, std=0.288 — best single augmentation
  - Graph E (all combined): AUC=0.731 — does not beat PharmGKB alone
- **Key validation of research question:** Reactome hurt performance because dumping 134K raw gene-pathway edges dilutes walk distributions; raw graph size ≠ signal. PharmGKB's drug-gene pharmacogenomics edges are the most useful augmentation. This confirmed that *how* biological knowledge is integrated matters more than volume.
- **Central remaining problem:** High LODO variance (±0.288–0.322); some diseases fail completely (AUC~0.0) while others are near-perfect. This drove the move to GNNs.

### GNN v1 — Heterogeneous GNN Baseline
- **Architecture:** 2-layer HeteroConv with SAGEConv per edge type (80 typed edge types including reverse); learnable 64-dim node embeddings per type (10 node types); MLP link prediction head on [drug_emb || disease_emb]
- **Graph:** same 1-hop PrimeKG subgraph as v3 Graph A; target edges excluded from message-passing
- **Results (5-fold GroupKFold):** AUC=0.785 ± 0.019
- **Key finding:** GNN improves mean AUC by +0.047, but more importantly drops variance from ±0.322 to ±0.019 — the GNN generalizes much more consistently across held-out disease groups than RF+N2V

---

## 3. Confirmed Research Question and Plan

The MVP progression confirmed that the problem is real and tractable, and produced two specific refinements to the research question:

**Confirmed research question:**
> Can a heterogeneous GNN trained on a biologically enriched knowledge graph predict whether a drug is contraindicated for a patient with a given disease, generalizing to unseen diseases?

**The disease-only vs. genetic-information question:**

The experiments surface a fundamental fork in the project direction:

- **Disease-only formulation (current):** The model uses disease as a node in PrimeKG and learns from disease-gene, disease-pathway, drug-target, and drug-pathway edges. It does not incorporate genetic variants or patient-specific pharmacogenomic information. The label is at the (drug, disease) pair level.

- **Genetic/pharmacogenomic formulation (proposed extension):** Incorporate PharmGKB/ClinPGx to model at the (drug, variant/gene) level — i.e., given a patient's genetic background, predict altered drug metabolism or increased adverse risk. This shifts the task from disease-level contraindication prediction to patient-level pharmacogenomic risk prediction.

**Why this matters:** The MVP results showed that PharmGKB drug-gene edges (the genetic signal) gave the best LODO improvement and lowest variance among all augmentation sources tested. This is evidence that genetic information contains signal the disease-level graph does not capture. However, integrating PharmGKB as supervision labels (rather than graph edges) requires a different label schema and unit of analysis.

**Current plan:** Complete architecture search at the disease-level formulation first (the current multi-architecture pipeline). Once architecture is finalized, conduct a data ablation to quantify what genetic/pharmacogenomic signal adds — using PharmGKB edges as graph augmentation vs. as labels.

---

## 4. Current Architecture Setup

The codebase is organized as a modular pipeline (`pipeline/`) that trains and evaluates three GNN architectures against the same data and evaluation protocol in a single run.

**Entry point:**
```
python run_pipeline.py --config config/default.yaml --eval group_kfold --models hetero_gnn,dual_encoder,metapath
```

**Data:**
- PrimeKG (`data/kg.csv`, ~4M edges) as the backbone KG
- 328 diseases × 1000 drugs; contraindication=1, indication=0
- Target relation edges (contraindication, indication, off-label use) excluded from message-passing graph
- Additional sources configurable in `config/default.yaml` (PharmGKB, DISEASES db, Reactome — currently disabled)

**Graph construction (`pipeline/data/graph_builder.py`):**
- Builds PyG `HeteroData` with typed node/edge sets; reverse edges added for undirected message-passing
- Zero-degree drug/disease nodes explicitly initialized so all dataset entities have learnable embeddings
- Metapath adjacency matrices precomputed via sparse matrix multiplication for the MetapathModel

**Three architectures (`pipeline/models/`):**

| Model | Architecture | Current AUC (GroupKFold-5) |
|---|---|---|
| `HeteroGNN` | 2-layer HeteroConv + SAGEConv per edge type; MLP link head on [drug \|\| disease] | 0.751 ± 0.018 |
| `DualEncoder` | Two independent HeteroConv stacks (drug encoder / disease encoder) + cross-attention; MLP on [drug \|\| disease \|\| attended_drug \|\| attended_disease] | **0.774 ± 0.012** |
| `MetapathModel` | HAN-style semantic attention over precomputed metapath adjacency matrices (drug→gene→disease; drug→gene→BP→gene→disease) | 0.550 ± 0.006 (config bug fixed, needs re-run) |

All models share: 128-dim embeddings, 128-dim hidden, 2 layers, cosine LR scheduler, BCEWithLogitsLoss with pos_weight, Youden's J threshold selection.

**Evaluation (`pipeline/evaluation/evaluator.py`):**
- **GroupKFold-5:** 5 folds grouped by disease — no disease appears in both train and test; primary metric for architecture comparison
- **LODO-50:** 50 sampled diseases each held out individually; GNN retrained from scratch per split; directly comparable to MVP v3 baseline (0.738 ± 0.322); checkpointed for resumability

**Results directory:** `results/` — per-model pkl files; `results/analysis/` — summary CSV, boxplot, loss curves, embedding PCA plots

---

## 5. Open Questions

**Architecture:**
- MetapathModel has not yet produced a valid result (config bug fixed in March 2026; re-run pending). The core question is whether explicit biological pathway traversal (drug→gene→disease) adds signal over the end-to-end GNN learning of the same structure implicitly.
- DualEncoder (best so far at 0.774) uses cross-attention between drug and disease embeddings — this is the closest implementation to the original GDi/GDr interaction hypothesis. Worth investigating whether performance gap vs. HeteroGNN (0.751) holds under LODO.
- All three architectures are still below the RF+N2V LODO mean AUC (0.738) but with dramatically lower variance — it is unclear whether GroupKFold-5 and LODO-50 are directly comparable enough to make this claim confidently.

**Data:**
- The disease-only vs. genetic-information question is unresolved. PharmGKB gave the best signal as graph edges in v3 — does it also help as a supervision signal or richer node feature in the GNN?
- Reactome hurt v3 (raw edge dump). The future experiment plan calls for using Reactome to compute shared-pathway features as explicit edge attributes rather than adding raw edges — this has not been run yet.
- Disease nodes have sparse neighborhoods in the current 1-hop graph. Augmenting with explicit disease→GO biological process edges (by walking disease→gene→BP in PrimeKG) may improve disease representation.

**Evaluation:**
- LODO variance (±0.322 for RF+N2V) is the key unresolved issue — some diseases fail completely (AUC ~0.0). Understanding which diseases fail and why (data sparsity, graph position, label noise) is critical before claiming generalization.
- Drug-side generalization has not been measured. A leave-one-drug-out experiment would test whether the model can generalize to new drugs entering the market with no KG history.

**Research question scope:**
- The genetic/pharmacogenomic extension (predicting at the (drug, variant) level using ClinPGx) is a natural next chapter but requires a different label schema and has not been started. The current work needs to be complete and written up before this becomes the focus.
- The paper needs to avoid the "TxGNN for side effects" framing — the differentiation is the pathway-interaction hypothesis. The open question is whether the final results actually support that hypothesis or whether the GNN is learning something simpler (e.g., graph proximity).
