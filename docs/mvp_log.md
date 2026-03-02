MVP 1:
Started with 10 diseases with known adverse outcomes, expanded to all 328 diseases; started with 20 drugs (related to chosen diseases), expanded to 1000 drugs
Build GDi (disease → genes → pathways) and GDr mappings (drug → genes → pathways)
Interaction features: for each drug-disease pair, extract shared genes count, shared pathways count, pathway coverage overlap, graph distance between drug and disease nodes
Baseline: simple node embeddings for PrimeKG using graph structure (node2vec and graph structure features)
Train logistic regression and random forest models on (1) interaction features only, (2) baseline embeddings only, (3) combined features
Experiments with 5-fold cross-validation performed very well for RF, with baseline .977, .984 for RF combined
Data leaks – generalizing patterns for diseases and drugs 
Leave-one-disease-out test (test on unseen diseases): dropped to .831 for RF combined
For feature importance, graph structure is generally more important than biological features
Key issues: PrimeKG embedding baseline is super high, data leakage from embeddings (test disease and drugs show up in the embeddings), interaction features are built from same data as PrimeKG (enriching with interaction features doesn’t meaningfully add much)

MVP 2:
Goals: more formally separate experiments with different levels of data, remove data leakage, lower baseline
Setup:
Graph A: PrimeKG without bio edges (bipartite drugs and diseases – should perform .5)
Graph B: 1-hop subgraph of PrimeKG around selected drugs/diseases (all node/edge types)
Graph C: PrimeKG with DisGeNET disease-gene edges
Three link prediction methods: heuristic scores, Node2Vec embeddings, combined
Same 328 diseases, same top 1000 dugs
Iterations (goal: get rid of leakage & lower baseline):
Compute heuristics with target edges removed, train node embeddings
First iteration had massive leakage – embeddings trained on target edges
GroupKFold evaluation (no disease appears both in train and test, with 5-fold CV where diseases are the grouping variable)
Graph A outperformed B and C in this test
Found that drug-frequency (contraindication rate across training data) performs .870

MVP 3:
Goals: test whether external knowledge graph sources (beyond PrimeKG) add predictive signal
Setup:
Graph A: PrimeKG 1-hop base (same as v2 Graph B, the strong baseline)
Graph B: A + DISEASES db (Jensen Lab, 7,772 disease-gene rows → 4,836 new edges)
Graph C: A + Reactome (153K human gene-pathway rows → 134,198 new edges, largest augmentation)
Graph D: A + PharmGKB (21,972 gene-chemical rows → 7,331 new edges)
Graph E: A + all three sources combined (+146,365 edges total)
Key methodological fix vs v2: Node2Vec retrained fresh inside every CV fold and every LODO split (25 training runs for CV, 250 for LODO) – eliminates indirect embedding leakage through test node representations
Results:
5-fold CV: all graphs tightly clustered at 0.993–0.994 AUC; external sources show minimal effect on this metric
LODO (50 diseases): Graph A=0.738±0.322, B=0.737±0.311, C=0.698±0.308, D=0.745±0.288, E=0.731±0.327
LODO is lower than v2's 0.831 – more honest because within-fold N2V eliminates the indirect leakage that was inflating v2's LODO
Key findings:
No external source meaningfully improves 5-fold CV (all ~0.994); PrimeKG already overlaps heavily with these sources
Reactome actually hurts LODO performance (-0.040 vs baseline A), likely because 134K new pathway nodes dilute the walk distribution and make embeddings less drug/disease-specific
PharmGKB gives the best LODO (+0.007 vs A) and lowest variance (0.288 vs 0.322) – drug-gene pharmacogenomics edges are the most useful augmentation tested
Combining all three sources (Graph E) doesn't beat PharmGKB alone; more data ≠ better signal when sources overlap or add noise
High LODO variance (±0.288–0.327) is the central remaining problem – some diseases still completely fail (AUC ~0.0) while others are near-perfect

GNN v1:
Goals: test whether a heterogeneous GNN learning from typed edge structure outperforms RF + Node2Vec
Architecture: 2-layer HeteroConv with SAGEConv per edge type (80 edge types incl. reverse), learnable 64-dim node embeddings per type (10 types), MLP link prediction head on [drug_emb || disease_emb]
Graph: same 1-hop PrimeKG subgraph as v3 Graph A; contraindication/indication/off-label edges fully excluded from message passing
Evaluation: 5-fold GroupKFold by disease (no disease in both train and test) — same disease-grouped logic as run_evaluation.py but 5 folds rather than LODO-50 because each fold requires a full GNN training run (~2 min/fold on CPU)
Results:
  Fold 1: AUC=0.772  Fold 2: AUC=0.782  Fold 3: AUC=0.762  Fold 4: AUC=0.793  Fold 5: AUC=0.818
  GNN GroupKFold: AUC=0.785 ± 0.019
  Baseline (v3 LODO RF+N2V): AUC=0.738 ± 0.322
  Δ AUC: +0.047
Key findings:
GNN improves mean AUC by +0.047 over the RF+Node2Vec baseline
The variance drop (±0.019 vs ±0.322) is the more exciting result – the GNN generalizes much more consistently across disease groups; v3's high variance was likely driven by individual diseases failing completely in LODO, which gets smoothed across ~65 diseases per fold in GroupKFold
Loss was still declining at epoch 50 (final ~0.52) – more epochs or a scheduler likely improve results further
320 pairs had missing node mappings (small indexing issue to fix)
Note on eval comparability: GroupKFold-5 (65 diseases/fold) vs LODO-50 (1 disease/fold) differ in granularity; GNN variance would likely be higher under strict LODO
