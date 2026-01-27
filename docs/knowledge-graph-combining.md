# Knowledge Graph Combining: PrimeKG + PharmGKB

This document explains how the `combine_knowledge_graphs.py` script merges PrimeKG and PharmGKB into a unified knowledge graph for GNN-based drug-drug interaction prediction.

## Overview

| Source | Nodes | Edges | Primary Use |
|--------|-------|-------|-------------|
| PrimeKG | 129,375 | 8,100,498 | Broad biomedical relationships |
| PharmGKB | 12,537 | 127,600 | Pharmacogenomic associations |
| **Combined** | **139,375** | **8,228,098** | Unified graph for GNN training |

## The Challenge: Different ID Systems

PrimeKG and PharmGKB use incompatible identifier systems:

| Entity | PrimeKG IDs | PharmGKB IDs |
|--------|-------------|--------------|
| Genes | NCBI Gene IDs (e.g., `675`) | Accession IDs (e.g., `PA25410`) |
| Drugs | DrugBank IDs (e.g., `DB00945`) | Accession IDs (e.g., `PA450085`) |
| Diseases | MONDO IDs | Accession IDs |

Direct ID matching yields near-zero overlap. The solution: **name-based entity alignment**.

---

## Step 1: Entity Alignment via Gene Name Matching

### Why Genes?

Gene symbols (BRCA1, TP53, CYP2D6) are standardized across databases by HUGO Gene Nomenclature Committee (HGNC). This makes string matching reliable.

### The Algorithm

```python
def normalize_name(name):
    """Normalize gene names for matching."""
    return name.upper().strip()

# Example
primekg_name = "BRCA1"      # From PrimeKG
pharmgkb_name = "brca1 "    # From PharmGKB (lowercase, trailing space)

normalize_name(primekg_name)   # "BRCA1"
normalize_name(pharmgkb_name)  # "BRCA1"
# Match!
```

### Results

- **PrimeKG genes**: 27,671
- **PharmGKB genes**: 2,501
- **Matched**: 2,465 (98.6% of PharmGKB genes)

### Why Not Match Drugs?

Drug naming is inconsistent:
- Brand names: "Tylenol" vs "Panadol"
- Generic names: "acetaminophen" vs "paracetamol"
- Chemical names: "N-(4-hydroxyphenyl)acetamide"

Reliable drug matching requires external mappings (DrugBank, RxNorm) which adds complexity. For this implementation, drugs remain separate by source.

---

## Step 2: Unified Node Registry

Each node receives a unified ID while preserving original identifiers for traceability.

### Node Schema

```python
{
    "unified_id": "gene/protein::BRCA1",
    "node_type": "gene/protein",
    "name": "BRCA1",
    "primekg_id": "675",        # NCBI Gene ID (if from PrimeKG)
    "pharmgkb_id": "PA25410",   # PharmGKB accession (if from PharmGKB)
    "source": "Both"            # "PrimeKG", "PharmGKB", or "Both"
}
```

### Unified ID Format

```
{node_type}::{identifier}
```

Examples:
- `gene/protein::BRCA1`
- `drug::Aspirin`
- `disease::Breast Cancer`
- `variant::rs1234567`

### Source Distribution

| Source | Node Count | Description |
|--------|------------|-------------|
| PrimeKG only | 126,850 | Nodes unique to PrimeKG |
| PharmGKB only | 10,063 | Nodes unique to PharmGKB (variants, haplotypes) |
| Both | 2,462 | Matched genes present in both |

---

## Step 3: Relation Type Standardization

Different databases use different terminology for similar relationships. The script maps these to a unified schema.

### Mapping Table

| Source | Original Relation | Unified Relation |
|--------|-------------------|------------------|
| PrimeKG | `protein_protein` | `gene_gene_interaction` |
| PrimeKG | `drug_protein` | `drug_gene_target` |
| PrimeKG | `indication` | `drug_disease_indication` |
| PrimeKG | `contraindication` | `drug_disease_contraindication` |
| PrimeKG | `off-label use` | `drug_disease_offlabel` |
| PharmGKB | `associated` | `gene_drug_associated` |
| PharmGKB | `ambiguous` | `gene_drug_ambiguous` |
| PharmGKB | `not associated` | `gene_drug_no_association` |
| PharmGKB | `pharmacodynamic_association` | `gene_drug_pharmacodynamic` |
| PharmGKB | `pharmacokinetic_association` | `gene_drug_pharmacokinetic` |

### Why Standardize?

1. **Consistent GNN input**: Same relation type gets same embedding
2. **Queryability**: Find all drug-gene relationships regardless of source
3. **Interpretability**: Clear semantics for downstream analysis

---

## Step 4: Edge Merging with Provenance

### Edge Schema

```python
{
    "head": "gene/protein::CYP2D6",
    "tail": "drug::Codeine",
    "relation": "gene_drug_pharmacokinetic",
    "source_kg": "PharmGKB"
}
```

### Handling Duplicates

When the same edge exists in both sources, **both are kept**:

```python
# Edge from PrimeKG
{"head": "gene/protein::BRCA1", "tail": "drug::Olaparib",
 "relation": "drug_gene_target", "source_kg": "PrimeKG"}

# Edge from PharmGKB (same entities, same relation)
{"head": "gene/protein::BRCA1", "tail": "drug::Olaparib",
 "relation": "gene_drug_associated", "source_kg": "PharmGKB"}
```

**Rationale**: Independent evidence from multiple sources strengthens confidence. For GNN training, you can:
- Weight edges by source count
- Use source as an edge feature
- Filter to high-confidence (multi-source) edges

---

## Step 5: NetworkX Graph Construction

### Graph Type: MultiDiGraph

```python
G = nx.MultiDiGraph()
```

| Property | Meaning |
|----------|---------|
| **Multi** | Multiple edges allowed between same node pair |
| **Di** | Directed edges (head → tail) |
| **Graph** | Standard graph structure |

### Why MultiDiGraph?

The same gene-drug pair can have multiple relationship types:

```
CYP2D6 --[pharmacokinetic]--> Codeine
CYP2D6 --[pharmacodynamic]--> Codeine
```

A simple graph would collapse these into one edge, losing information.

### Node Attributes

```python
G.add_node(
    "gene/protein::BRCA1",
    node_type="gene/protein",
    name="BRCA1",
    primekg_id="675",
    pharmgkb_id="PA25410",
    source="Both"
)
```

### Edge Attributes

```python
G.add_edge(
    "gene/protein::CYP2D6",
    "drug::Codeine",
    relation="gene_drug_pharmacokinetic",
    source_kg="PharmGKB"
)
```

---

## Output Files

| File | Size | Description |
|------|------|-------------|
| `combined_kg.csv` | ~600MB | Edge list with unified IDs |
| `combined_nodes.csv` | ~8MB | Node registry with all attributes |
| `combined_graph.pkl` | ~200MB | Serialized NetworkX graph |
| `id_mappings.json` | ~150KB | Cross-reference lookup tables |
| `merge_report.txt` | ~2KB | Statistics and quality metrics |

### CSV Schemas

**combined_kg.csv**
```
head,tail,relation,source_kg
gene/protein::BRCA1,drug::Olaparib,gene_drug_associated,PharmGKB
```

**combined_nodes.csv**
```
unified_id,node_type,name,primekg_id,pharmgkb_id,source
gene/protein::BRCA1,gene/protein,BRCA1,675,PA25410,Both
```

---

## Combined Graph Statistics

```
Total nodes:        139,375
Total edges:        8,228,098
Connected components: 4
Largest component:  139,367 (99.99%)
Average degree:     59.02
Max degree:         17,369
```

### Node Types

| Type | Count | Source |
|------|-------|--------|
| biological_process | 28,642 | PrimeKG |
| gene/protein | 27,643 | Both |
| disease | 17,841 | Both |
| effect/phenotype | 15,311 | PrimeKG |
| anatomy | 14,033 | PrimeKG |
| molecular_function | 11,169 | PrimeKG |
| drug | 9,428 | Both |
| **variant** | **7,159** | PharmGKB only |
| cellular_component | 4,176 | PrimeKG |
| pathway | 2,516 | PrimeKG |
| exposure | 818 | PrimeKG |
| **haplotype** | **639** | PharmGKB only |

### Top Relation Types

| Relation | Count |
|----------|-------|
| anatomy_protein_present | 3,036,406 |
| drug_drug | 2,672,628 |
| gene_gene_interaction | 642,150 |
| disease_phenotype_positive | 300,634 |
| gene_drug_associated | 51,910 |
| gene_drug_pharmacodynamic | 22,560 |

---

## Usage for GNN Training

### Loading the Graph

```python
import pickle
import pandas as pd

# Option 1: NetworkX graph
with open('data/combined/combined_graph.pkl', 'rb') as f:
    G = pickle.load(f)

# Option 2: Edge/node DataFrames (for PyTorch Geometric)
edges = pd.read_csv('data/combined/combined_kg.csv')
nodes = pd.read_csv('data/combined/combined_nodes.csv')
```

### Converting to PyTorch Geometric

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Add node types
for node_type in nodes['node_type'].unique():
    mask = nodes['node_type'] == node_type
    data[node_type].num_nodes = mask.sum()

# Add edge types
for relation in edges['relation'].unique():
    mask = edges['relation'] == relation
    edge_df = edges[mask]
    # Map to node indices and add to data...
```

### Research Applications

The combined graph enables paths like:

```
Drug → Gene → Variant → Disease
       ↓
    Pathway → Biological Process
```

This supports the research question: *"Can we predict adverse or altered drug outcomes by modeling the interaction between drug-affected biological pathways and disease-perturbed pathways?"*

---

## Files

- Script: `combine_knowledge_graphs.py`
- Notebook: `combine_graphs.ipynb`
- Output: `data/combined/`
