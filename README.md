# GNN-DDI: Graph Neural Networks for Drug-Drug Interaction Prediction

A research project that combines **PrimeKG** and **PharmGKB** biomedical knowledge graphs into a unified graph optimized for training Graph Neural Networks to predict drug-drug interactions and related biomedical relationships.

## Research Question

> *Can we predict adverse or altered drug outcomes by modeling the interaction between drug-affected biological pathways and disease-perturbed pathways?*

## Features

- **Knowledge Graph Integration**: Merges PrimeKG (8.1M edges) with PharmGKB (127K edges) into a unified graph
- **Smart Entity Alignment**: Uses gene name normalization to achieve 98.6% entity matching between sources
- **Relation Standardization**: Maps 30+ heterogeneous relation types to a unified schema
- **Interactive Visualization**: Streamlit-based web app for exploring the knowledge graph
- **GNN-Ready Output**: Exports in formats compatible with PyTorch Geometric and NetworkX

## Quick Start

### Prerequisites

- Python 3.9+
- ~2GB disk space for data files

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gnn-ddi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For visualization only
pip install -r requirements-viz.txt
```

### Data Preparation

1. **Download PrimeKG** from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)
   - Place `kg.csv` in `data/`

2. **Download PharmGKB** from [PharmGKB Downloads](https://www.pharmgkb.org/downloads)
   - Extract to `data/pharmgkb/`

3. **Run exploration notebooks** to preprocess data:
   ```bash
   jupyter notebook explore_primekg.ipynb
   jupyter notebook explore_pharmgkb.ipynb
   ```

### Combine Knowledge Graphs

```bash
python combine_knowledge_graphs.py
```

This generates:
- `data/combined/combined_kg.csv` - Unified edge list (8.2M edges)
- `data/combined/combined_nodes.csv` - Node registry (139K nodes)
- `data/combined/combined_graph.pkl` - NetworkX graph object
- `data/combined/id_mappings.json` - ID translation tables
- `data/combined/merge_report.txt` - Statistics and quality metrics

### Explore the Graph

```bash
streamlit run visualize_kg.py
```

Opens an interactive web interface where you can:
- Search nodes by name (e.g., BRCA1, Aspirin, Diabetes)
- Filter by node types and relation types
- Explore N-hop neighborhoods
- Export subgraphs as CSV

## Project Structure

```
gnn-ddi/
├── combine_knowledge_graphs.py  # Main KG merging script
├── visualize_kg.py              # Interactive Streamlit visualization
├── requirements.txt             # Core dependencies
├── requirements-viz.txt         # Visualization dependencies
├── explore_primekg.ipynb        # PrimeKG data exploration
├── explore_pharmgkb.ipynb       # PharmGKB data exploration
├── combine_graphs.ipynb         # Interactive merging notebook
├── docs/
│   └── knowledge-graph-combining.md  # Technical documentation
├── data/
│   ├── kg.csv                   # PrimeKG edge list
│   ├── pharmgkb/                # PharmGKB raw data
│   │   ├── drugs.tsv
│   │   ├── genes.tsv
│   │   ├── relationships.tsv
│   │   ├── clinical_ann_*.tsv
│   │   └── pharmgkb_kg.csv      # Processed PharmGKB edges
│   └── combined/                # Merged outputs
│       ├── combined_kg.csv
│       ├── combined_nodes.csv
│       ├── combined_graph.pkl
│       ├── id_mappings.json
│       └── merge_report.txt
└── lib/                         # Frontend visualization libraries
```

## Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│    PrimeKG      │     │    PharmGKB     │
│  (8.1M edges)   │     │  (127K edges)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Gene Name Matching   │
         │  (2,465 genes matched)│
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Unified Node Registry│
         │  (139,375 nodes)      │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Relation Standardization
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │     Edge Merging      │
         │  (8,228,098 edges)    │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   NetworkX Graph &    │
         │   Export Outputs      │
         └───────────────────────┘
```

## Combined Graph Statistics

| Metric | Value |
|--------|-------|
| Total Nodes | 139,375 |
| Total Edges | 8,228,098 |
| Graph Density | 0.000423 |
| Connected Components | 4 |
| Largest Component | 139,367 (99.99%) |
| Average Degree | 59.02 |
| Max Degree | 17,369 |

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
| variant | 7,159 | PharmGKB |
| cellular_component | 4,176 | PrimeKG |
| pathway | 2,516 | PrimeKG |
| exposure | 818 | PrimeKG |
| haplotype | 639 | PharmGKB |

### Top Relation Types

| Relation | Count |
|----------|-------|
| anatomy_protein_present | 3,036,406 |
| drug_drug | 2,672,628 |
| gene_gene_interaction | 642,150 |
| disease_phenotype_positive | 300,634 |
| gene_drug_associated | 51,910 |

## Usage Examples

### Load the Combined Graph

```python
import pickle
import pandas as pd

# Option 1: NetworkX graph (for graph algorithms)
with open('data/combined/combined_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Option 2: DataFrames (for PyTorch Geometric)
edges = pd.read_csv('data/combined/combined_kg.csv')
nodes = pd.read_csv('data/combined/combined_nodes.csv')
```

### Query the Graph

```python
# Find all drugs that target BRCA1
brca1_neighbors = list(G.neighbors(brca1_node_id))
drugs = [n for n in brca1_neighbors
         if G.nodes[n].get('node_type') == 'drug']

# Get all gene-drug relationships
gene_drug_edges = edges[edges['relation'].str.contains('gene_drug')]
```

### Convert to PyTorch Geometric

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Add node features for each type
for node_type in nodes['node_type'].unique():
    mask = nodes['node_type'] == node_type
    data[node_type].num_nodes = mask.sum()

# Add edge indices for each relation type
for relation in edges['relation'].unique():
    mask = edges['relation'] == relation
    edge_df = edges[mask]
    # ... convert to edge_index tensors
```

## Key Design Decisions

### Why Gene-Based Entity Alignment?

Gene symbols (BRCA1, TP53, CYP2D6) are standardized by HGNC, making string matching reliable:
- Normalized matching achieves **98.6% alignment** of PharmGKB genes
- Drug names are inconsistent (brand vs generic vs chemical names)
- Gene matching provides the bridge for pharmacogenomic relationships

### Why Keep Duplicate Edges?

When the same relationship exists in both sources, both edges are preserved:
- Independent evidence strengthens confidence
- Source attribution enables filtering
- GNNs can weight by evidence count

### Why MultiDiGraph?

The same gene-drug pair can have multiple relationship types:
```
CYP2D6 --[pharmacokinetic]--> Codeine
CYP2D6 --[pharmacodynamic]--> Codeine
```
A simple graph would collapse these, losing information.

## Documentation

- [Knowledge Graph Combining Guide](docs/knowledge-graph-combining.md) - Detailed technical documentation

## Data Sources

- **PrimeKG**: Chandak, P., Huang, K., & Zitnik, M. (2022). Building a knowledge graph to enable precision medicine. *bioRxiv*.
- **PharmGKB**: Whirl-Carrillo, M., et al. (2021). PharmGKB: A worldwide resource for pharmacogenomic information. *Wiley Interdisciplinary Reviews: Systems Biology and Medicine*.

## License

This project is for research purposes. Please cite the original data sources (PrimeKG, PharmGKB) when using this work.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

This research combines two excellent biomedical knowledge graphs:
- [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/) from the Zitnik Lab at Harvard
- [PharmGKB](https://www.pharmgkb.org/) from Stanford University
