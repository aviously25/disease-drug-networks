#!/usr/bin/env python3
"""
Combine PrimeKG and PharmGKB Knowledge Graphs

This script merges two biomedical knowledge graphs into a unified graph
for GNN-based drug-disease interaction prediction.

Usage:
    python combine_knowledge_graphs.py

Output:
    data/combined/combined_kg.csv       - Unified edge list
    data/combined/combined_nodes.csv    - Node registry with dual IDs
    data/combined/combined_graph.pkl    - NetworkX graph object
    data/combined/id_mappings.json      - ID translation tables
    data/combined/merge_report.txt      - Statistics and quality metrics

Author: Research Project
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import pickle
import os
from collections import defaultdict
from typing import Dict, Tuple, Set
from tqdm import tqdm


# Configuration
PRIMEKG_PATH = 'data/kg.csv'
PHARMGKB_KG_PATH = 'data/pharmgkb/pharmgkb_kg.csv'
OUTPUT_DIR = 'data/combined'

# Relation type standardization mapping
RELATION_MAPPING = {
    # PrimeKG relations
    'protein_protein': 'gene_gene_interaction',
    'drug_protein': 'drug_gene_target',
    'indication': 'drug_disease_indication',
    'contraindication': 'drug_disease_contraindication',
    'off-label use': 'drug_disease_offlabel',
    'disease_protein': 'disease_gene_association',
    'drug_effect': 'drug_phenotype_effect',
    'disease_phenotype_positive': 'disease_phenotype_positive',
    'disease_phenotype_negative': 'disease_phenotype_negative',
    # PharmGKB relations
    'associated': 'gene_drug_associated',
    'pharmacodynamic_association': 'gene_drug_pharmacodynamic',
    'pharmacokinetic_association': 'gene_drug_pharmacokinetic',
    'pk_pd_association': 'gene_drug_pk_pd',
    'no_association': 'gene_drug_no_association',
    'ambiguous_association': 'gene_drug_ambiguous',
    'unknown_association': 'gene_drug_unknown',
}


def load_knowledge_graphs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both knowledge graphs from CSV files."""
    print("Loading knowledge graphs...")
    print("="*60)

    # Load PrimeKG
    if os.path.exists(PRIMEKG_PATH):
        primekg = pd.read_csv(PRIMEKG_PATH, low_memory=False)
        print(f"  PrimeKG loaded: {len(primekg):,} edges")
    else:
        raise FileNotFoundError(f"PrimeKG not found at {PRIMEKG_PATH}. Run explore_primekg.ipynb first.")

    # Load PharmGKB KG
    if os.path.exists(PHARMGKB_KG_PATH):
        pharmgkb = pd.read_csv(PHARMGKB_KG_PATH, low_memory=False)
        print(f"  PharmGKB loaded: {len(pharmgkb):,} edges")
    else:
        raise FileNotFoundError(f"PharmGKB KG not found at {PHARMGKB_KG_PATH}. Run explore_pharmgkb.ipynb first.")

    return primekg, pharmgkb


def build_gene_mapping(primekg: pd.DataFrame, pharmgkb: pd.DataFrame) -> Dict[str, Dict]:
    """Build gene name to ID mappings for both KGs."""
    print("\nBuilding gene name mappings...")
    print("="*60)

    # Extract genes from PrimeKG
    primekg_genes_x = primekg[primekg['x_type'] == 'gene/protein'][['x_id', 'x_name']].drop_duplicates()
    primekg_genes_y = primekg[primekg['y_type'] == 'gene/protein'][['y_id', 'y_name']].drop_duplicates()
    primekg_genes_x.columns = ['id', 'name']
    primekg_genes_y.columns = ['id', 'name']
    primekg_genes = pd.concat([primekg_genes_x, primekg_genes_y]).drop_duplicates()

    # Extract genes from PharmGKB
    pharmgkb_genes_x = pharmgkb[pharmgkb['x_type'] == 'gene/protein'][['x_id', 'x_name']].drop_duplicates()
    pharmgkb_genes_y = pharmgkb[pharmgkb['y_type'] == 'gene/protein'][['y_id', 'y_name']].drop_duplicates()
    pharmgkb_genes_x.columns = ['id', 'name']
    pharmgkb_genes_y.columns = ['id', 'name']
    pharmgkb_genes = pd.concat([pharmgkb_genes_x, pharmgkb_genes_y]).drop_duplicates()

    print(f"  PrimeKG genes: {len(primekg_genes):,}")
    print(f"  PharmGKB genes: {len(pharmgkb_genes):,}")

    # Normalize gene names for matching
    primekg_name_to_id = {}
    for _, row in primekg_genes.iterrows():
        name_norm = str(row['name']).upper().strip()
        if name_norm not in primekg_name_to_id:
            primekg_name_to_id[name_norm] = row['id']

    pharmgkb_name_to_id = {}
    for _, row in pharmgkb_genes.iterrows():
        name_norm = str(row['name']).upper().strip()
        if name_norm not in pharmgkb_name_to_id:
            pharmgkb_name_to_id[name_norm] = row['id']

    # Find matches
    matched_genes = set(primekg_name_to_id.keys()) & set(pharmgkb_name_to_id.keys())

    print(f"  Matched genes (by name): {len(matched_genes):,}")
    print(f"  Match rate: {100*len(matched_genes)/len(pharmgkb_name_to_id):.1f}%")

    # Create mapping
    gene_mapping = {
        'primekg_name_to_id': primekg_name_to_id,
        'pharmgkb_name_to_id': pharmgkb_name_to_id,
        'matched_genes': matched_genes,
        'primekg_to_pharmgkb': {
            primekg_name_to_id[name]: pharmgkb_name_to_id[name]
            for name in matched_genes
        },
        'pharmgkb_to_primekg': {
            pharmgkb_name_to_id[name]: primekg_name_to_id[name]
            for name in matched_genes
        }
    }

    return gene_mapping


def create_node_registry(primekg: pd.DataFrame, pharmgkb: pd.DataFrame,
                         gene_mapping: Dict) -> pd.DataFrame:
    """Create unified node registry with all nodes from both KGs."""
    print("\nCreating unified node registry...")
    print("="*60)

    nodes = []
    node_id_counter = 0
    seen_nodes = set()

    # Process PrimeKG nodes
    for side in ['x', 'y']:
        id_col, name_col, type_col = f'{side}_id', f'{side}_name', f'{side}_type'
        unique_nodes = primekg[[id_col, name_col, type_col]].drop_duplicates()

        for _, row in tqdm(unique_nodes.iterrows(), total=len(unique_nodes),
                          desc=f"PrimeKG {side} nodes"):
            node_key = (row[type_col], str(row[id_col]))
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)

                # Check if this gene has a PharmGKB mapping
                pharmgkb_id = None
                if row[type_col] == 'gene/protein':
                    name_norm = str(row[name_col]).upper().strip()
                    if name_norm in gene_mapping['matched_genes']:
                        pharmgkb_id = gene_mapping['pharmgkb_name_to_id'].get(name_norm)

                nodes.append({
                    'unified_id': node_id_counter,
                    'node_type': row[type_col],
                    'name': row[name_col],
                    'primekg_id': row[id_col],
                    'pharmgkb_id': pharmgkb_id,
                    'source': 'PrimeKG' if pharmgkb_id is None else 'Both'
                })
                node_id_counter += 1

    # Process PharmGKB-only nodes (variants, haplotypes, and unmatched)
    for side in ['x', 'y']:
        id_col, name_col, type_col = f'{side}_id', f'{side}_name', f'{side}_type'
        unique_nodes = pharmgkb[[id_col, name_col, type_col]].drop_duplicates()

        for _, row in tqdm(unique_nodes.iterrows(), total=len(unique_nodes),
                          desc=f"PharmGKB {side} nodes"):
            node_type = row[type_col]
            pharmgkb_id = row[id_col]

            # Skip if already added via PrimeKG (matched genes)
            if node_type == 'gene/protein':
                if pharmgkb_id in gene_mapping['pharmgkb_to_primekg']:
                    continue  # Already added

            node_key = (node_type, f"pharmgkb_{pharmgkb_id}")
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                nodes.append({
                    'unified_id': node_id_counter,
                    'node_type': node_type,
                    'name': row[name_col],
                    'primekg_id': None,
                    'pharmgkb_id': pharmgkb_id,
                    'source': 'PharmGKB'
                })
                node_id_counter += 1

    nodes_df = pd.DataFrame(nodes)

    # Statistics
    print(f"\n  Total unified nodes: {len(nodes_df):,}")
    print(f"\n  Node type distribution:")
    for ntype, count in nodes_df['node_type'].value_counts().items():
        print(f"    {ntype}: {count:,}")

    print(f"\n  Source distribution:")
    for source, count in nodes_df['source'].value_counts().items():
        print(f"    {source}: {count:,}")

    return nodes_df


def create_id_lookup(nodes_df: pd.DataFrame) -> Dict:
    """Create lookup tables for ID translation."""
    # PrimeKG ID to unified ID
    primekg_to_unified = {}
    for _, row in nodes_df[nodes_df['primekg_id'].notna()].iterrows():
        key = (row['node_type'], str(row['primekg_id']))
        primekg_to_unified[key] = row['unified_id']

    # PharmGKB ID to unified ID
    pharmgkb_to_unified = {}
    for _, row in nodes_df[nodes_df['pharmgkb_id'].notna()].iterrows():
        key = (row['node_type'], str(row['pharmgkb_id']))
        pharmgkb_to_unified[key] = row['unified_id']

    return {
        'primekg_to_unified': primekg_to_unified,
        'pharmgkb_to_unified': pharmgkb_to_unified
    }


def standardize_relation(relation: str) -> str:
    """Standardize relation type using mapping."""
    return RELATION_MAPPING.get(relation, relation)


def merge_edges(primekg: pd.DataFrame, pharmgkb: pd.DataFrame,
                id_lookup: Dict, gene_mapping: Dict) -> pd.DataFrame:
    """Merge edges from both KGs with unified IDs."""
    print("\nMerging edges...")
    print("="*60)

    edges = []

    # Process PrimeKG edges
    print(f"  Processing PrimeKG edges ({len(primekg):,})...")
    for _, row in tqdm(primekg.iterrows(), total=len(primekg), desc="PrimeKG"):
        x_key = (row['x_type'], str(row['x_id']))
        y_key = (row['y_type'], str(row['y_id']))

        x_unified = id_lookup['primekg_to_unified'].get(x_key)
        y_unified = id_lookup['primekg_to_unified'].get(y_key)

        if x_unified is not None and y_unified is not None:
            edges.append({
                'source_id': x_unified,
                'target_id': y_unified,
                'source_type': row['x_type'],
                'target_type': row['y_type'],
                'relation': standardize_relation(row['relation']),
                'original_relation': row['relation'],
                'source_kg': 'PrimeKG',
                'evidence': None
            })

    # Process PharmGKB edges
    print(f"  Processing PharmGKB edges ({len(pharmgkb):,})...")
    for _, row in tqdm(pharmgkb.iterrows(), total=len(pharmgkb), desc="PharmGKB"):
        x_type = row['x_type']
        y_type = row['y_type']
        x_id = str(row['x_id'])
        y_id = str(row['y_id'])

        # Try to find unified ID
        x_unified = None
        y_unified = None

        # For genes, check if matched to PrimeKG
        if x_type == 'gene/protein' and x_id in gene_mapping['pharmgkb_to_primekg']:
            primekg_id = gene_mapping['pharmgkb_to_primekg'][x_id]
            x_unified = id_lookup['primekg_to_unified'].get((x_type, str(primekg_id)))
        if x_unified is None:
            x_unified = id_lookup['pharmgkb_to_unified'].get((x_type, x_id))

        if y_type == 'gene/protein' and y_id in gene_mapping['pharmgkb_to_primekg']:
            primekg_id = gene_mapping['pharmgkb_to_primekg'][y_id]
            y_unified = id_lookup['primekg_to_unified'].get((y_type, str(primekg_id)))
        if y_unified is None:
            y_unified = id_lookup['pharmgkb_to_unified'].get((y_type, y_id))

        if x_unified is not None and y_unified is not None:
            edges.append({
                'source_id': x_unified,
                'target_id': y_unified,
                'source_type': x_type,
                'target_type': y_type,
                'relation': standardize_relation(row['relation']),
                'original_relation': row['relation'],
                'source_kg': 'PharmGKB',
                'evidence': row.get('evidence', None)
            })

    edges_df = pd.DataFrame(edges)

    print(f"\n  Total merged edges: {len(edges_df):,}")
    print(f"\n  Edges by source:")
    for source, count in edges_df['source_kg'].value_counts().items():
        print(f"    {source}: {count:,}")

    return edges_df


def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from unified nodes and edges."""
    print("\nBuilding NetworkX graph...")
    print("="*60)

    G = nx.Graph()

    # Add nodes
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Adding nodes"):
        G.add_node(
            row['unified_id'],
            node_type=row['node_type'],
            name=row['name'],
            primekg_id=row['primekg_id'],
            pharmgkb_id=row['pharmgkb_id'],
            source=row['source']
        )

    # Add edges
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Adding edges"):
        G.add_edge(
            row['source_id'],
            row['target_id'],
            relation=row['relation'],
            original_relation=row['original_relation'],
            source_kg=row['source_kg'],
            evidence=row['evidence']
        )

    print(f"\n  Graph built:")
    print(f"    Nodes: {G.number_of_nodes():,}")
    print(f"    Edges: {G.number_of_edges():,}")

    return G


def compute_statistics(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                       G: nx.Graph) -> Dict:
    """Compute statistics for the combined graph."""
    print("\nComputing statistics...")
    print("="*60)

    stats = {
        'total_nodes': len(nodes_df),
        'total_edges': len(edges_df),
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'node_types': nodes_df['node_type'].value_counts().to_dict(),
        'edge_sources': edges_df['source_kg'].value_counts().to_dict(),
        'relation_types': edges_df['relation'].value_counts().to_dict(),
        'density': nx.density(G),
        'connected_components': nx.number_connected_components(G),
    }

    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    stats['largest_cc_size'] = len(largest_cc)
    stats['largest_cc_percent'] = 100 * len(largest_cc) / G.number_of_nodes()

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    stats['avg_degree'] = np.mean(degrees)
    stats['max_degree'] = max(degrees)
    stats['median_degree'] = np.median(degrees)

    return stats


def export_combined_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                          G: nx.Graph, gene_mapping: Dict, stats: Dict):
    """Export all outputs to files."""
    print("\nExporting combined graph...")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Export edge list (KG format)
    kg_path = f"{OUTPUT_DIR}/combined_kg.csv"
    edges_df.to_csv(kg_path, index=False)
    print(f"  Saved: {kg_path} ({len(edges_df):,} edges)")

    # 2. Export node registry
    nodes_path = f"{OUTPUT_DIR}/combined_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    print(f"  Saved: {nodes_path} ({len(nodes_df):,} nodes)")

    # 3. Export NetworkX graph
    graph_path = f"{OUTPUT_DIR}/combined_graph.pkl"
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"  Saved: {graph_path}")

    # 4. Export ID mappings
    mappings_path = f"{OUTPUT_DIR}/id_mappings.json"
    # Convert sets to lists for JSON serialization
    mappings = {
        'matched_genes': list(gene_mapping['matched_genes']),
        'primekg_to_pharmgkb': {str(k): str(v) for k, v in gene_mapping['primekg_to_pharmgkb'].items()},
        'pharmgkb_to_primekg': {str(k): str(v) for k, v in gene_mapping['pharmgkb_to_primekg'].items()},
    }
    with open(mappings_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"  Saved: {mappings_path}")

    # 5. Export merge report
    report_path = f"{OUTPUT_DIR}/merge_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMBINED KNOWLEDGE GRAPH - MERGE REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total nodes: {stats['total_nodes']:,}\n")
        f.write(f"Total edges: {stats['total_edges']:,}\n")
        f.write(f"Graph density: {stats['density']:.6f}\n")
        f.write(f"Connected components: {stats['connected_components']:,}\n")
        f.write(f"Largest component: {stats['largest_cc_size']:,} ({stats['largest_cc_percent']:.1f}%)\n")
        f.write(f"Average degree: {stats['avg_degree']:.2f}\n")
        f.write(f"Max degree: {stats['max_degree']:,}\n\n")

        f.write("NODE TYPES\n")
        f.write("-"*40 + "\n")
        for ntype, count in sorted(stats['node_types'].items(), key=lambda x: -x[1]):
            f.write(f"  {ntype}: {count:,}\n")
        f.write("\n")

        f.write("EDGE SOURCES\n")
        f.write("-"*40 + "\n")
        for source, count in stats['edge_sources'].items():
            f.write(f"  {source}: {count:,}\n")
        f.write("\n")

        f.write("RELATION TYPES\n")
        f.write("-"*40 + "\n")
        for rel, count in sorted(stats['relation_types'].items(), key=lambda x: -x[1]):
            f.write(f"  {rel}: {count:,}\n")

    print(f"  Saved: {report_path}")


def main():
    """Main function to combine knowledge graphs."""
    print("\n" + "="*60)
    print("COMBINING PRIMEKG AND PHARMGKB KNOWLEDGE GRAPHS")
    print("="*60 + "\n")

    # Load KGs
    primekg, pharmgkb = load_knowledge_graphs()

    # Build gene mapping
    gene_mapping = build_gene_mapping(primekg, pharmgkb)

    # Create unified node registry
    nodes_df = create_node_registry(primekg, pharmgkb, gene_mapping)

    # Create ID lookup tables
    id_lookup = create_id_lookup(nodes_df)

    # Merge edges
    edges_df = merge_edges(primekg, pharmgkb, id_lookup, gene_mapping)

    # Build NetworkX graph
    G = build_networkx_graph(nodes_df, edges_df)

    # Compute statistics
    stats = compute_statistics(nodes_df, edges_df, G)

    # Export everything
    export_combined_graph(nodes_df, edges_df, G, gene_mapping, stats)

    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"\nCombined graph saved to: {OUTPUT_DIR}/")
    print(f"  - combined_kg.csv ({stats['total_edges']:,} edges)")
    print(f"  - combined_nodes.csv ({stats['total_nodes']:,} nodes)")
    print(f"  - combined_graph.pkl (NetworkX)")
    print(f"  - id_mappings.json")
    print(f"  - merge_report.txt")

    return nodes_df, edges_df, G, stats


if __name__ == '__main__':
    nodes_df, edges_df, G, stats = main()
