"""Pipeline orchestration: load data → build graph → evaluate models → save results."""
import os
import pickle
import time

import numpy as np
import pandas as pd

from pipeline.data.loader import DataLoader
from pipeline.data.graph_builder import GraphBuilder
from pipeline.evaluation.evaluator import Evaluator
from pipeline.models.hetero_gnn import HeteroGNN
from pipeline.models.dual_encoder import DualEncoder
from pipeline.models.metapath import MetapathModel


MODEL_REGISTRY = {
    'hetero_gnn': HeteroGNN,
    'dual_encoder': DualEncoder,
    'metapath': MetapathModel,
}


def run(cfg: dict, model_filter: list[str] | None = None, eval_override: str | None = None):
    """Run the full pipeline.

    Parameters
    ----------
    cfg           : loaded YAML config dict
    model_filter  : if given, only run models in this list
    eval_override : if given, override evaluation.strategy
    """
    if eval_override:
        cfg['evaluation']['strategy'] = eval_override

    results_dir = cfg['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()

    # ------------------------------------------------------------------
    # [1/4] Load data
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Pipeline: Three-Architecture Drug-Disease Prediction")
    print("=" * 65)
    print("\n[1/4] Loading data...")
    loader = DataLoader(cfg)
    samples_df, graph_df = loader.load()
    n_dis = samples_df['disease_id'].nunique()
    n_drug = samples_df['drug_id'].nunique()
    n_pos = (samples_df['label'] == 1).sum()
    n_neg = (samples_df['label'] == 0).sum()
    print(f"  {n_dis} diseases, {n_drug} drugs")
    print(f"  Dataset: {len(samples_df)} samples ({n_pos} contra, {n_neg} indica)")
    print(f"  Graph: {len(graph_df):,} edges (target relations excluded)")

    # ------------------------------------------------------------------
    # [2/4] Build graph
    # ------------------------------------------------------------------
    print("\n[2/4] Building PyG heterogeneous graph...")
    builder = GraphBuilder(cfg)
    data, drug_map, disease_map = builder.build(graph_df, samples_df)

    n_node_types = len(data.node_types)
    n_edge_types = len(data.edge_index_dict)
    total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    total_edges = sum(et.edge_index.shape[1] for et in data.edge_stores)
    print(f"  {n_node_types} node types, {n_edge_types} edge types (incl. reverse)")
    print(f"  Total nodes: {total_nodes:,}  |  Total edges: {total_edges:,}")

    # Map dataset drug/disease pairs → local PyG indices
    drug_local = (
        samples_df['drug_id'].astype(str).map(drug_map).fillna(-1).astype(int).values
    )
    dis_local = (
        samples_df['disease_id'].astype(str).map(disease_map).fillna(-1).astype(int).values
    )
    n_missing = (drug_local == -1).sum() + (dis_local == -1).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} pairs missing node mapping — dropping.")
        keep = (drug_local != -1) & (dis_local != -1)
        samples_df = samples_df[keep].reset_index(drop=True)
        drug_local = drug_local[keep]
        dis_local = dis_local[keep]
    else:
        print("  All pairs mapped successfully.")
    y = samples_df['label'].values

    # ------------------------------------------------------------------
    # [3/4] Build metapath graphs (if metapath model is enabled)
    # ------------------------------------------------------------------
    metapath_graphs = {}
    model_cfgs = cfg.get('models', {})
    mp_cfg = model_cfgs.get('metapath', {})
    if mp_cfg.get('enabled', False) and (
        model_filter is None or 'metapath' in model_filter
    ):
        print("\n[3/4] Building metapath adjacency matrices...")
        metapaths = mp_cfg.get('metapaths', [])
        metapath_graphs = builder.build_metapath_graphs(
            graph_df, metapaths, drug_map, disease_map
        )
    else:
        print("\n[3/4] Metapath model disabled — skipping metapath construction.")

    # ------------------------------------------------------------------
    # [4/4] Train and evaluate each enabled model
    # ------------------------------------------------------------------
    print("\n[4/4] Training and evaluating models...")
    evaluator = Evaluator(cfg)

    all_results = {}
    summary_rows = []

    for model_name, model_cls in MODEL_REGISTRY.items():
        if model_filter is not None and model_name not in model_filter:
            continue
        m_cfg = model_cfgs.get(model_name, {})
        if not m_cfg.get('enabled', False):
            print(f"  Skipping {model_name} (disabled in config)")
            continue

        print(f"\n{'─'*65}")
        print(f"  Model: {model_name}")
        print(f"{'─'*65}")

        results = evaluator.run(
            model_cls=model_cls,
            model_cfg=m_cfg,
            data=data,
            metapath_graphs=metapath_graphs,
            samples_df=samples_df,
            drug_local=drug_local,
            dis_local=dis_local,
            y=y,
            model_name=model_name,
            drug_map=drug_map,
            disease_map=disease_map,
        )
        all_results[model_name] = results

        # Save per-model results
        out_path = os.path.join(results_dir, f'{model_name}_results.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump({'model': model_name, 'config': m_cfg, **results}, f)
        print(f"  Saved: {out_path}")

        summary_rows.append({
            'model': model_name,
            'strategy': results['strategy'],
            'mean_auc': results['mean_auc'],
            'std_auc': results['std_auc'],
            'mean_acc': results['mean_acc'],
            'mean_f1': results['mean_f1'],
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False, float_format='{:.3f}'.format))

        csv_path = os.path.join(results_dir, 'summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    total_time = (time.time() - t0) / 60
    print(f"\nTotal runtime: {total_time:.1f} min")
    print("=" * 65)
    return all_results
