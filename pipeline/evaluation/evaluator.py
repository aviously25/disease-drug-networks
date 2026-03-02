"""Evaluation strategies: GroupKFold and LODO with checkpointing."""
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipeline.models.base import BaseModel


class Evaluator:
    def __init__(self, cfg: dict):
        self.eval_cfg = cfg['evaluation']
        self.output_cfg = cfg['output']
        self.strategy = self.eval_cfg.get('strategy', 'group_kfold')
        self.n_splits = self.eval_cfg.get('n_splits', 5)
        self.lodo_n = self.eval_cfg.get('lodo_n_diseases', 50)
        self.seed = self.eval_cfg.get('seed', 42)
        self.checkpoint_dir = self.eval_cfg.get('checkpoint_dir', 'results/checkpoints/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run(
        self,
        model_cls,
        model_cfg: dict,
        data,
        metapath_graphs: dict,
        samples_df: pd.DataFrame,
        drug_local: np.ndarray,
        dis_local: np.ndarray,
        y: np.ndarray,
        model_name: str = '',
        drug_map: dict | None = None,
        disease_map: dict | None = None,
    ) -> dict:
        """Dispatch to the configured evaluation strategy.

        Parameters
        ----------
        model_cls      : BaseModel subclass (not instance)
        model_cfg      : model-specific config dict (emb_dim, lr, etc.)
        data           : PyG HeteroData
        metapath_graphs: precomputed metapath adjacency dict (for MetapathModel)
        samples_df     : full samples DataFrame
        drug_local     : int array mapping samples → PyG drug node index
        dis_local      : int array mapping samples → PyG disease node index
        y              : label array (0/1)
        model_name     : string tag for logging and checkpoints
        """
        drug_map = drug_map or {}
        disease_map = disease_map or {}
        if self.strategy == 'group_kfold':
            return self._group_kfold(
                model_cls, model_cfg, data, metapath_graphs,
                samples_df, drug_local, dis_local, y, model_name,
                drug_map, disease_map
            )
        elif self.strategy == 'lodo':
            return self._lodo(
                model_cls, model_cfg, data, metapath_graphs,
                samples_df, drug_local, dis_local, y, model_name,
                drug_map, disease_map
            )
        else:
            raise ValueError(f"Unknown evaluation strategy: {self.strategy!r}")

    # ------------------------------------------------------------------
    # GroupKFold
    # ------------------------------------------------------------------

    def _group_kfold(
        self, model_cls, model_cfg, data, metapath_graphs,
        samples_df, drug_local, dis_local, y, model_name,
        drug_map, disease_map
    ) -> dict:
        print(f"\n  [{model_name}] {self.n_splits}-fold GroupKFold (grouped by disease)...")
        groups = samples_df['disease_id'].values
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(drug_local, y, groups=groups)
        ):
            t_fold = time.time()
            n_held = len(set(groups[test_idx]))
            print(
                f"  Fold {fold+1}/{self.n_splits} | "
                f"train={len(train_idx):,}  test={len(test_idx):,} ({n_held} diseases)"
            )

            model = self._make_model(
                model_cls, model_cfg, data, metapath_graphs,
                drug_map, disease_map
            )
            metrics = model.train_and_eval(
                train_idx, test_idx, samples_df, drug_local, dis_local, y,
                label=f'{model_name} fold{fold+1}'
            )
            elapsed = time.time() - t_fold
            print(
                f"  → AUC={metrics['auc']:.3f}  "
                f"Acc={metrics['acc']:.3f}  F1={metrics['f1']:.3f}  ({elapsed:.0f}s)\n"
            )
            fold_results.append({
                'fold': fold + 1,
                'n_test': len(test_idx),
                'n_diseases': n_held,
                **metrics,
            })

        aucs = [r['auc'] for r in fold_results]
        print(
            f"  [{model_name}] GroupKFold: AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
        )
        return {
            'strategy': 'group_kfold',
            'fold_results': fold_results,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'mean_acc': float(np.mean([r['acc'] for r in fold_results])),
            'mean_f1': float(np.mean([r['f1'] for r in fold_results])),
        }

    # ------------------------------------------------------------------
    # LODO
    # ------------------------------------------------------------------

    def _lodo(
        self, model_cls, model_cfg, data, metapath_graphs,
        samples_df, drug_local, dis_local, y, model_name,
        drug_map, disease_map
    ) -> dict:
        groups = samples_df['disease_id'].values
        unique_diseases = sorted(set(groups))
        rng = np.random.RandomState(self.seed)
        lodo_diseases = sorted(
            rng.choice(unique_diseases, size=min(self.lodo_n, len(unique_diseases)), replace=False)
        )

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'{model_name}_lodo_checkpoint.pkl'
        )
        lodo_results = []
        completed_ids = set()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                lodo_results = pickle.load(f)
            completed_ids = {str(r['disease_id']) for r in lodo_results}
            print(
                f"  [{model_name}] LODO resuming from checkpoint: "
                f"{len(completed_ids)} splits already done."
            )

        total = len(lodo_diseases)
        print(f"\n  [{model_name}] LODO-{total} (GNN retrained each split)...")

        for i, hd in enumerate(lodo_diseases):
            if str(hd) in completed_ids:
                print(f"  [{i+1:2d}/{total}] disease={hd}  SKIPPED (checkpoint)", flush=True)
                continue

            tm = np.array([d == hd for d in groups])
            if tm.sum() < 5:
                continue
            yt = y[tm]
            if len(np.unique(yt)) < 2:
                continue

            train_idx = np.where(~tm)[0]
            test_idx = np.where(tm)[0]

            t_lodo = time.time()
            model = self._make_model(
                model_cls, model_cfg, data, metapath_graphs,
                drug_map, disease_map
            )
            metrics = model.train_and_eval(
                train_idx, test_idx, samples_df, drug_local, dis_local, y
            )
            elapsed = time.time() - t_lodo
            print(
                f"  [{i+1:2d}/{total}] disease={hd}  n_test={tm.sum()}  "
                f"AUC={metrics['auc']:.3f}  ({elapsed:.0f}s)",
                flush=True,
            )

            result = {'disease_id': str(hd), 'n_test': int(tm.sum()), **metrics}
            lodo_results.append(result)
            completed_ids.add(str(hd))

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(lodo_results, f)

        aucs = [r['auc'] for r in lodo_results]
        print(
            f"\n  [{model_name}] LODO-{total}: AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
        )
        return {
            'strategy': 'lodo',
            'disease_results': lodo_results,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'mean_acc': float(np.mean([r['acc'] for r in lodo_results])),
            'mean_f1': float(np.mean([r['f1'] for r in lodo_results])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_model(self, model_cls, model_cfg, data, metapath_graphs, drug_map, disease_map):
        """Instantiate a fresh model — handles both standard and MetapathModel."""
        from pipeline.models.metapath import MetapathModel
        if issubclass(model_cls, MetapathModel):
            return model_cls(
                model_cfg, data,
                drug_map=drug_map,
                disease_map=disease_map,
                metapath_graphs=metapath_graphs,
            )
        return model_cls(model_cfg, data, drug_map=drug_map, disease_map=disease_map)
