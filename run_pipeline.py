"""Entry point for the three-architecture drug-disease prediction pipeline.

Usage
-----
# Default: 5-fold GroupKFold, all enabled models
python run_pipeline.py

# Custom config
python run_pipeline.py --config config/default.yaml

# Override evaluation strategy
python run_pipeline.py --eval lodo
python run_pipeline.py --eval group_kfold

# Run a subset of models
python run_pipeline.py --models hetero_gnn
python run_pipeline.py --models hetero_gnn,dual_encoder

# Combine flags
python run_pipeline.py --config config/default.yaml --eval lodo --models hetero_gnn
"""
import argparse
import sys
import warnings

import yaml

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Drug-disease prediction pipeline (HeteroGNN / DualEncoder / Metapath)'
    )
    parser.add_argument(
        '--config', default='config/default.yaml',
        help='Path to YAML config file (default: config/default.yaml)'
    )
    parser.add_argument(
        '--models', default=None,
        help='Comma-separated list of models to run, e.g. hetero_gnn,dual_encoder'
    )
    parser.add_argument(
        '--eval', default=None, choices=['group_kfold', 'lodo'],
        help='Override evaluation strategy from config'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(',')]

    from pipeline.runner import run
    run(cfg, model_filter=model_filter, eval_override=args.eval)


if __name__ == '__main__':
    main()
