"""Seed the LODO checkpoint with results from the cancelled job.
Run from project root: python scripts/seed_lodo_checkpoint.py
"""
import os, pickle

os.makedirs('results', exist_ok=True)

completed = [
    {'disease_id': '10198',                       'auc': 0.500, 'acc': 0.0, 'f1': 0.0, 'n_test': 18},
    {'disease_id': '10481',                       'auc': 0.488, 'acc': 0.0, 'f1': 0.0, 'n_test': 108},
    {'disease_id': '11122',                       'auc': 0.653, 'acc': 0.0, 'f1': 0.0, 'n_test': 204},
    {'disease_id': '1187',                        'auc': 0.667, 'acc': 0.0, 'f1': 0.0, 'n_test': 10},
    {'disease_id': '1315',                        'auc': 0.556, 'acc': 0.0, 'f1': 0.0, 'n_test': 22},
    {'disease_id': '14412_7761_7762_18473_37748', 'auc': 0.928, 'acc': 0.0, 'f1': 0.0, 'n_test': 60},
    {'disease_id': '15277',                       'auc': 0.667, 'acc': 0.0, 'f1': 0.0, 'n_test': 10},
    {'disease_id': '16158',                       'auc': 0.522, 'acc': 0.0, 'f1': 0.0, 'n_test': 38},
    {'disease_id': '16248',                       'auc': 0.997, 'acc': 0.0, 'f1': 0.0, 'n_test': 82},
    {'disease_id': '16684',                       'auc': 1.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 8},
    {'disease_id': '1725',                        'auc': 0.583, 'acc': 0.0, 'f1': 0.0, 'n_test': 14},
    {'disease_id': '17327',                       'auc': 1.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 82},
    {'disease_id': '18271',                       'auc': 1.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 6},
    {'disease_id': '18312',                       'auc': 1.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 12},
    {'disease_id': '18673',                       'auc': 0.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 6},
    {'disease_id': '19182',                       'auc': 0.644, 'acc': 0.0, 'f1': 0.0, 'n_test': 204},
    {'disease_id': '19345',                       'auc': 0.657, 'acc': 0.0, 'f1': 0.0, 'n_test': 24},
    {'disease_id': '19624_15055_15056',           'auc': 0.814, 'acc': 0.0, 'f1': 0.0, 'n_test': 68},
    {'disease_id': '21187_11237_11470',           'auc': 0.759, 'acc': 0.0, 'f1': 0.0, 'n_test': 112},
    {'disease_id': '2206',                        'auc': 1.000, 'acc': 0.0, 'f1': 0.0, 'n_test': 6},
    {'disease_id': '22687_4742',                  'auc': 0.847, 'acc': 0.0, 'f1': 0.0, 'n_test': 68},
    {'disease_id': '2571',                        'auc': 0.816, 'acc': 0.0, 'f1': 0.0, 'n_test': 28},
    {'disease_id': '3792',                        'auc': 0.987, 'acc': 0.0, 'f1': 0.0, 'n_test': 82},
    {'disease_id': '4247',                        'auc': 0.618, 'acc': 0.0, 'f1': 0.0, 'n_test': 330},
    {'disease_id': '4577',                        'auc': 0.958, 'acc': 0.0, 'f1': 0.0, 'n_test': 20},
    {'disease_id': '4765',                        'auc': 0.791, 'acc': 0.0, 'f1': 0.0, 'n_test': 242},
    {'disease_id': '4784',                        'auc': 0.740, 'acc': 0.0, 'f1': 0.0, 'n_test': 242},
]

path = 'results/gnn_lodo_checkpoint.pkl'
with open(path, 'wb') as f:
    pickle.dump(completed, f)

print(f"Seeded checkpoint with {len(completed)} completed splits → {path}")
