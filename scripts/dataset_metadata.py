"""One row per dataset: size, features, classes and imbalance, as the runs see them.

``n_rows`` is after the ``MAX_DATASET_ROWS`` subsample every run applies;
``n_rows_original`` is before it. ``minority_share`` is the smallest class's
share of the rows.

Usage: uv run python -m scripts.dataset_metadata
"""
import numpy as np
import pandas as pd

from benchmark.data import evaluated_datasets, load_data_df
from config import DUPLICATE_DATASETS, MAX_DATASET_ROWS, RANDOM_STATE

OUTPUT = "results/datasets.csv"

rows = []
for name in evaluated_datasets():
    X, y, cat_cols = load_data_df(name)
    n_original = len(y)
    if n_original > MAX_DATASET_ROWS:
        y = y[np.random.default_rng(RANDOM_STATE).choice(n_original, MAX_DATASET_ROWS, replace=False)]
    counts = np.bincount(y)
    rows.append(dict(dataset=name, n_rows=len(y), n_rows_original=n_original,
                     n_features=X.shape[1], n_categorical=len(cat_cols),
                     n_classes=len(counts), minority_share=round(counts.min() / len(y), 4),
                     uci_duplicate=name in DUPLICATE_DATASETS))
pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}: {len(rows)} datasets")
