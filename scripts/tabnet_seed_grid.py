"""TabNet's run-to-run noise next to its fold-to-fold spread, on the same pipeline.

Each outer fold's final model is refit from its stored best parameters under
several seeds. The seed sets both sources of training randomness in the wrapper:
the early-stopping validation split and TabNet's own initialisation. Seed 0 is the
benchmark's run, so its scores are checked against the stored ones. One grid gives
both spreads: across seeds within a fold, and across folds at one seed.

Usage: uv run python -m scripts.tabnet_seed_grid [n_seeds] [dataset ...]
"""
import sys
import torch  # must precede the other torch users to win the OpenMP init race

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models import torch_wrappers
from benchmark.models.build import build_final_model
from config import MAX_DATASET_ROWS, N_OUTER_FOLDS, RANDOM_STATE

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DATASETS = sys.argv[2:] or ["abalone-3class", "volcanoes-a3"]
OUTPUT = "results/tabnet_seed_grid.csv"

stored = joblib.load(ckpt_path("tabnet"))
rows = []
for dataset in DATASETS:
    X, y, cat_cols = load_data_df(dataset)
    if len(X) > MAX_DATASET_ROWS:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), MAX_DATASET_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    n_classes = int(np.unique(y).size)
    folds = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    for fold, (train, test) in enumerate(folds):
        X_train, X_test = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
        for seed in range(N_SEEDS):
            torch_wrappers.RANDOM_STATE = seed
            model = build_final_model("tabnet", stored[dataset]["best_params"][fold], n_classes, cat_cols)
            score = pr_auc_score(y[test], model.fit(X_train, y[train]).predict_proba(X_test))
            rows.append(dict(dataset=dataset, fold=fold, seed=seed, pr_auc=score,
                             stored=stored[dataset]["scores"][fold]))
            print(f"{dataset:20s} fold {fold} seed {seed}  PR AUC {score:.4f}", flush=True)

df = pd.DataFrame(rows)
df.round(5).to_csv(OUTPUT, index=False)
for dataset, g in df.groupby("dataset"):
    grid = g.pivot(index="seed", columns="fold", values="pr_auc")
    seed0 = g[g.seed == 0]
    print(f"\n{dataset}: PR AUC, seeds x outer folds")
    print(grid.round(4).to_string())
    print(f"seed 0 vs the stored run, max |diff|: {np.abs(seed0.pr_auc - seed0.stored).max():.4f}")
    print(f"sd across seeds, within a fold (mean over folds): {grid.std(ddof=1).mean():.4f}")
    print(f"sd across folds, at one seed (mean over seeds):   {grid.std(axis=1, ddof=1).mean():.4f}")
    print(f"sd across seeds of the 4-fold mean (the benchmark's number): {grid.mean(axis=1).std(ddof=1):.4f}")
print(f"Saved {OUTPUT}")
