"""The effect of the prior start alone: same hyperparameters, two starts.

Every outer fold's final CatBoost model is refit from the parameters the
uniform-start run chose, once as that run fitted it and once from the class
prior. No tuning, so the difference is the start and nothing else; the full
re-run also re-tunes, and reports the start and the tuning change together.
The uniform arm is checked against the stored predictions.

Usage: uv run python -m scripts.catboost_start_effect [catboost checkpoint]
       (default: results/ckpt/catboost.joblib.bak-uniform-start, a local backup of
       the uniform-start checkpoint; checkpoints are not tracked)
"""
import sys
import torch  # must precede catboost to win the OpenMP init race

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.calibration import brier_score, confidence_gap, expected_calibration_error
from benchmark.data import load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models.build import build_final_model
from config import MAX_DATASET_ROWS, N_OUTER_FOLDS, RANDOM_STATE

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "results/ckpt/catboost.joblib.bak-uniform-start"
THREADS = 4  # CatBoost's result does not depend on the thread count
OUTPUT = "results/catboost_start_effect.csv"
ARMS = {False: "uniform", True: "prior"}  # prior_start -> column tag

stored = joblib.load(CHECKPOINT)
rows = []
for dataset in pd.read_csv("results/calibration_datasets.csv")["dataset"]:
    X, y, cat_cols = load_data_df(dataset)
    if len(X) > MAX_DATASET_ROWS:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), MAX_DATASET_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    n_classes = int(np.unique(y).size)
    folds = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    preds, labels, reproduced = {False: [], True: []}, [], 0.0
    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        for prior_start in ARMS:
            model = build_final_model("catboost", stored[dataset]["best_params"][fold], n_classes, cat_cols)
            model.set_params(prior_start=prior_start, thread_count=THREADS)
            preds[prior_start].append(model.fit(X_train, y[train_idx]).predict_proba(X_test))
        reproduced = max(reproduced, float(np.abs(preds[False][-1] - stored[dataset]["preds"][fold]).max()))
        labels.append(y[test_idx])
    y_all, share = np.concatenate(labels), np.bincount(y) / len(y)
    row = dict(dataset=dataset, n_rows=len(y), n_classes=n_classes,
               # 0 on balanced data, where the prior is the uniform start.
               kl_prior_uniform=float(np.sum(share * np.log(share * n_classes))),
               shrink=float(np.median([p["learning_rate"] * p["n_estimators"]
                                       for p in stored[dataset]["best_params"]])),
               max_diff_vs_stored=reproduced)
    for prior_start, tag in ARMS.items():
        prob = np.vstack(preds[prior_start])
        row |= {f"brier_{tag}": brier_score(y_all, prob),
                f"ece_{tag}": expected_calibration_error(y_all, prob),
                f"gap_{tag}": confidence_gap(y_all, prob),
                f"prauc_{tag}": float(np.mean([pr_auc_score(l, p) for l, p in zip(labels, preds[prior_start])]))}
    rows.append(row)
    print(f"{dataset:40s} Brier {row['brier_uniform']:.4f} -> {row['brier_prior']:.4f}  "
          f"ECE {row['ece_uniform']:.4f} -> {row['ece_prior']:.4f}  max|diff| {reproduced:.1e}", flush=True)

df = pd.DataFrame(rows)
df.round(6).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}")
