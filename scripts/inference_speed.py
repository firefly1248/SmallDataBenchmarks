"""Time fit and predict separately, on the published models, without re-tuning.

The checkpoints' ``time`` is the whole nested CV, tuning included, so it cannot
say what a fitted model costs to use. Each outer fold's final model is rebuilt
from its stored ``best_params`` and refit on the same rows, then timed on the
test fold and on a single row. The rebuilt predictions are compared with the
stored ones, so the timing is of the published model and not a lookalike.

Run on an otherwise idle machine: every number here is wall clock.

Usage: .venv/bin/python -m scripts.inference_speed [n_datasets] [dataset ...]
"""
import torch  # must precede xgboost / lightgbm / catboost to win the OpenMP init race
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import datasets_to_run, load_data_df
from benchmark.models.build import build_final_model
from benchmark.models.grid_search import GRID_SEARCH_MODELS, build_grid_search
from config import MAX_DATASET_ROWS, N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

warnings.filterwarnings("ignore")

# (model, device, outer folds timed). TabFM ran on MPS; on CPU it is 8-55x
# slower, so one fold is enough to place it.
RUNS = [
    ("catboost", None, N_OUTER_FOLDS), ("xgboost", None, N_OUTER_FOLDS),
    ("lgbm", None, N_OUTER_FOLDS), ("random_forest", None, N_OUTER_FOLDS),
    ("tabpfn35", None, N_OUTER_FOLDS), ("tabpfn35fast", None, N_OUTER_FOLDS),
    ("tabicl", None, N_OUTER_FOLDS),
    ("tabfm", "mps", N_OUTER_FOLDS), ("tabfm", "cpu", 1),
]
OUTPUT = "results/inference_speed.csv"


def rebuild(model_name, best_params, n_classes, cat_cols, device):
    if model_name in GRID_SEARCH_MODELS:
        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        model = clone(build_grid_search(model_name, inner_cv, cat_cols).estimator).set_params(**best_params)
    else:
        model = build_final_model(model_name, best_params, n_classes, cat_cols)
    return model.set_params(device=device) if device else model


def timed(f):
    start = time.perf_counter()
    out = f()
    return out, time.perf_counter() - start


if __name__ == "__main__":
    ckpts = {m: joblib.load(ckpt_path(m)) for m in {m for m, _, _ in RUNS}}
    if len(sys.argv) > 2:
        datasets = sys.argv[2:]
    else:
        # Evenly spaced by size over the datasets every model can be rebuilt on,
        # so the scaling with training rows is visible. TabICL's first run did
        # not store its grid choice, on 49 datasets.
        eligible = sorted((sum(map(len, ckpts["tabfm"][d]["labels"])), d) for d in datasets_to_run()
                          if all(ckpts[m].get(d, {}).get("preds") is not None
                                 and ckpts[m][d].get("best_params") is not None for m in ckpts))
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
        datasets = [eligible[i][1] for i in np.linspace(0, len(eligible) - 1, n).round().astype(int)]

    rows = []
    for dataset in datasets:
        X, y, cat_cols = load_data_df(dataset)
        if len(X) > MAX_DATASET_ROWS:
            idx = np.random.default_rng(RANDOM_STATE).choice(len(X), MAX_DATASET_ROWS, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]
        n_classes = int(np.unique(y).size)
        folds = list(StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                     random_state=RANDOM_STATE).split(X, y))
        print(f"{dataset}  shape={X.shape}", flush=True)
        for model_name, device, n_folds in RUNS:
            entry = ckpts[model_name][dataset]
            for fold, (train_idx, test_idx) in enumerate(folds[:n_folds]):
                X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
                model = rebuild(model_name, entry["best_params"][fold], n_classes, cat_cols, device)
                _, fit_s = timed(lambda: model.fit(X_train, y[train_idx]))
                pred, predict_s = timed(lambda: model.predict_proba(X_test))
                one_row_s = np.median([timed(lambda: model.predict_proba(X_test.iloc[:1]))[1] for _ in range(3)])
                rows.append(dict(model=model_name, device=device or "cpu", dataset=dataset,
                                 n_train=len(train_idx), n_test=len(test_idx), n_features=X.shape[1],
                                 n_classes=n_classes, fold=fold, fit_s=fit_s, predict_s=predict_s,
                                 predict_1row_s=one_row_s,
                                 max_abs_diff=float(np.abs(pred - entry["preds"][fold]).max())))
                # TabFM loads its 1.6B weights per fit; without this, nursery-4class ran out of MPS memory.
                del model
                if device == "mps":
                    torch.mps.empty_cache()
            last = pd.DataFrame(rows[-n_folds:])
            print(f"  {model_name:13s} {device or 'cpu':3s}  fit {last.fit_s.mean():8.2f}s  "
                  f"predict {last.predict_s.mean():8.3f}s  1 row {last.predict_1row_s.mean():7.3f}s  "
                  f"max|diff| {last.max_abs_diff.max():.1e}", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print("Saved", OUTPUT)
