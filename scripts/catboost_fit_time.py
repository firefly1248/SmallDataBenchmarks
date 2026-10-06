"""Fit time of one CatBoost model: the start, the chosen depth and tree count, and the rest.

On one outer fold's training data, times a fit with the uniform-start run's best
parameters from each start, then the same parameters with only the re-run's depth,
only its tree count, and the re-run's full parameters. The first arm runs again last,
so the two copies show the timing noise. Everything else is held, so the arms read:
  flag:   old params, prior / old params, uniform
  depth:  old params + new depth, prior / old params, prior
  trees:  old params + new n_estimators, prior / old params, prior
  params: new params, prior / old params, prior

Usage: uv run python -m scripts.catboost_fit_time [fold] [dataset ...]
"""
import sys
import time
import torch  # must precede catboost to win the OpenMP init race

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.models.build import build_final_model
from config import INNER_FIT_THREADS, MAX_DATASET_ROWS, N_OUTER_FOLDS, RANDOM_STATE

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUTPUT = "results/catboost_fit_time.csv"
old = joblib.load("results/ckpt/catboost.joblib.bak-uniform-start")
new = joblib.load(ckpt_path("catboost"))
datasets = sys.argv[2:] or sorted(d for d in new if d in old and np.isfinite(new[d]["time"]))

rows = []
for dataset in datasets:
    X, y, cat_cols = load_data_df(dataset)
    if len(X) > MAX_DATASET_ROWS:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), MAX_DATASET_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    train = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[FOLD][0]
    X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
    p_old, p_new = old[dataset]["best_params"][FOLD], new[dataset]["best_params"][FOLD]
    arms = [("old, uniform", p_old, False), ("old, prior", p_old, True),
            ("old + new depth, prior", p_old | {"depth": p_new["depth"]}, True),
            ("old + new trees, prior", p_old | {"n_estimators": p_new["n_estimators"]}, True),
            ("new, prior", p_new, True), ("old, uniform again", p_old, False)]
    row = dict(dataset=dataset, fold=FOLD, n_classes=int(np.unique(y).size), rows=len(X_train),
               depth_old=p_old["depth"], depth_new=p_new["depth"],
               trees_old=p_old["n_estimators"], trees_new=p_new["n_estimators"])
    for name, params, prior_start in arms:
        model = build_final_model("catboost", params, row["n_classes"], cat_cols)
        model.set_params(prior_start=prior_start, thread_count=INNER_FIT_THREADS)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        row[name] = time.perf_counter() - start
    rows.append(row)
    print(f"{dataset:32s} " + "  ".join(f"{name} {row[name]:.1f}s" for name, *_ in arms), flush=True)

df = pd.DataFrame(rows)
df.round(3).to_csv(OUTPUT, index=False)
base = df["old, prior"]
ratios = pd.DataFrame({"noise (uniform again / uniform)": df["old, uniform again"] / df["old, uniform"],
                       "flag (prior / uniform)": base / df["old, uniform"],
                       "depth": df["old + new depth, prior"] / base,
                       "trees": df["old + new trees, prior"] / base,
                       "params (new / old)": df["new, prior"] / base}).set_index(df.dataset)
print(ratios.round(2).to_string())
print("geometric mean:", np.exp(np.log(ratios).mean()).round(2).to_dict())
print(f"Saved {OUTPUT}")
