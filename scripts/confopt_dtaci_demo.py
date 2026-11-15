"""Does the DtACI fix make ConfOpt's interval widths adapt? One outer fold of one dataset,
50 trials of LightGBM's search with the published wrapper (lgbm_confopt) and with the fixed
one (lgbm_confopt_dtaci_trials100), printing the alphas the searcher predicts with.

Usage: uv run python -m scripts.confopt_dtaci_demo [dataset]
"""
import sys

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from benchmark.data import load_data_df
from benchmark.models.registry import MODELS
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)
dataset = sys.argv[1] if len(sys.argv) > 1 else "thyroid-newthyroid"
X, y, cat_cols = load_data_df(dataset)
train = next(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[0]
X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
inner = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for key in ("lgbm_confopt", "lgbm_confopt_dtaci_trials100"):
    spec = MODELS[key]
    sampler = spec.sampler(seed=spec.sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    alphas = []
    for _ in range(50):
        study.optimize(lambda t: spec.objective(t, X_train, y_train, inner, int(np.unique(y).size), cat_cols),
                       n_trials=1)
        alphas.append(np.round(sampler.get_thread_searcher_copy().sampler.fetch_alphas(), 3).tolist())
    changes = sum(a != b for a, b in zip(alphas, alphas[1:]))
    print(f"{key} on {dataset}, fold 0: alphas after trials 33, 40, 50: {alphas[32]}, {alphas[39]}, {alphas[49]}; "
          f"changed between consecutive trials {changes} times; best inner score {study.best_value:.4f}")
