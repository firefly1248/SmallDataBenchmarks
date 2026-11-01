"""Final-fit noise of DART: refit every kept lgbm_dart and lgbm model under other LightGBM seeds.

The search is not repeated. Each outer fold's stored best_params are refitted on the same training rows with
random_state 0 (the stored fit) and 1 to 3; LightGBM derives its drop seed and its row and column sampling from
random_state. It answers two questions for the DART notes: how far a fold's score moves with the fit seed alone,
and whether the folds far below lgbm (pendigits, appendicitis, colon32) are so under every seed or under one.

Exploratory. Usage: uv run python -m scripts.dart_refit_seeds [dataset ...]   (no names: every dataset of lgbm_dart)
"""
import torch  # must precede lightgbm, as in optuna_models.py  # noqa: F401
import sys

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models.registry import MODELS
from config import N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/dart_refit_seeds.csv"
SEEDS = [0, 1, 2, 3]     # 0 is the stored fit
BASES = ["lgbm", "lgbm_dart"]
FOLD_GAP = 0.10          # a fold this far below the lgbm fold
WORKERS = 8


def refit(dataset, params):
    X, y, cat_cols = load_data_df(dataset)
    n_classes = int(np.unique(y).size)
    splits = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    rows = []
    for fold, (train, test) in enumerate(splits):
        X_train, X_test = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
        for model_name in BASES:
            for seed in SEEDS:
                model = MODELS[model_name].build(params[model_name][fold], n_classes, cat_cols, n_jobs=1)
                model.set_params(lgbm__random_state=seed).fit(X_train, y[train])
                rows.append({"dataset": dataset, "model": model_name, "fold": fold, "seed": seed,
                             "score": pr_auc_score(y[test], model.predict_proba(X_test))})
    return rows


ckpt = {m: joblib.load(ckpt_path(m)) for m in BASES}
datasets = sys.argv[1:] or list(ckpt["lgbm_dart"])
rows = sum(Parallel(n_jobs=WORKERS)(delayed(refit)(d, {m: ckpt[m][d]["best_params"] for m in BASES}) for d in datasets), [])
df = pd.DataFrame(rows)
df["stored"] = [ckpt[m][d]["scores"][f] for m, d, f in zip(df.model, df.dataset, df.fold)]
df.to_csv(OUTPUT, index=False)

zero = df[df.seed == 0]
print(f"{len(datasets)} datasets, {df.groupby(['dataset', 'fold']).ngroups} folds, seeds {SEEDS}")
print(f"Seed 0 reproduces the stored fold score: {(zero.score - zero.stored).abs().lt(1e-9).mean():.1%} of fits, "
      f"largest difference {(zero.score - zero.stored).abs().max():.2e}")

sd = df.groupby(["model", "dataset", "fold"]).score.std(ddof=1)
print("\nSd of a fold's score over the four fit seeds:")
print(sd.groupby("model").agg(median="median", p90=lambda s: s.quantile(0.9), max="max").round(4).to_string())

wide = df.pivot_table(index=["dataset", "fold", "seed"], columns="model", values="score")
wide["gap"] = wide.lgbm_dart - wide.lgbm
by_seed = wide.groupby("seed").gap.agg(mean="mean", folds_below=lambda g: int((g < -FOLD_GAP).sum()))
print(f"\nlgbm_dart - lgbm over folds, by fit seed (folds more than {FOLD_GAP} below counted):")
print(by_seed.round(4).to_string())
means = wide.groupby(["dataset", "seed"]).gap.mean().groupby("seed").mean()
print(f"Mean gap over datasets: seed 0 {means[0]:+.4f}; seeds 1-3 mean {means.loc[1:].mean():+.4f}, "
      f"range {means.loc[1:].min():+.4f} to {means.loc[1:].max():+.4f}")
hit = wide[wide.gap < -FOLD_GAP].reset_index().groupby(["dataset", "fold"]).seed.apply(list)
print("\nFolds below the threshold, with the fit seeds at which they are:")
print(hit.to_string() if len(hit) else "none")
print(f"Saved {OUTPUT}")
