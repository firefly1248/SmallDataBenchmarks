"""Did 100 trials find better configurations than 50, and did that carry to the outer fold?

The searches are deterministic, so a 100-trial run repeats its 50-trial run's first 50
trials: where the two picked different configurations, the 100-trial pick has the higher
inner CV score. Inner scores are not stored, so both picks are re-scored here with the
search's own objective (optuna FixedTrial) on the outer fold's training part, twice: on
the search's inner split, where the pick was made, and on a fresh inner split (seed
RANDOM_STATE + 1), which the selection never saw.

ConfOpt: lgbm_confopt against lgbm_confopt_trials100. TPE: lgbm_rerun against lgbm_trials100.

Usage: uv run python -m scripts.search_budget_inner
"""
import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.models.registry import MODELS
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/search_budget_inner.csv"
PAIRS = {"confopt": ("lgbm_confopt", "lgbm_confopt_trials100"), "tpe": ("lgbm_rerun", "lgbm_trials100")}
SPLITS = {"search": RANDOM_STATE, "fresh": RANDOM_STATE + 1}

tasks = []
for search, (short, long) in PAIRS.items():
    a, b = joblib.load(ckpt_path(short)), joblib.load(ckpt_path(long))
    for dataset in sorted(a.keys() & b.keys()):
        if not (a[dataset]["best_params"] and b[dataset]["best_params"]):
            continue
        for fold in range(N_OUTER_FOLDS):
            p50, p100 = a[dataset]["best_params"][fold], b[dataset]["best_params"][fold]
            if p50 != p100:
                tasks.append({"search": search, "dataset": dataset, "fold": fold, "p50": p50, "p100": p100,
                              "outer50": a[dataset]["scores"][fold], "outer100": b[dataset]["scores"][fold]})
print(f"{len(tasks)} folds with a changed pick")

data = {d: load_data_df(d) for d in {t["dataset"] for t in tasks}}


def rescore(t):
    X, y, cat_cols = data[t["dataset"]]
    train = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[t["fold"]][0]
    X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
    objective = MODELS[PAIRS[t["search"]][1]].objective
    row = {k: t[k] for k in ("search", "dataset", "fold", "outer50", "outer100")}
    for split, seed in SPLITS.items():
        inner = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=seed)
        for n in (50, 100):
            row[f"{split}{n}"] = objective(optuna.trial.FixedTrial(t[f"p{n}"]), X_train, y_train,
                                           inner, int(np.unique(y).size), cat_cols)
    return row


# Four concurrent folds, as in the searches (ModelSpec.parallel_folds).
df = pd.DataFrame(Parallel(n_jobs=N_OUTER_FOLDS, prefer="threads")(delayed(rescore)(t) for t in tasks))
for m in ("search", "fresh", "outer"):
    df[f"d_{m}"] = df[f"{m}100"] - df[f"{m}50"]
df.round(6).to_csv(OUTPUT, index=False)

for search, g in df.groupby("search", sort=False):
    print(f"\n{search}: {len(g)} folds; search-split gain <= 0 on {(g.d_search <= 0).sum()}")
    for m in ("search", "fresh", "outer"):
        d = g[f"d_{m}"]
        print(f"  {m:6s} gain mean {d.mean():+.4f} median {d.median():+.4f}, up {(d > 0).sum()} "
              f"down {(d < 0).sum()}, p {wilcoxon(d[d != 0]).pvalue:.2g}")
    print(f"  Spearman with the outer gain: search {spearmanr(g.d_search, g.d_outer)[0]:+.2f}, "
          f"fresh {spearmanr(g.d_fresh, g.d_outer)[0]:+.2f}")
print(f"Saved {OUTPUT}")
