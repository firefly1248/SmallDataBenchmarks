"""Does ConfOpt score higher than TPE inside the search, and does that reach new rows?

scripts/search_budget_inner.py re-scored only the folds whose pick changed between 50 and 100
trials, on seed 0. Here every fold's chosen configuration is re-scored, for both samplers at 50
and 100 trials on sampler seeds 0 and 1 and at 50 trials on seeds 2 to 4 where those runs exist, with the search's own objective (optuna
FixedTrial) on the outer fold's training part: on the search's inner split (`search`) and on a
fresh inner split, seed RANDOM_STATE + 1 (`fresh`). `outer` is the stored outer-fold score.
A configuration shared by a run's 50- and 100-trial picks is scored once.

Not pre-registered: chosen after the results. The unit is the dataset (mean over its four
folds); ConfOpt - TPE per seed and averaged over the seeds, with a paired bootstrap CI and
Wilcoxon's p. Holm is over the seed 0 and seed 1 `search` tests at 50 trials; the rest is
raw p. The yardstick is one sampler's seed 1 - seed 0 on `search` at 50 trials and, with more
seeds, the spread of its per-seed means.

Usage: uv run python -m scripts.search_inner_all
"""
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import trim_mean, wilcoxon
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.models.registry import MODELS
from benchmark.stats import holm
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/search_inner_all.csv"
RUNS = {("confopt", 0): {50: "lgbm_confopt", 100: "lgbm_confopt_trials100"},
        ("confopt", 1): {50: "lgbm_confopt_seed1", 100: "lgbm_confopt_trials100_seed1"},
        ("tpe", 0): {50: "lgbm_rerun", 100: "lgbm_trials100"},
        ("tpe", 1): {50: "lgbm_seed1", 100: "lgbm_trials100_seed1"},
        **{("confopt", s): {50: f"lgbm_confopt_seed{s}"} for s in (2, 3, 4)},
        **{("tpe", s): {50: f"lgbm_seed{s}"} for s in (2, 3, 4)}}
RUNS = {k: run for k, run in RUNS.items() if all(os.path.exists(ckpt_path(m)) for m in run.values())}
SPLITS = {"search": RANDOM_STATE, "fresh": RANDOM_STATE + 1}
SCORES = ["search", "fresh", "outer"]
rng = np.random.default_rng(0)

ckpts = {model: joblib.load(ckpt_path(model)) for run in RUNS.values() for model in run.values()}
rows, tasks, missing = [], {}, []
for (sampler, seed), run in RUNS.items():
    for dataset in sorted(set().union(*(ckpts[m].keys() for m in run.values()))):
        for trials, model in run.items():
            entry = ckpts[model].get(dataset)
            if not entry or not entry["best_params"]:
                missing.append((model, dataset))
                continue
            for fold, params in enumerate(entry["best_params"]):
                key = (sampler, seed, dataset, fold, repr(sorted(params.items())))
                tasks.setdefault(key, (model, params))
                rows.append({"sampler": sampler, "seed": seed, "trials": trials, "dataset": dataset, "fold": fold,
                             "key": key, "outer": entry["scores"][fold]})
print(f"{len(rows)} rows, {len(tasks)} distinct configurations to score; missing: {missing or 'none'}")

data = {d: load_data_df(d) for d in {k[2] for k in tasks}}


def rescore(key, model, params):
    X, y, cat_cols = data[key[2]]
    train = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[key[3]][0]
    X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
    return key, {split: MODELS[model].objective(optuna.trial.FixedTrial(params), X_train, y_train,
                                                StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=s),
                                                int(np.unique(y).size), cat_cols)
                 for split, s in SPLITS.items()}


start = time.time()
# Four concurrent folds, as in the searches (ModelSpec.parallel_folds).
scored = dict(Parallel(n_jobs=N_OUTER_FOLDS, prefer="threads")(delayed(rescore)(k, *t) for k, t in tasks.items()))
print(f"Scored in {(time.time() - start) / 60:.1f} min")
df = pd.DataFrame(rows)
for split in SPLITS:
    df[split] = df.key.map(lambda k: scored[k][split])
df = df.drop(columns="key")[["sampler", "seed", "trials", "dataset", "fold"] + SCORES]
df.round(6).to_csv(OUTPUT, index=False)


def gap(label, diff):
    """Mean with a paired bootstrap 95 % CI, median, 10 % trimmed mean, ahead / behind and
    Wilcoxon's p of a per-dataset gap, as in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:40s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")
    return diff.mean(), p


print("\nChecks:")
per = df.groupby(["sampler", "seed", "trials", "dataset"])[SCORES].mean()
controls = pd.read_csv("results/confopt_controls.csv").set_index("comparison")["mean"]
for s in (0, 1):
    d = (per.loc[("confopt", s, 50), "outer"] - per.loc[("tpe", s, 50), "outer"]).mean()
    print(f"  outer ConfOpt - TPE at 50 trials, seed {s}: {d:+.4f} "
          f"(confopt_controls.csv {controls[f'ConfOpt - TPE, seed {s}']:+.4f})")
old = pd.read_csv("results/search_budget_inner.csv")
new = df[df.seed == 0].set_index(["sampler", "dataset", "fold", "trials"])
worst = max(abs(new.loc[(r.search, r.dataset, r.fold, n), split] - getattr(r, f"{split}{n}"))
            for r in old.itertuples() for split in SPLITS for n in (50, 100))
print(f"  search and fresh against search_budget_inner.csv, {len(old)} folds: largest difference {worst:.1e}")
wide = df.pivot_table(index=["sampler", "seed", "dataset", "fold"], columns="trials", values="search")
below = wide[wide[100] < wide[50] - 1e-9]
print(f"  search at 100 trials below 50 trials: {len(below)} of {len(wide)} folds")
print(f"  rows {len(df)} of {131 * N_OUTER_FOLDS * sum(map(len, RUNS.values()))}")

print("\nConfOpt - TPE per dataset (mean over its folds):")
results = {}
for trials in (50, 100):
    seeds = sorted({s for (_, s), run in RUNS.items() if trials in run})
    diff = {s: per.loc[("confopt", s, trials)] - per.loc[("tpe", s, trials)] for s in seeds}
    for m in SCORES:
        print(f" {m}, {trials} trials")
        for s in seeds:
            results[(m, trials, s)] = gap(f"seed {s}", diff[s][m])
        results[(m, trials, "mean")] = gap("mean of seeds 0 and 1", (diff[0][m] + diff[1][m]) / 2)
        if len(seeds) > 2:
            gap(f"mean of seeds {seeds[0]} to {seeds[-1]}", sum(diff[s][m] for s in seeds) / len(seeds))
print("\nHolm over the two per-seed search tests at 50 trials: " + ", ".join(
    f"seed {s} {p:.3g}" for s, p in holm({s: results[("search", 50, s)][1] for s in (0, 1)}).items()))
print("Yardstick, seed 1 - seed 0 on search at 50 trials:")
for sampler in ("confopt", "tpe"):
    gap(sampler, per.loc[(sampler, 1, 50), "search"] - per.loc[(sampler, 0, 50), "search"])
    means = {s: per.loc[(sampler, s, 50), "search"].mean() for (smp, s) in RUNS if smp == sampler}
    if len(means) > 2:
        print(f"  {sampler} mean search score per seed: " + ", ".join(f"{s} {v:.4f}" for s, v in means.items())
              + f"; SD {np.std(list(means.values()), ddof=1):.4f}")
print("Share of the search gap left on fresh and outer, mean of seeds:")
for trials in (50, 100):
    d = results[("search", trials, "mean")][0]
    print(f"  {trials} trials: fresh {results[('fresh', trials, 'mean')][0] / d:.0%}, "
          f"outer {results[('outer', trials, 'mean')][0] / d:.0%}")
print(f"Saved {OUTPUT}")
