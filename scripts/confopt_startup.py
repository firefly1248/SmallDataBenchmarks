"""Block K: does a lucky start explain the per-seed ConfOpt - TPE differences?

Both samplers open with 10 random trials whose configurations depend only on the sampler seed
(the same set in every study of a run). Here the 10 opening configurations of each sampler on
each of the five seeds are scored with the search's own objective (optuna FixedTrial) on the
search's inner split (seed RANDOM_STATE) of every outer fold of every dataset, and the best of
the 10 is compared with the run's final pick. Not pre-registered.

The configurations come from a 10-trial study on one small dataset and are checked against the
studies kept under --run traj (TPE and ConfOpt seed 0, ConfOpt seed 1 through its DtACI run,
whose first 33 trials are ConfOpt's); the scores are checked against the trajectories' values.

Usage: uv run python -m scripts.confopt_startup [dataset ...]  (datasets for a smoke test)
"""
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, trim_mean, wilcoxon
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.models.registry import MODELS
from benchmark.nested_cv import _storage
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/confopt_startup.csv"
N_START = 10
SEEDS = range(5)
PAIRS = {**{("tpe", s): "lgbm_rerun" if s == 0 else f"lgbm_seed{s}" for s in SEEDS},
         **{("confopt", s): "lgbm_confopt" if s == 0 else f"lgbm_confopt_seed{s}" for s in SEEDS}}
TRAJ = {("tpe", 0): "lgbm_trials100", ("confopt", 0): "lgbm_confopt_trials100",
        ("confopt", 1): "lgbm_confopt_dtaci_trials100_seed1"}
PROBE = "thyroid-newthyroid"
rng = np.random.default_rng(0)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner(X, y, fold):
    train = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[fold][0]
    return X.iloc[train].reset_index(drop=True), y[train]


X, y, cat_cols = load_data_df(PROBE)
X_probe, y_probe = inner(X, y, 0)
starts = {}
for pair, model in PAIRS.items():
    spec = MODELS[model]
    study = optuna.create_study(direction="maximize", sampler=spec.sampler(seed=spec.sampler_seed))
    study.optimize(lambda t: spec.objective(t, X_probe, y_probe, StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                                            int(np.unique(y).size), cat_cols), n_trials=N_START)
    starts[pair] = [t.params for t in sorted(study.trials, key=lambda t: t.number)]
print("Opening configurations against the kept studies:")
for pair, model in TRAJ.items():
    storage = _storage(model, "traj")
    names = optuna.study.get_all_study_names(storage)
    same = sum([t.params for t in sorted(optuna.load_study(study_name=n, storage=storage).trials,
                                         key=lambda t: t.number)[:N_START]] == starts[pair] for n in names)
    print(f"  {pair[0]} seed {pair[1]}: {same} of {len(names)} studies open with the same 10")

datasets = sys.argv[1:] or sorted(joblib.load(ckpt_path("lgbm_confopt")))
data = {d: load_data_df(d) for d in datasets}
tasks = {}
for pair, configs in starts.items():
    for i, params in enumerate(configs):
        tasks.setdefault(repr(sorted(params.items())), (PAIRS[pair], params))


def score(dataset, fold, key):
    model, params = tasks[key]
    X, y, cat_cols = data[dataset]
    X_train, y_train = inner(X, y, fold)
    return (dataset, fold, key), MODELS[model].objective(
        optuna.trial.FixedTrial(params), X_train, y_train,
        StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE), int(np.unique(y).size), cat_cols)


start = time.time()
jobs = [(d, f, k) for d in datasets for f in range(N_OUTER_FOLDS) for k in tasks]
print(f"\n{len(tasks)} distinct configurations x {len(datasets)} datasets x {N_OUTER_FOLDS} folds = {len(jobs)} fits of the inner CV")
scored = dict(Parallel(n_jobs=N_OUTER_FOLDS, prefer="threads")(delayed(score)(*j) for j in jobs))
print(f"Scored in {(time.time() - start) / 60:.1f} min")
df = pd.DataFrame([{"sampler": p[0], "seed": p[1], "config": i + 1, "dataset": d, "fold": f,
                    "search": scored[(d, f, repr(sorted(c.items())))]}
                   for p, configs in starts.items() for i, c in enumerate(configs)
                   for d in datasets for f in range(N_OUTER_FOLDS)])
df.round(6).to_csv(OUTPUT, index=False)

traj = pd.read_csv("results/search_trajectories.csv")
worst = 0.0
for pair, model in TRAJ.items():
    t = traj[(traj.model == model) & (traj.trial <= N_START) & traj.dataset.isin(datasets)]
    mine = df[(df.sampler == pair[0]) & (df.seed == pair[1])].set_index(["dataset", "fold", "config"]).search
    worst = max([worst] + [abs(mine[(r.dataset, r.fold, r.trial)] - r.value) for r in t.itertuples()])
print(f"Scores against the trajectories' first {N_START} trials: largest difference {worst:.1e}")


def gap(label, diff):
    """As in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue if (diff != 0).sum() > 1 else float("nan")
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:34s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")
    return diff


best = df.groupby(["sampler", "seed", "dataset", "fold"]).search.max().groupby(["sampler", "seed", "dataset"]).mean()
mean10 = df.groupby(["sampler", "seed", "dataset"]).search.mean()
final = pd.read_csv("results/search_inner_all.csv").query("trials == 50 and dataset in @datasets") \
    .groupby(["sampler", "seed", "dataset"])[["search", "outer"]].mean()
print("\nMean over datasets per sampler seed (best of the 10 opening configurations per fold; final 50-trial pick):")
for sampler in ("confopt", "tpe"):
    for s in SEEDS:
        print(f"  {sampler:8s} seed {s}: best of 10 {best[(sampler, s)].mean():.4f}, mean of 10 {mean10[(sampler, s)].mean():.4f}, "
              f"final search {final.loc[(sampler, s), 'search'].mean():.4f}, outer {final.loc[(sampler, s), 'outer'].mean():.4f}")
print("\nConfOpt - TPE per seed:")
rows = []
for s in SEEDS:
    start_gap = gap(f"seed {s}, best of 10", best[("confopt", s)] - best[("tpe", s)])
    final_gap = final.loc[("confopt", s)] - final.loc[("tpe", s)]
    rho = spearmanr(start_gap, final_gap.search[start_gap.index])[0]
    rows.append({"seed": s, "start": start_gap.mean(), "search": final_gap.search.mean(),
                 "outer": final_gap.outer.mean(), "rho": rho})
table = pd.DataFrame(rows).set_index("seed")
print("\nPer seed: ConfOpt - TPE best of 10, final search and outer gaps at 50 trials, and per dataset the Spearman"
      " of the opening gap with the final search gap:")
print(table.round(4).to_string())
print(f"Across the five seeds, Pearson of the opening gap with the final search gap {table.start.corr(table.search):+.2f}, "
      f"with the outer gap {table.start.corr(table.outer):+.2f} (n 5, descriptive)")
print(f"Saved {OUTPUT}")
