"""Follow-ups to the ConfOpt block, from the checkpoints and results/search_inner_all.csv
(scripts/search_inner_all.py). Not pre-registered: chosen after the results.

B. ConfOpt - TPE on outer PR AUC at 100 trials, per sampler seed and averaged.
C. The ahead / behind counts of each sampler's seed 1 - seed 0.
D. ConfOpt - TPE by dataset size (outer and search, 50 trials, mean of seeds 0 and 1): by
   tercile of rows and Spearman with the rows.
E. Calibration: the chosen parameters of the two samplers over the 108 calibration datasets
   (medians, share of folds at the edge of the range) and, per dataset, the Spearman of the
   ECE gap with the gap in optimism (search - outer) and in each parameter.

Usage: uv run python -m scripts.confopt_followups
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.models import objectives as o
from benchmark.results import CALIBRATION_POPULATION, per_dataset

SEEDS = {0: ("lgbm_confopt", "lgbm_rerun"), 1: ("lgbm_confopt_seed1", "lgbm_seed1")}
SEEDS100 = {0: ("lgbm_confopt_trials100", "lgbm_trials100"), 1: ("lgbm_confopt_trials100_seed1", "lgbm_trials100_seed1")}
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
inner = pd.read_csv("results/search_inner_all.csv")
rows = pd.read_csv("results/datasets.csv").set_index("dataset")["n_rows"]
space = o.lgbm_space()
rng = np.random.default_rng(0)


def load(model):
    return joblib.load(ckpt_path(model))


def prauc(model):
    return per_dataset(load(model), set())["prauc"].astype(float)


def gap(label, diff):
    """As in scripts/confopt_controls.py: mean with a paired bootstrap 95 % CI, median, 10 %
    trimmed mean, ahead / behind, Wilcoxon's p."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:34s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")


print("B. ConfOpt - TPE, outer PR AUC at 100 trials:")
d100 = {s: prauc(c) - prauc(t) for s, (c, t) in SEEDS100.items()}
for s, d in d100.items():
    gap(f"seed {s}", d)
gap("mean of seeds", (d100[0] + d100[1]) / 2)

print("\nC. Seed 1 - seed 0, outer PR AUC at 50 trials:")
for name, i in (("ConfOpt", 0), ("TPE", 1)):
    d = (prauc(SEEDS[1][i]) - prauc(SEEDS[0][i])).dropna()
    print(f"  {name}: ahead {(d > 0).sum()}, behind {(d < 0).sum()}, tied {(d == 0).sum()}, mean {d.mean():+.4f}")

print("\nD. ConfOpt - TPE at 50 trials, mean of seeds 0 and 1, by rows:")
per = inner[inner.trials == 50].groupby(["sampler", "seed", "dataset"])[["search", "outer"]].mean()
for m in ("outer", "search"):
    d = sum(per.loc[("confopt", s), m] - per.loc[("tpe", s), m] for s in (0, 1)) / 2
    band = pd.qcut(rows[d.index], 3, labels=["small", "medium", "large"])
    print(f" {m}: Spearman with rows {spearmanr(d, rows[d.index])[0]:+.2f} "
          f"(p {spearmanr(d, rows[d.index])[1]:.2g})")
    for b, g in d.groupby(band, observed=True):
        gap(f"{b} ({rows[g.index].min()}-{rows[g.index].max()} rows)", g)

print("\nE. Chosen parameters over the calibration folds:")
EDGE = 0.02  # share of the range, on the log scale for log-scaled parameters


def chosen(model):
    return pd.DataFrame([dict(p, dataset=d) for d, e in load(model).items() if d in calibration
                         for p in e["best_params"]])


def scaled(p, name):
    dist = space[name]
    f = np.log if dist.log else (lambda x: x)
    return (f(p[name]) - f(dist.low)) / (f(dist.high) - f(dist.low))


NUMERIC = [n for n in space if n != "class_weight"]
picks = {(s, i): chosen(SEEDS[s][i]) for s in SEEDS for i in (0, 1)}
for s in SEEDS:
    print(f" seed {s}: median ConfOpt / TPE, share at the edge of the range ConfOpt / TPE")
    for name in ["lr x trees"] + NUMERIC:
        c, t = (picks[(s, i)] for i in (0, 1))
        if name == "lr x trees":
            print(f"  {name:18s} {np.median(c.learning_rate * c.n_estimators):9.3g} / "
                  f"{np.median(t.learning_rate * t.n_estimators):9.3g}")
            continue
        edge = lambda p: np.mean((scaled(p, name) < EDGE) | (scaled(p, name) > 1 - EDGE))
        print(f"  {name:18s} {c[name].median():9.3g} / {t[name].median():9.3g}   {edge(c):.0%} / {edge(t):.0%}")

print(" Per dataset, Spearman of the ECE gap (ConfOpt - TPE) with:")
cal = sorted(calibration)
opt = inner[inner.trials == 50].assign(opt=lambda x: x.search - x.outer).groupby(["sampler", "seed", "dataset"]).opt.mean()
for s, (c, t) in SEEDS.items():
    ece = (per_dataset(load(c), calibration).loc[cal, "ece"] - per_dataset(load(t), calibration).loc[cal, "ece"]).astype(float)
    terms = {"optimism gap": opt.loc[("confopt", s)][cal] - opt.loc[("tpe", s)][cal]}
    for name in ["lr x trees"] + NUMERIC:
        val = (lambda p: np.log(p.learning_rate * p.n_estimators)) if name == "lr x trees" else \
              (lambda p, n=name: scaled(p, n))
        mean = {i: picks[(s, i)].assign(v=val).groupby("dataset").v.mean() for i in (0, 1)}
        terms[name] = (mean[0] - mean[1])[cal]
    print(f"  seed {s}: " + ", ".join(f"{k} {spearmanr(ece, v)[0]:+.2f} (p {spearmanr(ece, v)[1]:.2g})"
                                      for k, v in terms.items()))
