"""Two follow-ups to the ConfOpt block on all 131 datasets. Not pre-registered.

I. Five sampler seeds of each sampler at 50 trials: ConfOpt - TPE per seed and over the
   five, and the spread across seeds, to say whether ConfOpt is the noisier search.
J. Selected on log loss with class_weight "balanced" in both spaces (lgbm_confopt_logloss
   against lgbm_logloss, seeds 0 and 1), which separates the two changes of the earlier
   unweighted log-loss control: the selection metric and the class weights; the same
   comparison for that unweighted pair.
PR AUC over every dataset, ECE and Brier over the 108 calibration datasets; raw p.

Usage: uv run python -m scripts.confopt_seeds_logloss
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, per_dataset

CONFOPT = ["lgbm_confopt"] + [f"lgbm_confopt_seed{s}" for s in range(1, 5)]
TPE = ["lgbm_rerun"] + [f"lgbm_seed{s}" for s in range(1, 5)]
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
cal = sorted(calibration)
rng = np.random.default_rng(0)
tables = {}


def table(model):
    if model not in tables:
        tables[model] = per_dataset(joblib.load(ckpt_path(model)), calibration).astype(float)
    return tables[model]


def gap(label, diff):
    """As in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:44s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")


def metrics(label, a, b):
    gap(f"{label}, PR AUC", table(a).prauc - table(b).prauc)
    for m in ("ece", "brier"):
        gap(f"{label}, {m}", table(a).loc[cal, m] - table(b).loc[cal, m])


print("I. ConfOpt - TPE at 50 trials per sampler seed:")
for s, (c, t) in enumerate(zip(CONFOPT, TPE)):
    metrics(f"seed {s}", c, t)
mean = {name: {m: sum(table(x)[m] for x in models) / len(models) for m in ("prauc", "ece", "brier")}
        for name, models in (("ConfOpt", CONFOPT), ("TPE", TPE))}
print(" mean over the five seeds:")
gap("PR AUC", mean["ConfOpt"]["prauc"] - mean["TPE"]["prauc"])
for m in ("ece", "brier"):
    gap(m, mean["ConfOpt"][m][cal] - mean["TPE"][m][cal])
print(" spread across seeds:")
for name, models in (("ConfOpt", CONFOPT), ("TPE", TPE)):
    means = [table(x).prauc.mean() for x in models]
    eces = [table(x).loc[cal, "ece"].mean() for x in models]
    print(f"  {name:8s} mean PR AUC per seed {np.round(means, 4).tolist()}, SD {np.std(means, ddof=1):.4f}; "
          f"mean ECE per seed {np.round(eces, 4).tolist()}, SD {np.std(eces, ddof=1):.4f}")
sd = {name: pd.concat([table(x).prauc for x in models], axis=1).std(axis=1, ddof=1)
      for name, models in (("ConfOpt", CONFOPT), ("TPE", TPE))}
gap("per-dataset SD across seeds, ConfOpt - TPE", sd["ConfOpt"] - sd["TPE"])

print("\nJ. Selected on log loss with 'balanced' in both spaces:")
metrics("seed 0, ConfOpt - TPE", "lgbm_confopt_logloss", "lgbm_logloss")
metrics("seed 1, ConfOpt - TPE", "lgbm_confopt_logloss_seed1", "lgbm_logloss_seed1")
print(" selected on log loss without class weights in both spaces:")
metrics("seed 0, ConfOpt - TPE", "lgbm_confopt_logloss_unweighted", "lgbm_logloss_unweighted")
metrics("seed 1, ConfOpt - TPE", "lgbm_confopt_logloss_unweighted_seed1", "lgbm_logloss_unweighted_seed1")
print(" what 'balanced' changes on log loss, per sampler:")
metrics("ConfOpt seed 0, balanced - unweighted", "lgbm_confopt_logloss", "lgbm_confopt_logloss_unweighted")
metrics("ConfOpt seed 1, balanced - unweighted", "lgbm_confopt_logloss_seed1", "lgbm_confopt_logloss_unweighted_seed1")
metrics("TPE seed 0, balanced - unweighted", "lgbm_logloss", "lgbm_logloss_unweighted")
metrics("TPE seed 1, balanced - unweighted", "lgbm_logloss_seed1", "lgbm_logloss_unweighted_seed1")
print(" what log loss changes against PR AUC selection, per sampler:")
metrics("ConfOpt seed 0, log loss - PR AUC", "lgbm_confopt_logloss", "lgbm_confopt")
metrics("ConfOpt seed 1, log loss - PR AUC", "lgbm_confopt_logloss_seed1", "lgbm_confopt_seed1")
metrics("TPE seed 0, log loss - PR AUC", "lgbm_logloss", "lgbm_rerun")
metrics("TPE seed 1, log loss - PR AUC", "lgbm_logloss_seed1", "lgbm_seed1")
