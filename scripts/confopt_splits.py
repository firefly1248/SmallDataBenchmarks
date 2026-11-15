"""ConfOpt against TPE on other outer splits (block H): lgbm_confopt on split seeds 1 and 2
over the 30 datasets of scripts/split_seeds.py, against lgbm (TPE seed 0) and lgbm_seed1 (TPE
seed 1) on the same splits; the default split (seed 0) for reference, against lgbm_rerun and
lgbm_seed1. Not pre-registered.

Usage: uv run python -m scripts.confopt_splits
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, per_dataset

RUNS = {0: ("", "lgbm_rerun"), 1: ("split1", "lgbm"), 2: ("split2", "lgbm")}
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
rng = np.random.default_rng(0)


def gap(label, diff):
    """As in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:36s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")


def table(model, run, datasets):
    ckpt = joblib.load(ckpt_path(model, run))
    return per_dataset({d: ckpt[d] for d in datasets}, calibration)


datasets = sorted(joblib.load(ckpt_path("lgbm_confopt", "split1")))
print(f"{len(datasets)} datasets; ConfOpt - TPE per outer split seed:")
for split, (run, tpe0) in RUNS.items():
    conf = table("lgbm_confopt", run, datasets)
    for label, tpe in ((f"TPE seed 0 ({tpe0})", tpe0), ("TPE seed 1 (lgbm_seed1)", "lgbm_seed1")):
        other = table(tpe, run, datasets)
        gap(f"split {split}, PR AUC vs {label}", (conf.prauc - other.prauc).astype(float))
        cal = conf.ece.dropna().index
        gap(f"split {split}, ECE vs {label}", (conf.ece[cal] - other.ece[cal]).astype(float))
