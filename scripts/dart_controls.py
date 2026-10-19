"""The DART block: LightGBM with forced DART dropout, drop rates tuned with the rest of the
space, against the published LightGBM and against LightGBM re-tuned with another sampler
seed; XGBoost re-tuned with another sampler seed as its search-noise control.

Pre-registered: one Holm family of four (lgbm_dart and xgboost_dart, each against its
base and its seed control). xgboost_dart runs last, so until it lands the Holm column is
the bound 4 x raw p, which Holm never exceeds. The seed controls are reported without a
test. Calibration is over the 108 calibration datasets, raw p.

Hours compare only runs with parallel outer folds: lgbm_dart against lgbm_seed1 re-timed
under --run partime, with the two pauses of that run removed.

Usage: uv run python -m scripts.dart_controls
"""
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, compare

OUTPUT = "results/dart_controls.csv"
# Wall-clock pauses of the partime run (power guard, then a manual stop), epoch seconds.
EEST = timezone(timedelta(hours=3))
PAUSES = [(1792304254, 1792305099),
          (datetime(2026, 10, 18, 10, 43, 36, tzinfo=EEST).timestamp(),
           datetime(2026, 10, 18, 12, 48, 34, tzinfo=EEST).timestamp())]
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])


def load(model, run=""):
    return joblib.load(ckpt_path(model, run))


def prauc(ckpt):
    return pd.Series({d: np.mean(e["scores"]) for d, e in ckpt.items()})


def tail(label, diff):
    diff = diff.dropna()
    print(f"  {label:24s} n {len(diff)}, mean {diff.mean():+.4f}, sd {diff.std():.4f}; beyond 0.02: "
          f"{(diff > 0.02).sum()} up, {(diff < -0.02).sum()} down; beyond 0.05: {(diff > 0.05).sum()} up, "
          f"{(diff < -0.05).sum()} down")


def hours(ckpt):
    """Wall hours with the pauses removed."""
    return sum(e["time"] - sum(max(0, min(e["start"] + e["time"], b) - max(e["start"], a)) for a, b in PAUSES)
               for e in ckpt.values()) / 3600


pd.set_option("display.width", 250)
lgbm, seed1, dart = load("lgbm"), load("lgbm_seed1"), load("lgbm_dart")
xgb, xseed1 = load("xgboost"), load("xgboost_seed1")

out, _ = compare({"lgbm_dart vs lgbm": (dart, lgbm), "lgbm_dart vs lgbm_seed1": (dart, seed1)}, calibration)
out["p_holm_bound"] = (4 * out.p_raw).clip(upper=1)
out.drop(columns=["p_holm", "hours_base", "hours_variant"]).to_csv(OUTPUT, index=False)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm_bound"]]
      .round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))

print("\nPer-dataset PR AUC differences:")
tail("lgbm_seed1 - lgbm", prauc(seed1) - prauc(lgbm))
tail("xgboost_seed1 - xgboost", prauc(xseed1) - prauc(xgb))
tail("lgbm_dart - lgbm", prauc(dart) - prauc(lgbm))
tail("lgbm_dart - lgbm_seed1", prauc(dart) - prauc(seed1))
gap = (prauc(dart) - prauc(lgbm)).dropna()
print(f"  lgbm_dart - lgbm without pendigits: {gap.drop('pendigits').mean():+.4f}; pendigits fold scores "
      f"{np.round(dart['pendigits']['scores'], 3).tolist()}")

print("\nChosen parameters over the outer folds:")
for name, ckpt in (("lgbm", lgbm), ("lgbm_seed1", seed1), ("lgbm_dart", dart)):
    p = pd.DataFrame([bp for e in ckpt.values() if e["best_params"] for bp in e["best_params"]])
    line = (f"  {name:11s} {len(p)} folds: n_estimators > 450 in {np.mean(p.n_estimators > 450):.0%}, "
            f"median learning rate {p.learning_rate.median():.3f}")
    if "drop_rate" in p:
        weak = (p.drop_rate < 0.02) | (p.skip_drop > 0.45)
        line += (f", weak-dropout corner (drop_rate < 0.02 or skip_drop > 0.45) in {weak.mean():.0%}, "
                 f"median drop_rate {p.drop_rate.median():.3f}, median skip_drop {p.skip_drop.median():.2f}")
    print(line)

print(f"\nHours with parallel outer folds, pauses removed: lgbm_seed1 {hours(load('lgbm_seed1', 'partime')):.2f}, "
      f"lgbm_dart {hours(dart):.2f}")
print(f"Saved {OUTPUT}")
