"""FLAML and LightAutoML, the two AutoML frameworks added in October 2026, against the
boosters they build on: XGBoost, LightGBM and CatBoost tuned here on PR AUC and on log
loss, and against each other.

Exploratory: the 13 pairs are one Holm family. PR AUC is compared over every dataset both
runs score, calibration over the 108 datasets the calibration table uses.

Hours are wall time with the pauses of the FLAML run removed from the datasets they fell in: a
manual stop while it sat on cnae-9 and two power-guard pauses (~/.sdb-runs/power_guard.log,
2026-10-19 and 2026-10-20), placed by replaying the checkpoint times from the run's start. FLAML's
budget is wall clock, so a fold that was paused searched for less than its 300 s.

Usage: uv run python -m scripts.automl_vs_boosters
"""
import warnings

import joblib
import numpy as np
import pandas as pd

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, automl_path, compare
from config import AUTOML_SEC, N_OUTER_FOLDS

OUTPUT = "results/automl_vs_boosters.csv"
BOOSTERS = ["catboost", "lgbm", "xgboost", "catboost_logloss", "lgbm_logloss", "xgboost_logloss"]
FLAML_PAUSES = {"cnae-9": 26609, "connectionist-vowel-reduced": 1325, "thyroid-allbp": 662}  # seconds
# AutoML stores float32 probabilities whose rows sum to one only within 6e-7.
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])

automl = {fw: joblib.load(automl_path(fw, checkpoint=True)) for fw in ("flaml", "lightautoml")}
for d, pause in FLAML_PAUSES.items():
    automl["flaml"][d]["time"] -= pause
pairs = {f"{fw} vs {b}": (automl[fw], joblib.load(ckpt_path(b))) for fw in automl for b in BOOSTERS}
pairs["lightautoml vs flaml"] = (automl["lightautoml"], automl["flaml"])
out, diffs = compare(pairs, calibration)
out.drop(columns=["hours_base", "hours_variant"]).round(6).to_csv(OUTPUT, index=False)

pd.set_option("display.width", 250)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm"]]
      .round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))

# LightAutoML stacks a second level on multiclass tasks only.
n_classes = pd.read_csv("results/datasets.csv").set_index("dataset")["n_classes"]
for b in ("catboost", "lgbm", "xgboost"):
    d = diffs[f"lightautoml vs {b}"]
    binary = n_classes[d.index] == 2
    print(f"lightautoml vs {b}: {d[binary].mean():+.4f} on {binary.sum()} binary, "
          f"{d[~binary].mean():+.4f} on {(~binary).sum()} multiclass")

budget = N_OUTER_FOLDS * AUTOML_SEC
for fw, ckpt in automl.items():
    t = pd.Series({d: e["time"] for d, e in ckpt.items()})
    hours = t.sum() / 3600
    scored = [d for d, e in ckpt.items() if not np.isnan(np.mean(e["scores"]))]
    print(f"{fw}: {len(scored)} of {len(ckpt)} datasets scored, mean PR AUC "
          f"{np.mean([np.mean(ckpt[d]['scores']) for d in scored]):.4f}, {hours:.1f} h against the "
          f"{len(ckpt) * budget / 3600:.1f} h budget; over {budget} s on {(t > budget).sum()}, "
          f"by > 10 % on {(t > 1.1 * budget).sum()}, max {t.max():.0f} s ({t.idxmax()})")
print("Booster hours over FLAML's datasets: " + ", ".join(
    f"{b} {sum(e['time'] for d, e in joblib.load(ckpt_path(b)).items() if d in automl['flaml']) / 3600:.1f}"
    for b in BOOSTERS))
print(f"Saved {OUTPUT}")
