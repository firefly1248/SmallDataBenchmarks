"""Why CatBoost's ECE trailed the other boosters: the undertrained tail.

Optuna tunes on PR AUC, which ignores probability scale, so it is free to pick
a tiny learning_rate * n_estimators. XGBoost and LightGBM boost from the class
prior and stay near it; CatBoost started from uniform probabilities (0.5, or
1/K) and stayed near those. The fix is in ``CatBoostNativeWrapper.fit``.

Reads the checkpoints with predictions (not tracked; regenerable) and the
calibration figure's population, ``results/calibration_datasets.csv``. Writes
the per-dataset diagnostics to ``results/catboost_calibration.csv``.

Usage: uv run python -m scripts.catboost_calibration [catboost checkpoint]
       (default: results/ckpt/catboost.joblib.bak-uniform-start)
"""
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from benchmark.calibration import confidence_gap, expected_calibration_error
from benchmark.checkpoints import ckpt_path

CATBOOST = sys.argv[1] if len(sys.argv) > 1 else "results/ckpt/catboost.joblib.bak-uniform-start"
SOURCES = {"catboost": CATBOOST, "xgboost": ckpt_path("xgboost"),
           "lgbm": ckpt_path("lgbm"), "random_forest": ckpt_path("random_forest")}
SMALL_SHRINK = 5  # learning_rate * n_estimators, the median over outer folds
OUTPUT = "results/catboost_calibration.csv"

population = set(pd.read_csv("results/calibration_datasets.csv")["dataset"])
rows = []
for model, path in SOURCES.items():
    for dataset, entry in joblib.load(path).items():
        if dataset not in population:
            continue
        prob, y = np.vstack(entry["preds"]), np.concatenate(entry["labels"])
        params = entry["best_params"]
        rows.append(dict(
            model=model, dataset=dataset, n_classes=prob.shape[1],
            ece=expected_calibration_error(y, prob),
            confidence_gap=confidence_gap(y, prob),
            learning_rate=np.median([p.get("learning_rate", np.nan) for p in params]),
            shrink=np.median([p.get("learning_rate", np.nan) * p.get("n_estimators", np.nan)
                              for p in params])))
df = pd.DataFrame(rows)
df.round(5).to_csv(OUTPUT, index=False)

print(df.groupby("model").agg(ece_mean=("ece", "mean"), ece_median=("ece", "median"),
                              confidence_gap=("confidence_gap", "mean")).round(4).to_string())
for model in ("catboost", "xgboost", "lgbm"):
    s = df[df.model == model]
    small = s.shrink < SMALL_SHRINK
    print(f"{model:9s} learning_rate * n_estimators < {SMALL_SHRINK} on {small.sum():3d}/{len(s)}: "
          f"ECE {s[small].ece.mean():.4f} (gap {s[small].confidence_gap.mean():+.3f}), "
          f"elsewhere {s[~small].ece.mean():.4f}")
cb = df[df.model == "catboost"].set_index("dataset")
print(f"CatBoost Spearman(ECE, learning_rate * n_estimators) = {spearmanr(cb.ece, cb.shrink).statistic:.2f}")
wide = df.pivot(index="dataset", columns="model", values="ece")
trained = cb.index[cb.shrink >= SMALL_SHRINK]
for other in ("xgboost", "lgbm"):
    diff = (wide.catboost - wide[other]).loc[trained]
    print(f"On CatBoost's {len(trained)} other datasets, CatBoost vs {other}: ECE {diff.mean():+.4f}, "
          f"p = {wilcoxon(diff).pvalue:.2g}")
print(f"Saved {OUTPUT}")
