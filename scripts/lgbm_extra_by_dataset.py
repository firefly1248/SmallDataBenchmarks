"""lgbm_extra against lgbm per dataset, for illustration, and how far the tuning pushed
LightGBM's regularisers from their defaults in each.

Brier pools the outer folds, as in the calibration table. ``min_gain_to_split`` and
``path_smooth`` are not in the search space, so they stay at their default of 0.

Usage: uv run python -m scripts.lgbm_extra_by_dataset
"""
import joblib
import pandas as pd

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.results import per_dataset

OUTPUT = "results/lgbm_extra_by_dataset.csv"
FEW_LEVELS = 5
meta = pd.read_csv("results/datasets.csv").set_index("dataset")

extra, base = (joblib.load(ckpt_path(m)) for m in ("lgbm_extra", "lgbm"))
extra, base = per_dataset(extra, set(extra)), per_dataset(base, set(base))
rows = []
for d in extra.index.intersection(base.index):
    X, _, _ = load_data_df(d)
    rows.append({"dataset": d, "n_rows": meta.loc[d, "n_rows"], "n_features": meta.loc[d, "n_features"],
                 "n_classes": meta.loc[d, "n_classes"],
                 f"share_features_le_{FEW_LEVELS}_values": round((X.nunique() <= FEW_LEVELS).mean(), 4),
                 "prauc_diff": round(extra.loc[d, "prauc"] - base.loc[d, "prauc"], 6),
                 "brier_diff": round(extra.loc[d, "brier"] - base.loc[d, "brier"], 6)})
out = pd.DataFrame(rows).sort_values("prauc_diff")
out.to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}: {len(out)} datasets, lgbm_extra minus lgbm")

print(f"\nShare of outer folds whose tuned value is away from the default, over the {len(out)} datasets:")
for model in ("lgbm", "lgbm_extra"):
    params = pd.DataFrame([p for d, e in joblib.load(ckpt_path(model)).items()
                           if d in set(out.dataset) for p in e["best_params"]])
    print(f"  {model:10s} folds {len(params)}  "
          f"reg_alpha (lambda_l1) > 1e-3: {(params.reg_alpha > 1e-3).mean():.0%}, > 0.1: "
          f"{(params.reg_alpha > 0.1).mean():.0%}  reg_lambda (lambda_l2) > 1e-3: "
          f"{(params.reg_lambda > 1e-3).mean():.0%}, > 0.1: {(params.reg_lambda > 0.1).mean():.0%}  "
          f"colsample_bytree (feature_fraction) < 0.9: {(params.colsample_bytree < 0.9).mean():.0%}")
