"""Variants of published models, each against the run it varies: the boosters tuned on log
loss instead of PR AUC, LightGBM with Extra Trees splits, and the AutoGluon re-run with
nothing excluded and Ray against the published run.

PR AUC is compared over every dataset both runs score, calibration over the 108 datasets
the calibration table uses. The four tuning variants share one Holm family; the
AutoGluon re-run is a single test.

Usage: uv run python -m scripts.variant_effects
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, automl_path, compare
from config import AUTOML_SEC, N_OUTER_FOLDS

OUTPUT = "results/variant_effects.csv"
PAIRS = [("xgboost_logloss", "xgboost"), ("catboost_logloss", "catboost"),
         ("lgbm_logloss", "lgbm"), ("lgbm_extra", "lgbm")]
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
meta = pd.read_csv("results/datasets.csv").set_index("dataset")

out, diffs = compare({f"{v} vs {b}": (joblib.load(ckpt_path(v)), joblib.load(ckpt_path(b)))
                      for v, b in PAIRS}, calibration)

new = joblib.load(automl_path("autogluon", checkpoint=True))
old = joblib.load(automl_path("autogluon", checkpoint=True) + ".bak-no-fastai")
ag, ag_diffs = compare({"autogluon nothing excluded + Ray vs published": (new, old)}, calibration)
ag_diff = ag_diffs["autogluon nothing excluded + Ray vs published"]
budget = N_OUTER_FOLDS * AUTOML_SEC
overrun = pd.Series({d: new[d]["time"] > 1.1 * budget for d in ag_diff.index})
ag["overrun_datasets"] = int(overrun.sum())
ag["overrun_hours"] = sum(max(0, new[d]["time"] - budget) for d in ag_diff.index) / 3600
out = pd.concat([out, ag], ignore_index=True)

out.round(6).to_csv(OUTPUT, index=False)
pd.set_option("display.width", 250)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm",
           "hours_base", "hours_variant"]].round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p",
           "gap_base", "gap_variant", "gap_p"]].round(4).to_string(index=False))

t_new = pd.Series({d: new[d]["time"] for d in ag_diff.index})
t_old = pd.Series({d: old[d]["time"] for d in ag_diff.index})
print(f"\nAutoGluon budget {budget} s per dataset: new run over it on {(t_new > budget).sum()}, by > 10 % on "
      f"{int(overrun.sum())}, by > 100 % on {(t_new > 2 * budget).sum()}, max {t_new.max():.0f} s; published run "
      f"over it on {(t_old > budget).sum()}, max {t_old.max():.0f} s")
print(f"PR AUC gain {ag_diff[overrun].mean():+.4f} on the {int(overrun.sum())} overruns "
      f"(p = {wilcoxon(ag_diff[overrun & (ag_diff != 0)]).pvalue:.2g}), {ag_diff[~overrun].mean():+.4f} on the rest "
      f"(p = {wilcoxon(ag_diff[~overrun & (ag_diff != 0)]).pvalue:.2g}), where the runs took "
      f"{t_new[~overrun].sum() / 3600:.1f} h and {t_old[~overrun].sum() / 3600:.1f} h")
inside = pd.Series(ag_diff.index.isin(sorted(calibration)), index=ag_diff.index)
for where, mask in (("the calibration datasets", inside), ("the rest", ~inside)):
    d = ag_diff[mask]
    print(f"On {where} ({mask.sum()}): {d.mean():+.4f} (p = {wilcoxon(d[d != 0]).pvalue:.2g}); overruns "
          f"{d[overrun].mean():+.4f} ({(mask & overrun).sum()}), within budget {d[~overrun].mean():+.4f}")

# What the tuning chose, over the calibration datasets: the budget learning_rate * trees, and
# LightGBM's class reweighting.
print("\nChosen parameters over the calibration datasets (share of outer folds):")
for base in ("xgboost", "catboost", "lgbm"):
    for m in (base, f"{base}_logloss"):
        params = [p for d, e in joblib.load(ckpt_path(m)).items() if d in calibration for p in e["best_params"]]
        lr_trees = np.array([p["learning_rate"] * p["n_estimators"] for p in params])
        balanced = np.mean([p.get("class_weight") == "balanced" for p in params])
        print(f"  {m:17s} folds {len(params)}  learning_rate * n_estimators < 5: {np.mean(lr_trees < 5):.0%}"
              + (f"  class_weight balanced: {balanced:.0%}" if base == "lgbm" else ""))

d = diffs["lgbm_extra vs lgbm"]
size = pd.qcut(meta.loc[d.index, "n_rows"], 3, labels=["small", "medium", "large"])
print("\nlgbm_extra - lgbm by size tercile (rows):")
print(pd.DataFrame({"diff": d, "size": size}).groupby("size", observed=True)["diff"]
      .agg(datasets="size", mean="mean", median="median", ahead=lambda s: (s > 0).sum(),
           behind=lambda s: (s < 0).sum()).round(4).to_string())
big = d[d < -0.05].sort_values()
print(f"\nmean {d.mean():+.4f}, median {d.median():+.4f}; losses / gains beyond "
      + ", ".join(f"{t}: {(d < -t).sum()}/{(d > t).sum()}" for t in (0.01, 0.02, 0.05)))
print(f"losses beyond 0.05 by size tercile: {size[big.index].value_counts().sort_index().to_dict()}")
print(meta.loc[big.index, ["n_rows", "n_features", "n_classes"]].assign(diff=big.round(3)).to_string())
print(f"\nSaved {OUTPUT}")
