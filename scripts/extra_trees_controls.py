"""The Extra Trees block: sklearn ExtraTreesClassifier against Random Forest, LightGBM
re-tuned with another sampler seed as the noise control for lgbm_extra, and the checks
on why lgbm_extra loses: few-valued features one-hot or as LightGBM categories on the
six rule-generated datasets, a cap of 2000 trees on the 21 datasets that lost more than
0.05, which thresholds LightGBM's random split can draw, and which features it never
splits on those 21.

The three tests over 131 datasets form one Holm family; the checks run on datasets
chosen for their lgbm_extra loss and are reported without a test. Calibration is over
the 108 calibration datasets, as in the calibration table.

Usage: uv run python -m scripts.extra_trees_controls
"""
from collections import Counter

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.models.registry import build_final_model
from benchmark.results import CALIBRATION_POPULATION, compare

OUTPUT = "results/extra_trees_controls.csv"
ENCODINGS = "results/lgbm_extra_encodings.csv"
CAP = "results/lgbm_extra_cap2000.csv"
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])


def load(model):
    return joblib.load(ckpt_path(model))


def prauc(model):
    return pd.Series({d: np.mean(e["scores"]) for d, e in load(model).items()})


def hours(model, datasets):
    return sum(load(model)[d]["time"] for d in datasets) / 3600


pd.set_option("display.width", 250)

out, diffs = compare({"extra_trees vs random_forest": (load("extra_trees"), load("random_forest")),
                      "lgbm_seed1 vs lgbm": (load("lgbm_seed1"), load("lgbm")),
                      "lgbm_extra vs lgbm_seed1": (load("lgbm_extra"), load("lgbm_seed1"))}, calibration)
out.to_csv(OUTPUT, index=False)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm",
           "hours_base", "hours_variant"]].round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))

seed, extra = diffs["lgbm_seed1 vs lgbm"], prauc("lgbm_extra") - prauc("lgbm")
extra = extra.dropna()
losers = extra[extra < -0.05].index
print(f"\nlgbm_seed1 - lgbm: sd {seed.std():.4f}, beyond 0.01 on {(seed.abs() > 0.01).sum()}, 0.02 on "
      f"{(seed.abs() > 0.02).sum()}, 0.05 on {(seed.abs() > 0.05).sum()} datasets; "
      f"on the {len(losers)} lgbm_extra losers {seed[losers].mean():+.4f}, at most {seed[losers].abs().max():.4f} either way")
print(f"lgbm_extra - lgbm: sd {extra.std():.4f}, beyond 0.01 on {(extra.abs() > 0.01).sum()}, 0.02 on "
      f"{(extra.abs() > 0.02).sum()}, 0.05 on {(extra.abs() > 0.05).sum()} datasets")
et = diffs["extra_trees vs random_forest"]
print(f"extra_trees - random_forest on the {len(losers)} lgbm_extra losers: {et[losers].mean():+.4f}, "
      f"{(et[losers] > 0).sum()} ahead, {(et[losers] < 0).sum()} behind; beyond 0.05: "
      f"{et[et.abs() > 0.05].round(3).to_dict()}")

enc = pd.DataFrame({m: prauc(m) for m in ("lgbm", "lgbm_extra", "lgbm_onehot", "lgbm_cat",
                                          "lgbm_extra_onehot", "lgbm_extra_cat")}).dropna()
enc.rename_axis("dataset").to_csv(ENCODINGS)
print(f"\nFew-valued features re-encoded, {len(enc)} rule-generated datasets:")
print(pd.concat([enc, enc.mean().to_frame("mean").T]).round(4).to_string())
print("lgbm_onehot and lgbm_extra_onehot identical:", np.allclose(enc.lgbm_onehot, enc.lgbm_extra_onehot, atol=1e-12))

cap = pd.DataFrame({m: prauc(m) for m in ("lgbm", "lgbm_extra", "lgbm_2000", "lgbm_extra_2000")}).dropna()
cap["gap_500"] = cap.lgbm_extra - cap.lgbm
cap["gap_2000"] = cap.lgbm_extra_2000 - cap.lgbm_2000
cap = cap.sort_values("gap_500")
cap.rename_axis("dataset").to_csv(CAP)
print(f"\nCap of 2000 trees, {len(cap)} datasets that lost more than 0.05:")
print(pd.concat([cap, cap.mean().to_frame("mean").T]).round(4).to_string())
print(f"gaps narrowed on {(cap.gap_2000 > cap.gap_500).sum()}; hours "
      + ", ".join(f"{m} {hours(m, cap.index):.1f}" for m in ("lgbm", "lgbm_2000", "lgbm_extra", "lgbm_extra_2000")))
for m in ("lgbm", "lgbm_extra", "lgbm_2000", "lgbm_extra_2000"):
    trees = np.array([p["n_estimators"] for d, e in load(m).items() if d in cap.index for p in e["best_params"]])
    print(f"  {m:16s} n_estimators median {np.median(trees):.0f}, above 500 in {np.mean(trees > 500):.0%} of folds")

print("\nThresholds LightGBM's extra_trees draws on one feature (20 trees x 10 seeds):")
rng = np.random.default_rng(0)
for levels in ([1, 2, 3], [1, 2, 3, 4], [1, 2], [0, 1]):
    x = rng.choice(levels, 2000)
    y = (x == max(levels)).astype(int)
    cuts = Counter()
    for seed in range(10):
        model = lgb.LGBMClassifier(n_estimators=20, num_leaves=4, min_child_samples=5, extra_trees=True,
                                   random_state=seed, verbose=-1).fit(x.reshape(-1, 1), y)
        for tree in model.booster_.dump_model()["tree_info"]:
            nodes = [tree["tree_structure"]]
            while nodes:
                node = nodes.pop()
                if "threshold" in node:
                    cuts[node["threshold"]] += 1
                    nodes += [node["left_child"], node["right_child"]]
    print(f"  levels {levels}: {dict(cuts) or 'never split'}")

print(f"\nFeatures lgbm_extra never splits on the {len(cap)} losers (tuned models refit on the full data):")
for d in cap.index:
    X, y, cat_cols = load_data_df(d)
    splits = {}
    for m in ("lgbm", "lgbm_extra"):
        model = build_final_model(m, load(m)[d]["best_params"][0], len(np.unique(y)), cat_cols)
        model.named_steps["lgbm"].set_params(n_jobs=2)
        splits[m] = model.fit(X, y).named_steps["lgbm"].booster_.feature_importance("split")
    dead = (splits["lgbm_extra"] == 0) & (splits["lgbm"] > 0)
    coded_1_2 = np.array([X[c].nunique() == 2 and set(X[c].dropna().unique()) <= {1, 2} for c in X.columns])
    print(f"  {d:36s} never split {dead.sum():2d} of {len(dead):3d} features, {(dead & coded_1_2).sum():2d} of the "
          f"{coded_1_2.sum():2d} coded 1/2")

et, cb = prauc("extra_trees"), prauc("catboost")
both = et.index.intersection(cb.index)
d = (et - cb)[both]
print(f"\nextra_trees vs catboost, outside the family: {d.mean():+.4f}, {(d > 0).sum()} ahead, {(d < 0).sum()} behind, "
      f"raw p = {wilcoxon(d[d != 0]).pvalue:.2g} (n = {len(d)})")
seconds = {m: pd.Series({d: load(m)[d]["time"] for d in extra.index}) for m in ("lgbm_seed1", "lgbm_extra", "lgbm")}
print(f"Same 131 datasets, October, one job at a time: lgbm_seed1 {hours('lgbm_seed1', extra.index):.2f} h, "
      f"lgbm_extra {hours('lgbm_extra', extra.index):.2f} h (medians {seconds['lgbm_seed1'].median():.0f} s and "
      f"{seconds['lgbm_extra'].median():.0f} s per dataset); the August lgbm run {hours('lgbm', extra.index):.2f} h")
print(f"Saved {OUTPUT}, {ENCODINGS}, {CAP}")
