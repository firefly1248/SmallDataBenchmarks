"""The DART block: LightGBM with forced DART dropout, drop rates tuned with the rest of the
space, against the published LightGBM and against LightGBM re-tuned with another sampler
seed; XGBoost re-tuned with another sampler seed as its search-noise control.

Pre-registered: one Holm family of four (lgbm_dart and xgboost_dart, each against its
base and its seed control), over the datasets both sides score. xgboost_dart skips both
plant-species datasets (100 classes, projected past the 12 h limit; agreed 2026-10-28), so
its pairs have 129. The seed controls are reported without a test. Calibration is over the
108 calibration datasets, raw p.

Hours compare only runs with parallel outer folds, pauses removed: lgbm_dart against
lgbm_seed1 re-timed under --run partime, xgboost_dart against xgboost_seed1. xgboost_dart's
multiple-features hit the 12 h limit and resumed its stored trials with no limit; its time
is the resumed part plus the 12 h. The datasets cut by the stops for other runs (micro-mass,
monks1, mushroom, plant-species-leaves-margin) had their studies dropped and restarted from
scratch, so their stored time is the whole search.

A follow-up outside the family asks whether DART's calibration loss is DART's or the PR AUC
selection's: lgbm_dart_logloss, the same search selected on log loss, against LightGBM
selected on log loss.

Usage: uv run python -m scripts.dart_controls
"""
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, compare, per_dataset

OUTPUT = "results/dart_controls.csv"
# Wall-clock pauses of the partime run (power guard, then a manual stop), epoch seconds.
EEST = timezone(timedelta(hours=3))
PAUSES = [(1792304254, 1792305099),
          (datetime(2026, 10, 18, 10, 43, 36, tzinfo=EEST).timestamp(),
           datetime(2026, 10, 18, 12, 48, 34, tzinfo=EEST).timestamp()),
          (1792819352, 1792819412)]  # the guard's one-minute pause in xgboost_dart's run
TIMED_OUT = {("xgboost_dart", "multiple-features"): 12 * 3600}
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


def seconds(ckpt, model=""):
    """Wall seconds per dataset with the pauses removed and a timed-out segment added."""
    return pd.Series({d: e["time"] + TIMED_OUT.get((model, d), 0)
                      - sum(max(0, min(e["start"] + e["time"], b) - max(e["start"], a)) for a, b in PAUSES)
                      for d, e in ckpt.items()})


def hours(ckpt, model=""):
    return seconds(ckpt, model).sum() / 3600


pd.set_option("display.width", 250)
lgbm, seed1, dart = load("lgbm"), load("lgbm_seed1"), load("lgbm_dart")
xgb, xseed1 = load("xgboost"), load("xgboost_seed1")

xdart = load("xgboost_dart")
out, _ = compare({"lgbm_dart vs lgbm": (dart, lgbm), "lgbm_dart vs lgbm_seed1": (dart, seed1),
                  "xgboost_dart vs xgboost": (xdart, xgb), "xgboost_dart vs xgboost_seed1": (xdart, xseed1)},
                 calibration)
out.drop(columns=["hours_base", "hours_variant"]).to_csv(OUTPUT, index=False)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm"]]
      .round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))
cal = sorted(calibration)
ece = lambda ckpt: per_dataset(ckpt, calibration).loc[cal, "ece"].astype(float)
for label, (variant, base) in {"lgbm_dart vs lgbm": (dart, lgbm), "lgbm_dart vs lgbm_seed1": (dart, seed1),
                               "xgboost_dart vs xgboost": (xdart, xgb),
                               "xgboost_dart vs xgboost_seed1": (xdart, xseed1)}.items():
    rise = ece(variant) - ece(base)
    print(f"  {label}: median ECE rise {rise.median():+.4f}, up on {(rise > 0).sum()} of {len(rise)}")

print("\nPer-dataset PR AUC differences:")
tail("lgbm_seed1 - lgbm", prauc(seed1) - prauc(lgbm))
tail("xgboost_seed1 - xgboost", prauc(xseed1) - prauc(xgb))
tail("lgbm_dart - lgbm", prauc(dart) - prauc(lgbm))
tail("lgbm_dart - lgbm_seed1", prauc(dart) - prauc(seed1))
tail("xgboost_dart - xgboost", prauc(xdart) - prauc(xgb))
tail("xgboost_dart - xgboost_seed1", prauc(xdart) - prauc(xseed1))
gap = (prauc(dart) - prauc(lgbm)).dropna()
print(f"  lgbm_dart - lgbm without pendigits: {gap.drop('pendigits').mean():+.4f}; pendigits fold scores "
      f"{np.round(dart['pendigits']['scores'], 3).tolist()}")

print("\nChosen parameters over the outer folds:")
for name, ckpt in (("lgbm", lgbm), ("lgbm_seed1", seed1), ("lgbm_dart", dart),
                   ("lgbm_dart_logloss", load("lgbm_dart_logloss")),
                   ("xgboost", xgb), ("xgboost_seed1", xseed1), ("xgboost_dart", xdart)):
    ckpt = {d: e for d, e in ckpt.items() if d in (xdart if name.startswith("xgboost") else dart)}
    p = pd.DataFrame([bp for e in ckpt.values() if e["best_params"] for bp in e["best_params"]])
    p = p.rename(columns={"rate_drop": "drop_rate"})
    line = (f"  {name:11s} {len(p)} folds: n_estimators > 450 in {np.mean(p.n_estimators > 450):.0%}, "
            f"median learning rate {p.learning_rate.median():.3f}")
    if "drop_rate" in p:
        weak = (p.drop_rate < 0.02) | (p.skip_drop > 0.45)
        line += (f", weak-dropout corner (drop_rate < 0.02 or skip_drop > 0.45) in {weak.mean():.0%}, "
                 f"median drop_rate {p.drop_rate.median():.3f}, median skip_drop {p.skip_drop.median():.2f}")
    print(line)

loss = (prauc(dart) - prauc(seed1)).dropna()
print("  lgbm_dart losses beyond 0.02 against lgbm_seed1, with lgbm_seed1's own move from lgbm: " + ", ".join(
    f"{d} {g:+.3f} ({prauc(seed1)[d] - prauc(lgbm)[d]:+.3f})" for d, g in loss[loss < -0.02].items()))

partime = load("lgbm_seed1", "partime")
xboth = [d for d in xdart if d in xseed1]
print(f"\nHours with parallel outer folds, pauses removed: lgbm_seed1 {hours(partime):.2f}, "
      f"lgbm_dart {hours(dart):.2f}; over xgboost_dart's {len(xboth)} datasets xgboost_seed1 "
      f"{seconds(xseed1)[xboth].sum() / 3600:.2f}, xgboost_dart {seconds(xdart, 'xgboost_dart')[xboth].sum() / 3600:.2f}")

# What DART costs per dataset, by the number of classes.
meta = pd.read_csv("results/datasets.csv").set_index("dataset")
ratio = pd.DataFrame({"LightGBM": seconds(dart) / seconds(partime),
                      "XGBoost": seconds(xdart, "xgboost_dart") / seconds(xseed1)[xboth]})
band = pd.cut(meta.n_classes, [0, 2, 4, 9, 200], labels=["2", "3-4", "5-9", "10+"])
print("\nDART time over its seed control's, median [quartiles] per dataset, by the number of classes:")
for b, g in ratio.groupby(band.loc[ratio.index], observed=True):
    print(f"  {b:4s} " + ", ".join(f"{m} {g[m].median():.1f}x [{g[m].quantile(.25):.1f}, {g[m].quantile(.75):.1f}] "
                                    f"(n {g[m].count()})" for m in ratio))
drop = pd.Series({d: np.mean([p["drop_rate"] for p in e["best_params"]]) for d, e in dart.items()})
print(f"  LightGBM DART ratio, Spearman with the chosen drop rate {spearmanr(drop, ratio.LightGBM[drop.index])[0]:.2f}, "
      f"with the rows {spearmanr(meta.n_rows[drop.index], ratio.LightGBM[drop.index])[0]:.2f}")
x = np.log(ratio.XGBoost.dropna())
trees = pd.Series({d: np.mean([p["n_estimators"] for p in e["best_params"]]) for d, e in xdart.items()})
logs = pd.DataFrame({"classes": np.log(meta.n_classes), "rows": np.log(meta.n_rows),
                     "features": np.log(meta.n_features), "DART's trees": np.log(trees)}).loc[x.index]
for cols in (["classes", "rows"], ["classes", "rows", "features", "DART's trees"]):
    design = np.column_stack([np.ones(len(x))] + [logs[c] for c in cols])
    coef = np.linalg.lstsq(design, x, rcond=None)[0]
    print(f"  XGBoost: log ratio = {coef[0]:.2f} " + " ".join(f"+ {b:.2f} log {c}" for b, c in zip(coef[1:], cols))
          + f", R2 {1 - (x - design @ coef).var() / x.var():.2f}")
print("  median rows by class band: " + ", ".join(
    f"{b} {v:.0f}" for b, v in meta.n_rows[x.index].groupby(band[x.index], observed=True).median().items()))
mf = "multiple-features"
print(f"  {mf}: xgboost_dart {seconds(xdart, 'xgboost_dart')[mf] / 3600:.1f} h, "
      f"xgboost_seed1 {seconds(xseed1)[mf] / 3600:.2f} h")
print("  slowest: " + ", ".join(f"{d} {r:.0f}x ({seconds(xdart, 'xgboost_dart')[d] / 3600:.1f} h)"
                                for d, r in np.exp(x).nlargest(5).items()))
# Follow-up, outside the family: DART selected on log loss against LightGBM selected on log loss.
ll, _ = compare({"lgbm_dart_logloss vs lgbm_logloss": (load("lgbm_dart_logloss"), load("lgbm_logloss")),
                 "lgbm_dart_logloss vs lgbm_dart": (load("lgbm_dart_logloss"), dart)}, calibration)
rng = np.random.default_rng(0)
for m in ("ece", "brier"):
    g = (per_dataset(load("lgbm_dart_logloss"), calibration).loc[cal, m]
         - per_dataset(load("lgbm_logloss"), calibration).loc[cal, m]).astype(float).values
    boot = g[rng.integers(0, len(g), (10_000, len(g)))].mean(1)
    print(f"  lgbm_dart_logloss - lgbm_logloss {m}: mean {g.mean():+.4f} "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}], median {np.median(g):+.4f}")
print("\nSelected on log loss (raw p):")
print(ll[["comparison", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "ece_base", "ece_variant", "ece_p",
          "brier_base", "brier_variant", "brier_p"]].round(4).to_string(index=False))
print(f"Saved {OUTPUT}")
