"""DART under other outer splits: lgbm_dart against lgbm on the 30 datasets of scripts/split_seeds.py under
outer split seeds 0 (published), 1, 2, 3 and 4.

Exploratory, not pre-registered, except MARGIN, which was set before the runs of splits 1 to 4 were read.
The same 30 datasets recur in every split, so the splits are not independent samples of datasets and the raw
Wilcoxon p of each split is printed per split, never combined. The verdict is read from the per-dataset gap
averaged over the splits, with its bootstrap interval over datasets and the one-sided upper bound against MARGIN.
Inner folds, samplers and model seeds keep their seeds, as in scripts/split_seeds.py. lgbm_seed1 is lgbm with
another sampler seed: the pooled base is the mean of the two, so a gap is read against search noise.

Needs the lgbm_dart, lgbm and lgbm_seed1 checkpoints of splits 1 to 4:
    uv run python optuna_models.py --models lgbm lgbm_dart lgbm_seed1 --split-seed <s> --datasets <the 30 below>

Usage: uv run python -m scripts.dart_split_seeds
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.checkpoints import ckpt_path

OUTPUT = "results/dart_split_seeds.csv"
# As in scripts/split_seeds.py (copied: importing it would run its analysis).
RUNS = {0: "", 1: "split1", 2: "split2", 3: "split3", 4: "split4"}
MODELS = ["lgbm_dart", "lgbm", "lgbm_seed1"]
MARGIN = 0.002      # a mean gain of at least this size counts as excluded when the one-sided 95% bound is below it
FOLD_GAP = 0.10     # a fold this far below the lgbm fold
BOOTSTRAPS = 10_000
DATASETS = """user-knowledge planning-relax leukemia-haslinger thyroid-newthyroid robot-failure-lp4 led7digit
hayes-roth autoUniv-au7-cpd1-500 heart-disease-processed-hungarian thoracic-surgery systhetic-control monks3
indian-liver-patient statlog-vehicle-silhouettes ozone-onehr hill-valley-without-noise car-evaluation
autoUniv-au6-1000 climate-simulation-craches statlog-image-segmentation volcanoes-d1 skin-segmentation ringnorm
volcanoes-c1 statlog-landsat-satellite phoneme mushroom-expanded artificial-characters nursery-4class
volcanoes-b2""".split()
rng = np.random.default_rng(0)

rows, far_below = [], {}
for seed, run in RUNS.items():
    ck = {m: joblib.load(ckpt_path(m, run)) for m in MODELS}
    for d in DATASETS:
        rows.append({"split_seed": seed, "dataset": d, **{m: np.mean(ck[m][d]["scores"]) for m in MODELS}})
    far_below[seed] = sum(int((np.array(ck["lgbm_dart"][d]["scores"]) - np.array(ck["lgbm"][d]["scores"]) < -FOLD_GAP).sum())
                          for d in DATASETS)
df = pd.DataFrame(rows)
df["gap"] = df.lgbm_dart - df.lgbm
df["gap_pooled"] = df.lgbm_dart - (df.lgbm + df.lgbm_seed1) / 2
df.to_csv(OUTPUT, index=False)

print(f"lgbm_dart - lgbm, PR AUC, {len(DATASETS)} datasets per split seed (raw Wilcoxon p, not combined):")
for seed, part in df.groupby("split_seed"):
    g = part["gap"]
    print(f"  seed {seed}: mean {g.mean():+.4f}, median {g.median():+.4f}, ahead {(g > 0).sum()}, behind {(g < 0).sum()}, "
          f"raw p {wilcoxon(g[g != 0]).pvalue:.3f}, past 0.02: +{(g > 0.02).sum()} / -{(g < -0.02).sum()}; "
          f"against the pooled base mean {part.gap_pooled.mean():+.4f}; folds more than {FOLD_GAP} below lgbm: {far_below[seed]}")

per_split = df.groupby("split_seed").gap.mean()
within_se = df.groupby("split_seed").gap.std().div(np.sqrt(len(DATASETS))).median()
print(f"Mean gap per split: range {per_split.min():+.4f} to {per_split.max():+.4f}, sd over splits {per_split.std(ddof=1):.4f}; "
      f"median within-split se over datasets {within_se:.4f}")
sd = df.pivot(index="dataset", columns="split_seed", values="gap").std(axis=1, ddof=1)
print(f"Sd of a dataset's gap over the splits: median {sd.median():.4f}, max {sd.max():.4f}")

print(f"\nPer-dataset gap averaged over the {len(RUNS)} splits, bootstrap over datasets, margin {MARGIN}:")
for column, label in [("gap", "against lgbm"), ("gap_pooled", "against the mean of lgbm and lgbm_seed1")]:
    g = df.pivot(index="dataset", columns="split_seed", values=column).mean(axis=1)
    boot = rng.choice(g.to_numpy(), (BOOTSTRAPS, len(g))).mean(1)
    lo, hi = np.percentile(boot, [5, 95])
    print(f"  {label}: mean {g.mean():+.4f}, 90% CI [{lo:+.4f}, {hi:+.4f}], one-sided 95% upper bound {hi:+.4f}, "
          f"gain of {MARGIN} excluded: {hi < MARGIN}, ahead {(g > 0).sum()} behind {(g < 0).sum()}, "
          f"raw p {wilcoxon(g[g != 0]).pvalue:.3f}")
wins = df.assign(win=df.gap > 0).groupby("dataset").win.sum()
losses = df.assign(loss=df.gap < 0).groupby("dataset").loss.sum()
print(f"Datasets with DART ahead of lgbm on at least {len(RUNS) - 1} of {len(RUNS)} splits: {(wins >= len(RUNS) - 1).sum()}, "
      f"behind on at least {len(RUNS) - 1}: {(losses >= len(RUNS) - 1).sum()}")

base, dart = joblib.load(ckpt_path("lgbm")), joblib.load(ckpt_path("lgbm_dart"))
common = [d for d in dart if d in base]
full = np.array([np.mean(dart[d]["scores"]) - np.mean(base[d]["scores"]) for d in common])
print(f"Split 0, all {len(common)} datasets: mean gap {full.mean():+.4f}; these {len(DATASETS)}: {per_split[0]:+.4f}")
print(f"Saved {OUTPUT}")
