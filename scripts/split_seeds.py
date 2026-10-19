"""Outer-CV split noise: seven models on 30 datasets (10 per n_rows tercile, sampled with
seed 0 from the 108 complete cases) under outer split seeds 0 (published), 1 and 2.

Exploratory, not pre-registered: raw Wilcoxon p per split, no correction. The question is
which published orderings survive another split, and how large a gap split noise alone
can produce.

Usage: uv run python -m scripts.split_seeds
"""
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from benchmark.checkpoints import ckpt_path

OUTPUT = "results/split_seeds.csv"
MODELS = ["tabpfn35", "tabpfn3", "tabicl", "tabfm", "catboost", "lgbm", "hgb"]
RUNS = {0: "", 1: "split1", 2: "split2"}
DATASETS = """user-knowledge planning-relax leukemia-haslinger thyroid-newthyroid robot-failure-lp4 led7digit
hayes-roth autoUniv-au7-cpd1-500 heart-disease-processed-hungarian thoracic-surgery systhetic-control monks3
indian-liver-patient statlog-vehicle-silhouettes ozone-onehr hill-valley-without-noise car-evaluation
autoUniv-au6-1000 climate-simulation-craches statlog-image-segmentation volcanoes-d1 skin-segmentation ringnorm
volcanoes-c1 statlog-landsat-satellite phoneme mushroom-expanded artificial-characters nursery-4class
volcanoes-b2""".split()

pd.set_option("display.width", 250)
score = {seed: pd.DataFrame({m: {d: np.mean(joblib.load(ckpt_path(m, run))[d]["scores"]) for d in DATASETS}
                             for m in MODELS}) for seed, run in RUNS.items()}
(pd.concat(score, names=["split_seed", "dataset"]).reset_index()
   .melt(id_vars=["split_seed", "dataset"], var_name="model", value_name="prauc")
   .to_csv(OUTPUT, index=False))

means = pd.DataFrame({f"seed{s}": v.mean() for s, v in score.items()})
ranks = pd.DataFrame({f"rank{s}": v.mean().rank(ascending=False) for s, v in score.items()})
print(f"Mean PR AUC on {len(DATASETS)} datasets per split seed:")
print(pd.concat([means.round(4), (means.max(1) - means.min(1)).round(4).rename("range"), ranks], axis=1).to_string())

sd = np.stack([v.values for v in score.values()]).std(0, ddof=1)
print("\nSd of a model's PR AUC on one dataset over the three splits:")
print(pd.DataFrame({"median": np.median(sd, 0), "p90": np.percentile(sd, 90, 0), "max": sd.max(0)},
                   index=MODELS).round(4).to_string())

rows = []
for a, b in combinations(MODELS, 2):
    gaps = {s: v[a] - v[b] for s, v in score.items()}
    rows.append({"pair": f"{a} - {b}", **{f"gap{s}": g.mean() for s, g in gaps.items()},
                 **{f"p{s}": wilcoxon(g[g != 0]).pvalue for s, g in gaps.items()}})
pairs = pd.DataFrame(rows)
gap_cols, p_cols = [f"gap{s}" for s in RUNS], [f"p{s}" for s in RUNS]
pairs["sign_kept"] = (np.sign(pairs[gap_cols]).nunique(axis=1) == 1)
pairs["significant"] = (pairs[p_cols] < 0.05).sum(axis=1)
print("\nPairs: mean gap and raw Wilcoxon p per split seed")
print(pairs.round(4).to_string(index=False))
print(f"\n{len(pairs)} pairs: sign kept on all splits {pairs.sign_kept.sum()}, p < 0.05 on all three "
      f"{(pairs.significant == 3).sum()}, on one or two {pairs.significant.between(1, 2).sum()}")
for s1, s2 in combinations(RUNS, 2):
    print(f"Spearman of the model means, seed {s1} vs {s2}: "
          f"{spearmanr(score[s1].mean(), score[s2].mean()).statistic:.3f}")
print(f"Saved {OUTPUT}")
