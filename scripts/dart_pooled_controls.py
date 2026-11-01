"""DART against pooled search-noise controls: intervals, a stated margin, subgroups and fold-level tails.

Reuses existing checkpoints, no search runs. Exploratory, not pre-registered. The notes read DART against single
seed controls by Holm p; this prints the mean gap with a bootstrap interval over datasets, the one-sided upper
bound against MARGIN, and the gap against the mean of the controls that share DART's 50-trial budget.

Usage: uv run python -m scripts.dart_pooled_controls
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.checkpoints import ckpt_path

OUTPUT = "results/dart_pooled_controls.csv"
MARGIN = 0.001            # a gain of at least this size over the controls counts as excluded when the bound is below it
FOLD_GAP = 0.10           # a fold this far below the base's fold score
BOOTSTRAPS = 10_000
# 50-trial controls only; lgbm_trials100 has twice DART's budget and is listed apart.
PAIRS = {"lgbm_dart": ["lgbm", "lgbm_seed1", "lgbm_rerun", "lgbm_trials100"],
         "xgboost_dart": ["xgboost", "xgboost_seed1"]}
POOLED = {"lgbm_dart": ["lgbm", "lgbm_seed1", "lgbm_rerun"], "xgboost_dart": ["xgboost", "xgboost_seed1"]}

meta = pd.read_csv("results/datasets.csv").set_index("dataset")
ckpt = {m: joblib.load(ckpt_path(m)) for m in {m for k, v in PAIRS.items() for m in [k, *v]}}
rng = np.random.default_rng(0)


def mean_score(model, d):
    return np.mean(ckpt[model][d]["scores"])


def summary(name, gaps):
    g = np.asarray(gaps)
    boot = rng.choice(g, (BOOTSTRAPS, len(g))).mean(1)
    nz = g[g != 0]
    return {"comparison": name, "n": len(g), "mean": g.mean(), "ci90_low": np.percentile(boot, 5),
            "ci90_high": np.percentile(boot, 95), "upper95_one_sided": np.percentile(boot, 95),
            "ahead": int((g > 0).sum()), "behind": int((g < 0).sum()), "raw_p": wilcoxon(nz).pvalue}


rows, per_dataset = [], {}
for dart, controls in PAIRS.items():
    datasets = [d for d in ckpt[dart] if all(d in ckpt[c] for c in controls)]
    for c in controls:
        rows.append(summary(f"{dart} - {c}", [mean_score(dart, d) - mean_score(c, d) for d in datasets]))
    pooled = [mean_score(dart, d) - np.mean([mean_score(c, d) for c in POOLED[dart]]) for d in datasets]
    rows.append(summary(f"{dart} - mean of {'+'.join(POOLED[dart])}", pooled))
    per_dataset[dart] = pd.Series(pooled, index=datasets)
out = pd.DataFrame(rows)
out["gain_of_margin_excluded"] = out.upper95_one_sided < MARGIN
print(f"DART minus controls, PR AUC; margin {MARGIN}: a gain at least that large is excluded when the one-sided "
      f"95% upper bound is below it")
print(out.round(4).to_string(index=False))

print("\nSubgroups of the pooled-control gap (exploratory, post hoc)")
sub = []
for dart, gaps in per_dataset.items():
    info = meta.loc[gaps.index]
    groups = {"binary": info.n_classes == 2, "multiclass": info.n_classes > 2,
              "rows tercile 1 (smallest)": pd.qcut(info.n_rows, 3, labels=False) == 0,
              "rows tercile 2": pd.qcut(info.n_rows, 3, labels=False) == 1,
              "rows tercile 3 (largest)": pd.qcut(info.n_rows, 3, labels=False) == 2}
    base_prauc = pd.Series({d: mean_score(POOLED[dart][0], d) for d in gaps.index})
    terc = pd.qcut(base_prauc, 3, labels=False)
    groups |= {"base PR AUC tercile 1 (hardest)": terc == 0, "base PR AUC tercile 3 (easiest)": terc == 2}
    for name, mask in groups.items():
        g = gaps[mask.to_numpy()]
        sub.append({"model": dart, "group": name, "n": len(g), "mean_gap": g.mean(), "median_gap": g.median(),
                    "ahead": int((g > 0).sum()), "behind": int((g < 0).sum())})
sub = pd.DataFrame(sub)
print(sub.round(4).to_string(index=False))

print(f"\nFolds more than {FOLD_GAP} below or above the base's fold score, on DART's datasets; "
      f"a base against another run of itself shows how many such folds noise alone gives")
tails = []
for model, base, dart in [("lgbm_dart", "lgbm", "lgbm_dart"), ("lgbm_seed1", "lgbm", "lgbm_dart"),
                          ("lgbm", "lgbm_seed1", "lgbm_dart"), ("lgbm_rerun", "lgbm", "lgbm_dart"),
                          ("xgboost_dart", "xgboost", "xgboost_dart"), ("xgboost_seed1", "xgboost", "xgboost_dart")]:
    gaps = {d: np.array(ckpt[model][d]["scores"]) - np.array(ckpt[base][d]["scores"]) for d in ckpt[dart]}
    flat = np.concatenate(list(gaps.values()))
    below = {d: int((v < -FOLD_GAP).sum()) for d, v in gaps.items() if (v < -FOLD_GAP).any()}
    tails.append({"model": model, "base": base, "folds": len(flat), "below": int((flat < -FOLD_GAP).sum()),
                  "above": int((flat > FOLD_GAP).sum()), "worst_fold_gap": flat.min(), "datasets_below": below})
print(pd.DataFrame(tails).to_string(index=False))

print("\nLearning rate chosen per fold on DART's datasets, search range up to 0.3")
lr = []
for base, dart in [("lgbm", "lgbm_dart"), ("xgboost", "xgboost_dart")]:
    for m in [base, dart]:
        a = np.array([p["learning_rate"] for d in ckpt[dart] for p in ckpt[m][d]["best_params"]])
        lr.append({"model": m, "folds": len(a), "median": np.median(a), "share_above_0.2": (a > 0.2).mean(),
                   "share_above_0.25": (a > 0.25).mean()})
print(pd.DataFrame(lr).round(3).to_string(index=False))

# The queue of the DART review: DART's own sampler seed, and the learning-rate cap of 1.
extra = {m: joblib.load(ckpt_path(m)) for m in ["lgbm_dart_seed1", "lgbm_wide", "lgbm_dart_wide"]}
ckpt |= extra
seed_rows = []
datasets = [d for d in ckpt["lgbm_dart"] if d in ckpt["lgbm_dart_seed1"] and all(d in ckpt[c] for c in POOLED["lgbm_dart"])]
seed_rows.append(summary("lgbm_dart_seed1 - lgbm_dart (DART's own sampler-seed noise)",
                         [mean_score("lgbm_dart_seed1", d) - mean_score("lgbm_dart", d) for d in datasets]))
seed_rows.append(summary("lgbm_seed1 - lgbm (the base's sampler-seed noise)",
                         [mean_score("lgbm_seed1", d) - mean_score("lgbm", d) for d in datasets]))
seed_rows.append(summary("mean of the two lgbm_dart seeds - mean of lgbm+lgbm_seed1+lgbm_rerun",
                         [np.mean([mean_score(m, d) for m in ("lgbm_dart", "lgbm_dart_seed1")])
                          - np.mean([mean_score(c, d) for c in POOLED["lgbm_dart"]]) for d in datasets]))
wide = [d for d in ckpt["lgbm_wide"] if d in ckpt["lgbm_dart_wide"] and d in ckpt["lgbm_dart"]]
seed_rows.append(summary(f"lgbm_dart_wide - lgbm_wide, cap 1.0 ({len(wide)} datasets)",
                         [mean_score("lgbm_dart_wide", d) - mean_score("lgbm_wide", d) for d in wide]))
seed_rows.append(summary(f"lgbm_dart - lgbm, cap 0.3, the same {len(wide)} datasets",
                         [mean_score("lgbm_dart", d) - mean_score("lgbm", d) for d in wide]))
seed_rows.append(summary("lgbm_wide - lgbm (what the cap does to the base)",
                         [mean_score("lgbm_wide", d) - mean_score("lgbm", d) for d in wide]))
seed_rows.append(summary("lgbm_dart_wide - lgbm_dart (what the cap does to DART)",
                         [mean_score("lgbm_dart_wide", d) - mean_score("lgbm_dart", d) for d in wide]))
extra_out = pd.DataFrame(seed_rows)
extra_out["gain_of_margin_excluded"] = extra_out.upper95_one_sided < MARGIN
print("\nDART's own seed noise, and the learning-rate cap")
print(extra_out.round(4).to_string(index=False))
for m in ["lgbm_wide", "lgbm_dart_wide"]:
    a = np.array([p["learning_rate"] for d in wide for p in ckpt[m][d]["best_params"]])
    print(f"{m}: median learning rate {np.median(a):.3f}, share above 0.3 {(a > 0.3).mean():.1%}, above 0.6 {(a > 0.6).mean():.1%}")

pd.concat([out.assign(table="controls"), sub.assign(table="subgroups"), extra_out.assign(table="seed_and_cap")]).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}")
