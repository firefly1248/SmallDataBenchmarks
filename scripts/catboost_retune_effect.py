"""The prior-start re-run against the uniform-start run: what the tuning chose, what it cost.

Compares, on datasets finished in both runs, the hyperparameters, the time and the
calibration of CatBoost re-tuned from the prior against the original uniform-start run.
The fixed-parameter arm of ``catboost_start_effect.csv`` sits in between: same
hyperparameters as the original, prior start, so it separates the start from the re-tuning.

Only dataset totals are timed, so time per fold is total / 4 and per trial total / 200
(4 outer folds x 50 trials; the four final refits are included, a small share).

Usage: uv run python -m scripts.catboost_retune_effect
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.calibration import brier_score, confidence_gap, expected_calibration_error
from benchmark.checkpoints import ckpt_path
from config import N_OUTER_FOLDS, N_TRIALS

OLD = "results/ckpt/catboost.joblib.bak-uniform-start"
SMALL_SHRINK, BALANCED = 5, 0.01
PARAMS = ["depth", "learning_rate", "n_estimators", "lr x trees"]

old, new = joblib.load(OLD), joblib.load(ckpt_path("catboost"))
both = sorted(d for d in new if d in old and np.isfinite(new[d]["time"]))
effect = pd.read_csv("results/catboost_start_effect.csv").set_index("dataset")

# Item 1: the chosen hyperparameters, per outer fold (the folds are the same split in both runs).
csv = pd.read_csv("results/best_params.csv")
csv = csv[csv.model == "catboost_uniform_start"].pivot_table(index=["dataset", "fold"], columns="param",
                                                             values="value", aggfunc="first")
rows = []
for d in both:
    n_classes = np.vstack(new[d]["preds"]).shape[1]
    for fold, (p_old, p_new) in enumerate(zip(old[d]["best_params"], new[d]["best_params"])):
        assert np.isclose(float(csv.loc[(d, fold), "depth"]), p_old["depth"])  # the csv is the old checkpoint
        for run, p in (("uniform", p_old), ("prior", p_new)):
            rows.append(dict(dataset=d, fold=fold, run=run, task="binary" if n_classes == 2 else "multiclass",
                             depth=p["depth"], learning_rate=p["learning_rate"], n_estimators=p["n_estimators"],
                             **{"lr x trees": p["learning_rate"] * p["n_estimators"]}))
folds = pd.DataFrame(rows)
wide = folds.pivot_table(index=["dataset", "fold", "task"], columns="run", values=PARAMS)

n_ds = folds.groupby("task").dataset.nunique()
print(f"Datasets finished in both runs: {len(both)} of 131 "
      f"({n_ds.get('binary', 0)} binary, {n_ds.get('multiclass', 0)} multiclass)\n")
print("1. Chosen hyperparameters, median over folds")
print(folds.groupby(["task", "run"])[PARAMS].median().unstack("run").round(3).to_string())
print("\nShare of folds where the re-run chose more / the same / less")
share = {}
for task, g in wide.groupby(level="task"):
    for param in PARAMS:
        diff = g[(param, "prior")] - g[(param, "uniform")]
        share[(task, param)] = {"more": (diff > 1e-12).mean(), "same": (diff.abs() <= 1e-12).mean(),
                                "less": (diff < -1e-12).mean()}
print(pd.DataFrame(share).T.round(2).to_string())
print("\nDepth, share of folds")
print(pd.crosstab([folds.task, folds.run], folds.depth, normalize="index").round(2).to_string())

# Item 2: time, and how much of its change the chosen depth and tree count explain.
per_ds = folds.groupby(["dataset", "task", "run"])[["depth", "n_estimators"]].mean().unstack("run")
time = pd.DataFrame({run: [ckpt[d]["time"] for d in per_ds.index.get_level_values("dataset")]
                     for run, ckpt in (("uniform", old), ("prior", new))}, index=per_ds.index)
print("\n2. Time, median over datasets (seconds)")
summary = {}
for task, t in time.groupby(level="task"):
    summary[task] = {"per dataset, uniform": t.uniform.median(), "per dataset, prior": t.prior.median(),
                     "per fold, uniform": t.uniform.median() / N_OUTER_FOLDS,
                     "per fold, prior": t.prior.median() / N_OUTER_FOLDS,
                     "per trial, uniform": t.uniform.median() / (N_OUTER_FOLDS * N_TRIALS),
                     "per trial, prior": t.prior.median() / (N_OUTER_FOLDS * N_TRIALS),
                     "median ratio prior / uniform": (t.prior / t.uniform).median(),
                     "total ratio prior / uniform": t.prior.sum() / t.uniform.sum()}
print(pd.DataFrame(summary).round(2).to_string())

# log(time ratio) ~ change in mean depth + log change in mean trees, over datasets. The best
# parameters stand in for where the 200 trials concentrated, so this is a proxy fit.
log_ratio = np.log(time.prior / time.uniform)
X = np.column_stack([np.ones(len(log_ratio)),
                     per_ds[("depth", "prior")] - per_ds[("depth", "uniform")],
                     np.log(per_ds[("n_estimators", "prior")] / per_ds[("n_estimators", "uniform")])])
coef, *_ = np.linalg.lstsq(X, log_ratio.to_numpy(), rcond=None)
fitted = X @ coef
r2 = 1 - np.sum((log_ratio - fitted) ** 2) / np.sum((log_ratio - log_ratio.mean()) ** 2)
mean_x = X.mean(axis=0)
print(f"\nlog(time ratio) = {coef[0]:+.3f} + {coef[1]:+.3f} * d(depth) + {coef[2]:+.3f} * dlog(trees), "
      f"R^2 = {r2:.2f}")
print(f"At the mean change (depth {mean_x[1]:+.2f}, trees x{np.exp(mean_x[2]):.2f}): depth x{np.exp(coef[1] * mean_x[1]):.2f}, "
      f"trees x{np.exp(coef[2] * mean_x[2]):.2f}, rest x{np.exp(coef[0]):.2f}; observed x{np.exp(log_ratio.mean()):.2f}")

# Item 4: learning_rate * n_estimators below 5, median over folds, on the calibration population.
shrink = folds.groupby(["dataset", "run"])["lr x trees"].median().unstack("run")
shrink = shrink.loc[shrink.index.intersection(effect.index)]
shrink["group"] = np.where(effect.loc[shrink.index, "kl_prior_uniform"] < BALANCED, "balanced", "imbalanced")
print(f"\n4. learning_rate * n_estimators below {SMALL_SHRINK}, median over folds "
      f"({len(shrink)} of the 108 calibration datasets finished in both runs)")
small = shrink.groupby("group").agg(datasets=("uniform", "size"),
                                    uniform=("uniform", lambda s: (s < SMALL_SHRINK).sum()),
                                    prior=("prior", lambda s: (s < SMALL_SHRINK).sum()))
small.loc["all"] = small.sum()
print(small.to_string())

# Item 5: calibration, re-tuned from the prior against the original and the fixed-parameter arm.
cal = []
for d in shrink.index:
    row = {"dataset": d, "group": shrink.loc[d, "group"]}
    for run, ckpt in (("uniform", old), ("prior retuned", new)):
        prob, y = np.vstack(ckpt[d]["preds"]), np.concatenate(ckpt[d]["labels"])
        row |= {f"Brier {run}": brier_score(y, prob), f"ECE {run}": expected_calibration_error(y, prob),
                f"gap {run}": confidence_gap(y, prob), f"PR AUC {run}": np.mean(ckpt[d]["scores"])}
    e = effect.loc[d]
    row |= {"Brier prior fixed": e.brier_prior, "ECE prior fixed": e.ece_prior, "gap prior fixed": e.gap_prior,
            "PR AUC prior fixed": e.prauc_prior}
    assert np.isclose(row["Brier uniform"], e.brier_uniform, atol=1e-4)  # both read the same uniform-start run
    cal.append(row)
cal = pd.DataFrame(cal).set_index("dataset")
print(f"\n5. Calibration on the same {len(cal)} datasets: mean (median), and Wilcoxon p of re-tuned vs uniform")
out = {}
for group, g in [("all", cal), *cal.groupby("group")]:
    for metric in ("Brier", "ECE", "gap", "PR AUC"):
        cols = [f"{metric} uniform", f"{metric} prior fixed", f"{metric} prior retuned"]
        stat = g[cols].abs() if metric == "gap" else g[cols]
        label = "|gap|" if metric == "gap" else metric
        out[(group, label)] = {c.removeprefix(f"{metric} "): f"{stat[c].mean():.4f} ({stat[c].median():.4f})"
                               for c in cols}
        out[(group, label)]["p retuned vs uniform"] = f"{wilcoxon(stat[cols[2]] - stat[cols[0]]).pvalue:.2g}"
    out[(group, "datasets")] = {"uniform": len(g)}
print(pd.DataFrame(out).T.fillna("").to_string())

folds.to_csv("results/catboost_retune_params.csv", index=False)
cal.round(5).to_csv("results/catboost_retune_calibration.csv")
print("\nSaved results/catboost_retune_params.csv, results/catboost_retune_calibration.csv")
