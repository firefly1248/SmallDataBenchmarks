"""The ConfOpt block: LightGBM's search with ConfOpt (OptunaHub wrapper) in place of TPE,
same space and budget, two sampler seeds per method, all on the same library versions.

Not pre-registered: the design grew over the experiment (100 trials, then the second seeds
and the TPE re-run on current versions). The main pair is one Holm family of two, ConfOpt
against TPE per seed; the seed-averaged gap and the budget pairs are reported with a paired
bootstrap CI over datasets and a raw Wilcoxon p. Calibration is over the 108 calibration
datasets, raw p, with Holm over the four per-seed tests.

lgbm_rerun is lgbm re-run on the current library versions with parallel outer folds; lgbm
itself ran in August. Hours compare only runs with parallel outer folds; lgbm_seed1 re-timed
that way is 3.95 h (scripts/dart_controls.py).

The log-loss runs without class_weight "balanced" were chosen after seeing the PR AUC
results; they are reported apart, raw p, with Holm over their six per-seed tests for reference,
as the four 100-trial tests have.

lgbm_logloss_unweighted_seed1 was paused by hand on mfeat-fourier; the window, known to a few
minutes from logs/block_x.log, is subtracted from its time.

Usage: uv run python -m scripts.confopt_controls
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, compare, holm, per_dataset

OUTPUT = "results/confopt_controls.csv"
# The manual pause of lgbm_logloss_unweighted_seed1, 2026-10-30 about 10:05 to 16:29 local time, epoch seconds.
PAUSE = (1793347500, 1793370573)
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
rng = np.random.default_rng(0)


def load(model):
    return joblib.load(ckpt_path(model))


def prauc(ckpt):
    return per_dataset(ckpt, set())["prauc"].astype(float)


def gap(label, diff):
    """Mean, median, 10 % trimmed mean, a paired bootstrap 95 % CI and the Wilcoxon p of a
    per-dataset gap."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:34s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")
    return {"comparison": label, "mean": diff.mean(), "ci_lo": lo, "ci_hi": hi, "median": diff.median(),
            "trim10": trim_mean(diff, 0.1), "p_raw": p}


def picks(ckpt, param):
    """The chosen value of ``param`` per fold over the calibration datasets."""
    return [p[param] for d, e in sorted(ckpt.items()) if d in calibration for p in e["best_params"]]


def class_weights(ckpt, dataset):
    return [str(p["class_weight"]) for p in ckpt[dataset]["best_params"]]


pd.set_option("display.width", 250)
c0, c1 = load("lgbm_confopt"), load("lgbm_confopt_seed1")
t0, t1 = load("lgbm_rerun"), load("lgbm_seed1")

out, _ = compare({"lgbm_confopt vs lgbm_rerun": (c0, t0), "lgbm_confopt_seed1 vs lgbm_seed1": (c1, t1)}, calibration)
print(out[["comparison", "datasets", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "p_holm"]]
      .round(4).to_string(index=False))
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))

print("\nPer-dataset PR AUC gaps (paired bootstrap 95 % CI):")
rows = [gap("ConfOpt - TPE, seed 0", prauc(c0) - prauc(t0)),
        gap("ConfOpt - TPE, seed 1", prauc(c1) - prauc(t1)),
        gap("ConfOpt - TPE, mean of seeds", (prauc(c0) + prauc(c1)) / 2 - (prauc(t0) + prauc(t1)) / 2),
        gap("ConfOpt seed 1 - seed 0", prauc(c1) - prauc(c0)),
        gap("TPE seed 1 - seed 0", prauc(t1) - prauc(t0)),
        gap("ConfOpt 100 - 50 trials", prauc(load("lgbm_confopt_trials100")) - prauc(c0)),
        gap("TPE 100 - 50 trials", prauc(load("lgbm_trials100")) - prauc(t0)),
        gap("TPE re-run - August", prauc(t0) - prauc(load("lgbm"))),
        gap("ConfOpt 100 - 50 trials, seed 1", prauc(load("lgbm_confopt_trials100_seed1")) - prauc(c1)),
        gap("TPE 100 - 50 trials, seed 1", prauc(load("lgbm_trials100_seed1")) - prauc(t1))]
c100_1 = load("lgbm_confopt_trials100_seed1")
budget = {"ConfOpt": (("lgbm_confopt", "lgbm_confopt_seed1"),
                     ("lgbm_confopt_trials100", "lgbm_confopt_trials100_seed1")),
          "TPE": (("lgbm_rerun", "lgbm_seed1"), ("lgbm_trials100", "lgbm_trials100_seed1"))}
for search, (fifty, hundred) in budget.items():
    rows.append(gap(f"{search} 100 - 50 trials, mean of seeds",
                    sum(prauc(load(m)) for m in hundred) / 2 - sum(prauc(load(m)) for m in fifty) / 2))
rows.append(gap("ConfOpt 100 - 50, seed 1, no pendigits", (prauc(c100_1) - prauc(c1)).drop("pendigits")))
print(f"  pendigits fold scores, ConfOpt seed 1: 50 trials {np.round(c1['pendigits']['scores'], 3).tolist()}, "
      f"100 trials {np.round(c100_1['pendigits']['scores'], 3).tolist()}; collapsed pick "
      f"{ {k: round(v, 3) if isinstance(v, float) else v for k, v in c100_1['pendigits']['best_params'][2].items()} }")
budget_tests = {r["comparison"]: r["p_raw"] for r in rows
                if "100 - 50 trials" in r["comparison"] and "mean" not in r["comparison"]}
print("  Holm over the four 100-trial tests: " + ", ".join(f"{k} {v:.2g}" for k, v in holm(budget_tests).items()))
largest = (prauc(c0) - prauc(t0)).abs().nlargest(5).index
print("  the five largest seed-0 gaps:")
print(pd.DataFrame({"ConfOpt 0": prauc(c0), "ConfOpt 1": prauc(c1), "TPE 0": prauc(t0), "TPE 1": prauc(t1)})
      .loc[largest].round(3).to_string())
august = load("lgbm")
pairs = [(a, b) for d in t0 for a, b in zip(t0[d]["best_params"], august[d]["best_params"])]
close = sum(a.keys() == b.keys() and all(np.isclose(a[k], b[k], rtol=1e-9, atol=0) if isinstance(a[k], float)
                                         else a[k] == b[k] for k in a) for a, b in pairs)
print(f"  TPE re-run picks August's configuration in {sum(a == b for a, b in pairs)} of {len(pairs)} folds, "
      f"{close} to float precision")

cal = sorted(calibration)
pd_ = {k: per_dataset(v, calibration).loc[cal] for k, v in {"c0": c0, "c1": c1, "t0": t0, "t1": t1}.items()}
print("\nCalibration, seed-averaged ConfOpt - TPE and per-seed Holm over the four tests:")
for m in ("ece", "brier"):
    d = ((pd_["c0"][m] + pd_["c1"][m]) / 2 - (pd_["t0"][m] + pd_["t1"][m]) / 2).astype(float)
    print(f"  {m}: {d.mean():+.4f}, worse on {(d > 0).sum()} of {len(d)}, p {wilcoxon(d).pvalue:.1e}")
per_seed = {f"{m} seed {s}": wilcoxon((pd_[f"c{s}"][m] - pd_[f"t{s}"][m]).astype(float)).pvalue
            for m in ("ece", "brier") for s in "01"}
print("  " + ", ".join(f"{k} raw {p:.3g} Holm {holm(per_seed)[k]:.3g}" for k, p in per_seed.items()))
for a, b in (("c1", "c0"), ("t1", "t0")):
    print(f"  ECE {a} against {b}: p {wilcoxon((pd_[a].ece - pd_[b].ece).astype(float)).pvalue:.2g}")

print("\nclass_weight and learning rate over the calibration folds:")
for k, e in {"ConfOpt 0": c0, "ConfOpt 1": c1, "TPE 0": t0, "TPE 1": t1}.items():
    print(f"  {k}: 'balanced' {np.mean([str(w) == 'balanced' for w in picks(e, 'class_weight')]):.2f}, "
          f"learning rate >= 0.1 {np.mean(np.array(picks(e, 'learning_rate')) >= 0.1):.2f}")
for s, (c, t) in {"0": (c0, t0), "1": (c1, t1)}.items():
    gap_ = (pd_[f"c{s}"].ece - pd_[f"t{s}"].ece).astype(float)
    extra = pd.Series({d: class_weights(c, d).count("balanced") - class_weights(t, d).count("balanced") for d in cal})
    never = [d for d in cal if "balanced" not in class_weights(c, d) + class_weights(t, d)]
    log_lr = lambda e, d: np.mean([np.log(p["learning_rate"]) for p in e[d]["best_params"]])
    lr_gap = pd.Series({d: log_lr(c, d) - log_lr(t, d) for d in cal})
    print(f"  seed {s}: Spearman of the ECE gap with the extra 'balanced' folds {spearmanr(extra, gap_)[0]:.2f}; "
          f"never 'balanced' on {len(never)} datasets, ECE gap {gap_[never].mean():+.4f}, worse on "
          f"{(gap_[never] > 0).sum()}, p {wilcoxon(gap_[never]).pvalue:.2g}; Spearman with the log learning-rate gap "
          f"{spearmanr(lr_gap, gap_)[0]:.2f}")
print(f"  lgbm_logloss 'balanced' {np.mean([str(w) == 'balanced' for w in picks(load('lgbm_logloss'), 'class_weight')]):.2f}")

inner = pd.read_csv("results/search_budget_inner.csv")
print("\n100 against 50 trials where the pick changed, mean gain per dataset (scripts/search_budget_inner.py):")
for search, g in inner.groupby("search", sort=False):
    d = g.groupby("dataset")[["d_search", "d_fresh", "d_outer"]].mean()
    print(f"  {search:8s} {len(g)} folds, {len(d)} datasets: search split {d.d_search.mean():+.4f}, "
          f"fresh split {d.d_fresh.mean():+.4f}, outer fold {d.d_outer.mean():+.4f}; lost on the fresh split "
          f"{1 - d.d_fresh.mean() / d.d_search.mean():.1%}, reaching the outer fold {d.d_outer.mean() / d.d_search.mean():.1%}; "
          f"outer up on {(d.d_outer > 0).sum()} / down on {(d.d_outer < 0).sum()} datasets, "
          f"p {wilcoxon(d.d_outer[d.d_outer != 0]).pvalue:.2g}; per fold Spearman with the search split "
          f"{spearmanr(g.d_search, g.d_outer)[0]:.2f}, "
          f"with the fresh split {spearmanr(g.d_fresh, g.d_outer)[0]:.2f}")

def seconds(ckpt):
    """Wall seconds per dataset with the manual pause removed."""
    return pd.Series({d: e["time"] - max(0, min(e["start"] + e["time"], PAUSE[1]) - max(e["start"], PAUSE[0]))
                      for d, e in ckpt.items()})


hours = {m: seconds(load(m)).sum() / 3600
         for m in ("lgbm_rerun", "lgbm_confopt", "lgbm_confopt_seed1", "lgbm_trials100", "lgbm_confopt_trials100",
                   "lgbm_logloss_unweighted", "lgbm_confopt_logloss_unweighted", "lgbm_confopt_trials100_seed1",
                   "lgbm_trials100_seed1", "lgbm_logloss_unweighted_seed1", "lgbm_confopt_logloss_unweighted_seed1")}
print("\nHours with parallel outer folds: " + ", ".join(f"{m} {h:.2f}" for m, h in hours.items()))
secs = pd.DataFrame({"ConfOpt": per_dataset(c0, set())["time"], "TPE": per_dataset(t0, set())["time"]})
print(f"  seconds per dataset, median: ConfOpt {secs.ConfOpt.median():.0f}, TPE {secs.TPE.median():.0f}, "
      f"overhead {(secs.ConfOpt - secs.TPE).median():.0f}")

# Exploratory, outside the family: both searches selected on log loss without "balanced".
# lgbm_logloss against lgbm_logloss_unweighted: what allowing "balanced" does to TPE on log loss.
ll, _ = compare({"lgbm_confopt_logloss_unweighted vs lgbm_logloss_unweighted":
                 (load("lgbm_confopt_logloss_unweighted"), load("lgbm_logloss_unweighted")),
                 "lgbm_logloss vs lgbm_logloss_unweighted": (load("lgbm_logloss"), load("lgbm_logloss_unweighted")),
                 "seed 1: lgbm_confopt_logloss_unweighted_seed1 vs lgbm_logloss_unweighted_seed1":
                 (load("lgbm_confopt_logloss_unweighted_seed1"), load("lgbm_logloss_unweighted_seed1"))},
                calibration)
print("\nLog loss (raw p):")
print(ll[["comparison", "prauc_base", "prauc_variant", "ahead", "behind", "p_raw", "ece_base", "ece_variant", "ece_p",
          "brier_base", "brier_variant", "brier_p"]].round(4).to_string(index=False))
ll_pairs = {"0": ("lgbm_confopt_logloss_unweighted", "lgbm_logloss_unweighted"),
            "1": ("lgbm_confopt_logloss_unweighted_seed1", "lgbm_logloss_unweighted_seed1")}
for s_, (c, t) in ll_pairs.items():
    rows.append(gap(f"log loss ConfOpt - TPE, seed {s_}", prauc(load(c)) - prauc(load(t))))
rows.append(gap("log loss ConfOpt - TPE, mean of seeds",
                sum(prauc(load(c)) - prauc(load(t)) for c, t in ll_pairs.values()) / 2))
rows.append(gap("log loss ConfOpt seed 1 - seed 0", prauc(load(ll_pairs["1"][0])) - prauc(load(ll_pairs["0"][0]))))
rows.append(gap("log loss TPE seed 1 - seed 0", prauc(load(ll_pairs["1"][1])) - prauc(load(ll_pairs["0"][1]))))
seeded = ll[ll.comparison.str.contains("unweighted vs|unweighted_seed1 vs")].reset_index(drop=True)
tests = {f"{m} seed {i}": seeded.loc[i, col] for i in (0, 1) for m, col in
         (("PR AUC", "p_raw"), ("ECE", "ece_p"), ("Brier", "brier_p"))}
print("  Holm over the six log-loss tests: " + ", ".join(f"{k} {v:.2g}" for k, v in holm(tests).items()))

pd.concat([out.drop(columns=["hours_base", "hours_variant"]), pd.DataFrame(rows), ll.drop(columns="p_holm")]
          ).round(6).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}")
