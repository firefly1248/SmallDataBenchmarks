"""Numbers of the five-seed ConfOpt section that no other script prints, from the checkpoints,
results/search_inner_all.csv and results/confopt_startup.csv. Not pre-registered: chosen after
the results and after a review of the section.

A. Holm over the five per-seed outer PR AUC tests; the three seeds added after the two-seed
   version alone; a t interval over the five per-seed gaps; all 25 seed pairings.
B. The five largest five-seed gains; the mean without them, and without the five largest gains
   and losses; terciles of rows with and without the five.
C. Inner re-score: fresh - search and outer - search of the ConfOpt - TPE gap; per dataset the
   Spearman of the search gap with the outer gap.
D. Opening configurations: the best of 10 averaged over the five seeds; the final gap where
   ConfOpt opened ahead and where it did not.
E. Calibration, all five seeds: "balanced" shares, Spearman of the ECE gap with the optimism gap,
   the extra "balanced" folds and log(lr x trees); the ECE gap where neither picks "balanced".
F. Selection metric: (ConfOpt - TPE under log loss) - (ConfOpt - TPE under PR AUC), seeds 0, 1.
G. Hours per seed and the per-dataset median ratio of seconds.
H. 100 against 50 trials on the search, fresh and outer scores, seeds 0 and 1.

Usage: uv run python -m scripts.confopt_writeup
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student, trim_mean, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.results import CALIBRATION_POPULATION, holm, per_dataset

CONFOPT = ["lgbm_confopt"] + [f"lgbm_confopt_seed{s}" for s in range(1, 5)]
TPE = ["lgbm_rerun"] + [f"lgbm_seed{s}" for s in range(1, 5)]
SEEDS = range(5)
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
cal = sorted(calibration)
rows = pd.read_csv("results/datasets.csv").set_index("dataset")["n_rows"]
inner = pd.read_csv("results/search_inner_all.csv").query("trials == 50") \
    .groupby(["sampler", "seed", "dataset"])[["search", "fresh", "outer"]].mean()
rng = np.random.default_rng(0)
ckpts = {m: joblib.load(ckpt_path(m)) for m in CONFOPT + TPE}
tab = {m: per_dataset(e, calibration).astype(float) for m, e in ckpts.items()}


def gap(label, diff):
    """As in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:44s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")


def five(metric, models, index=None):
    return sum(tab[m][metric] for m in models).loc[index if index is not None else slice(None)] / len(models)


print("A. Outer PR AUC, ConfOpt - TPE")
per_seed = {s: tab[c].prauc - tab[t].prauc for s, (c, t) in enumerate(zip(CONFOPT, TPE))}
raw = {f"seed {s}": wilcoxon(d[d != 0]).pvalue for s, d in per_seed.items()}
print("  Holm over the five seeds: " + ", ".join(f"{k} raw {raw[k]:.4f} Holm {v:.3f}" for k, v in holm(raw).items()))
gap("seeds 2 to 4 (added later)", sum(per_seed[s] for s in (2, 3, 4)) / 3)
means = np.array([per_seed[s].mean() for s in SEEDS])
half = student.ppf(0.975, 4) * means.std(ddof=1) / np.sqrt(5)
print(f"  t interval over the five per-seed means: {means.mean():+.4f} [{means.mean() - half:+.4f}, {means.mean() + half:+.4f}]")
for metric, idx in (("prauc", None), ("ece", cal)):
    pairs = pd.Series({(i, j): (tab[c][metric] - tab[t][metric]).loc[idx if idx else slice(None)].mean()
                       for i, c in enumerate(CONFOPT) for j, t in enumerate(TPE)})
    print(f"  {metric}: all 25 pairings, positive {(pairs > 0).sum()}, range {pairs.min():+.4f} to {pairs.max():+.4f}")
    print(f"    per-seed means ConfOpt {[round(float(tab[m][metric].loc[idx if idx else slice(None)].mean()), 5) for m in CONFOPT]}, "
          f"TPE {[round(float(tab[m][metric].loc[idx if idx else slice(None)].mean()), 5) for m in TPE]}")

print("\nB. Where the five-seed gain sits")
d = five("prauc", CONFOPT) - five("prauc", TPE)
top = d.nlargest(5)
print("  five largest: " + ", ".join(f"{k} ({rows[k]} rows) {v:+.4f}" for k, v in top.items()))
print(f"  their share of the summed gain {top.sum() / d.sum():.0%}")
gap("without the five largest gains", d.drop(top.index))
gap("without the five largest gains and losses", d.drop(top.index).drop(d.nsmallest(5).index))
band = pd.qcut(rows[d.index], 3, labels=["small", "medium", "large"])
for b, g in d.groupby(band, observed=True):
    rest = g.drop(top.index, errors="ignore")
    print(f"  {b} ({rows[g.index].min()}-{rows[g.index].max()} rows): {g.mean():+.4f}, p {wilcoxon(g[g != 0]).pvalue:.2g}; "
          f"without the five {rest.mean():+.4f}")
print(f"  Spearman with rows {spearmanr(d, rows[d.index])[0]:+.2f} (p {spearmanr(d, rows[d.index])[1]:.2g})")

print("\nC. Inner re-score, ConfOpt - TPE averaged over the five seeds")
g = {m: sum(inner.loc[("confopt", s), m] - inner.loc[("tpe", s), m] for s in SEEDS) / 5 for m in ("search", "fresh", "outer")}
gap("fresh - search", g["fresh"] - g["search"])
gap("outer - search", g["outer"] - g["search"])
print(f"  Spearman of the search gap with the outer gap {spearmanr(g['search'], g['outer'])[0]:.2f} "
      f"(p {spearmanr(g['search'], g['outer'])[1]:.1g}); without the five "
      f"{spearmanr(g['search'].drop(top.index), g['outer'].drop(top.index))[0]:.2f}")
for m in ("search", "fresh", "outer"):
    gap(f"{m}, without the five largest gains", g[m].drop(top.index))

print("\nD. Opening configurations (best of 10 per fold, mean over folds)")
start = pd.read_csv("results/confopt_startup.csv").groupby(["sampler", "seed", "dataset", "fold"]).search.max() \
    .groupby(["sampler", "seed", "dataset"]).mean()
opening = {s: start.loc[("confopt", s)] - start.loc[("tpe", s)] for s in SEEDS}
gap("best of 10, mean of the five seeds", sum(opening.values()) / 5)
for s in SEEDS:
    fin = {m: inner.loc[("confopt", s), m] - inner.loc[("tpe", s), m] for m in ("search", "outer")}
    ahead = opening[s] > 0
    print(f"  seed {s}: opened ahead on {ahead.sum()}, final search {fin['search'][ahead].mean():+.4f} / "
          f"{fin['search'][~ahead].mean():+.4f} (ahead / not), outer {fin['outer'][ahead].mean():+.4f} / "
          f"{fin['outer'][~ahead].mean():+.4f}")

print("\nE. Calibration over the 108 datasets, per seed")
opt = {k: v.search - v.outer for k, v in (("confopt", inner.loc["confopt"]), ("tpe", inner.loc["tpe"]))}


def weights(ckpt, dataset):
    return [str(p["class_weight"]) for p in ckpt[dataset]["best_params"]]


def boost(ckpt, dataset):
    return np.mean([np.log(p["learning_rate"] * p["n_estimators"]) for p in ckpt[dataset]["best_params"]])


for s, (c, t) in enumerate(zip(CONFOPT, TPE)):
    ece = tab[c].loc[cal, "ece"] - tab[t].loc[cal, "ece"]
    share = [np.mean([w == "balanced" for x in cal for w in weights(ckpts[m], x)]) for m in (c, t)]
    extra = pd.Series({x: weights(ckpts[c], x).count("balanced") - weights(ckpts[t], x).count("balanced") for x in cal})
    never = [x for x in cal if "balanced" not in weights(ckpts[c], x) + weights(ckpts[t], x)]
    terms = {"optimism": opt["confopt"].loc[s][cal] - opt["tpe"].loc[s][cal], "extra balanced": extra,
             "log lr x trees": pd.Series({x: boost(ckpts[c], x) - boost(ckpts[t], x) for x in cal})}
    print(f"  seed {s}: ECE gap {ece.mean():+.4f}; 'balanced' {share[0]:.2f} / {share[1]:.2f}; "
          + ", ".join(f"{k} rho {spearmanr(ece, v)[0]:+.2f} (p {spearmanr(ece, v)[1]:.2g})" for k, v in terms.items())
          + f"; neither picks 'balanced' on {len(never)}: ECE gap {ece[never].mean():+.4f}")

print("\nF. Change in the ConfOpt - TPE gap from PR AUC to log-loss selection ('balanced' allowed)")
ll = {0: ("lgbm_confopt_logloss", "lgbm_logloss"), 1: ("lgbm_confopt_logloss_seed1", "lgbm_logloss_seed1")}
for m, e in {m: joblib.load(ckpt_path(m)) for pair in ll.values() for m in pair}.items():
    tab[m] = per_dataset(e, calibration).astype(float)
for metric, idx in (("prauc", slice(None)), ("ece", cal)):
    change = {s: ((tab[c][metric] - tab[t][metric]) - (tab[CONFOPT[s]][metric] - tab[TPE[s]][metric])).loc[idx]
              for s, (c, t) in ll.items()}
    for s in ll:
        gap(f"{metric}, seed {s}", change[s])
    gap(f"{metric}, mean of seeds 0 and 1", (change[0] + change[1]) / 2)

print("\nG. Time at 50 trials (TPE seed 1 has sequential folds; its parallel re-time is 3.95 h, scripts/dart_controls.py)")
for s, (c, t) in enumerate(zip(CONFOPT, TPE)):
    ratio = (tab[c].time / tab[t].time).median()
    print(f"  seed {s}: ConfOpt {tab[c].time.sum() / 3600:.2f} h, TPE {tab[t].time.sum() / 3600:.2f} h, "
          f"median per-dataset ratio {ratio:.1f}, median seconds {tab[c].time.median():.0f} / {tab[t].time.median():.0f}")

print("\nH. 100 - 50 trials per dataset on the three scores (seeds 0 and 1)")
both = pd.read_csv("results/search_inner_all.csv").groupby(["trials", "sampler", "seed", "dataset"])[["search", "fresh", "outer"]].mean()
for sampler in ("confopt", "tpe"):
    for s in (0, 1):
        g = both.loc[(100, sampler, s)] - both.loc[(50, sampler, s)]
        print(f"  {sampler:8s} seed {s}: search {g.search.mean():+.4f}, fresh {g.fresh.mean():+.4f}, outer {g.outer.mean():+.4f}; "
              f"lost on the fresh split {1 - g.fresh.mean() / g.search.mean():.0%}, fresh p {wilcoxon(g.fresh[g.fresh != 0]).pvalue:.2g}")

print("\nI. Checks added after the second review")
from scipy.stats import ttest_ind
for metric, idx in (("prauc", slice(None)), ("ece", cal)):
    c = np.array([tab[m][metric].loc[idx].mean() for m in CONFOPT])
    t = np.array([tab[m][metric].loc[idx].mean() for m in TPE])
    w = ttest_ind(c, t, equal_var=False)
    lo, hi = w.confidence_interval(0.95)
    print(f"  {metric}: Welch over the 5 + 5 seed means {c.mean() - t.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], p {w.pvalue:.3f}; "
          f"SD of seed means ConfOpt {c.std(ddof=1):.4f}, TPE {t.std(ddof=1):.4f}")
ece_p = {f"seed {s}": wilcoxon(tab[c].loc[cal, "ece"] - tab[t].loc[cal, "ece"]).pvalue for s, (c, t) in enumerate(zip(CONFOPT, TPE))}
print("  ECE per seed, Holm over five: " + ", ".join(f"{k} raw {ece_p[k]:.2g} Holm {v:.2g}" for k, v in holm(ece_p).items()))
print("  Brier per seed, ConfOpt / TPE: " + ", ".join(f"{tab[c].loc[cal, 'brier'].mean():.4f} / {tab[t].loc[cal, 'brier'].mean():.4f}"
                                                    for c, t in zip(CONFOPT, TPE)))
for s, (c, t) in ll.items():
    print(f"  log loss with 'balanced', seed {s}, ConfOpt / TPE: PR AUC {tab[c].prauc.mean():.4f} / {tab[t].prauc.mean():.4f}, "
          f"ECE {tab[c].loc[cal, 'ece'].mean():.4f} / {tab[t].loc[cal, 'ece'].mean():.4f}, "
          f"Brier {tab[c].loc[cal, 'brier'].mean():.4f} / {tab[t].loc[cal, 'brier'].mean():.4f}")

print(" Fresh split and kr-vs-k (five-seed means)")
folds = pd.read_csv("results/search_inner_all.csv").query("trials == 50 and dataset == 'kr-vs-k'")
print(folds[folds.sampler == "confopt"].assign(drop=lambda x: x.fresh - x.search).nsmallest(6, "drop")
      [["seed", "fold", "search", "fresh", "outer"]].round(3).to_string(index=False))
print(f"  kr-vs-k, TPE's largest fold drop from search to fresh {(folds[folds.sampler == 'tpe'].fresh - folds[folds.sampler == 'tpe'].search).min():+.3f}")
g5 = {m: sum(inner.loc[("confopt", s), m] - inner.loc[("tpe", s), m] for s in SEEDS) / 5 for m in ("search", "fresh")}
print(f"  kr-vs-k fresh - search, ConfOpt - TPE: {(g5['fresh'] - g5['search'])['kr-vs-k']:+.4f}")
gap("fresh, without kr-vs-k", g5["fresh"].drop("kr-vs-k"))
gap("fresh - search, without kr-vs-k", (g5["fresh"] - g5["search"]).drop("kr-vs-k"))

print(" Opening")
own = sum((inner.loc[("confopt", s), "search"] - start.loc[("confopt", s)])
          - (inner.loc[("tpe", s), "search"] - start.loc[("tpe", s)]) for s in SEEDS) / 5
gap("gain over own opening, ConfOpt - TPE", own)
op = np.array([opening[s].mean() for s in SEEDS])
print(f"  opening gap per seed SD {op.std(ddof=1):.4f}; final search gap per seed "
      f"{[round(float((inner.loc[('confopt', s), 'search'] - inner.loc[('tpe', s), 'search']).mean()), 4) for s in SEEDS]}")
for s in (0, 2, 4):
    not_ahead = opening[s] <= 0
    for m in ("search", "outer"):
        x = (inner.loc[("confopt", s), m] - inner.loc[("tpe", s), m])[not_ahead]
        print(f"  seed {s} where not ahead ({not_ahead.sum()}), {m}: {x.mean():+.4f}, p {wilcoxon(x[x != 0]).pvalue:.2g}")

print(" Calibration causes")
none10 = [x for x in cal if all("balanced" not in weights(ckpts[m], x) for m in CONFOPT + TPE)]
gap(f"ECE, five-seed gap where no run picks 'balanced'", (five("ece", CONFOPT, none10) - five("ece", TPE, none10)))
for s, (c, t) in enumerate(zip(CONFOPT, TPE)):
    ece = tab[c].loc[cal, "ece"] - tab[t].loc[cal, "ece"]
    never = [x for x in cal if "balanced" not in weights(ckpts[c], x) + weights(ckpts[t], x)]
    med = [np.median([p["learning_rate"] * p["n_estimators"] for x in cal for p in ckpts[m][x]["best_params"]]) for m in (c, t)]
    lead = inner.loc[("confopt", s), "search"][cal] - inner.loc[("tpe", s), "search"][cal]
    print(f"  seed {s}: neither picks 'balanced' on {len(never)}: median ECE gap {ece[never].median():+.4f}, "
          f"p {wilcoxon(ece[never]).pvalue:.2g}; median lr x trees {med[0]:.1f} / {med[1]:.1f}; "
          f"ECE gap vs search lead rho {spearmanr(ece, lead)[0]:+.2f} (p {spearmanr(ece, lead)[1]:.2g})")

print(" 100 against 50 trials")
lead = {n: sum(both.loc[(n, "confopt", s), "search"] - both.loc[(n, "tpe", s), "search"] for s in (0, 1)) / 2 for n in (50, 100)}
gap("inner lead, 100 - 50 trials", lead[100] - lead[50])
for s in (0, 1):
    c50, c100 = ("lgbm_confopt", "lgbm_confopt_trials100") if s == 0 else ("lgbm_confopt_seed1", "lgbm_confopt_trials100_seed1")
    t50, t100 = ("lgbm_rerun", "lgbm_trials100") if s == 0 else ("lgbm_seed1", "lgbm_trials100_seed1")
    pr = {m: per_dataset(joblib.load(ckpt_path(m)), set()).prauc.astype(float) for m in (c50, c100, t50, t100)}
    gap(f"outer 50-to-100 gain, ConfOpt - TPE, seed {s}", (pr[c100] - pr[c50]) - (pr[t100] - pr[t50]))

print(" TPE seed 1 against its parallel re-time")
par = joblib.load(ckpt_path("lgbm_seed1", "partime"))
seq = ckpts["lgbm_seed1"]
same = sum(a == b for d in seq for a, b in zip(seq[d]["best_params"], par[d]["best_params"]))
print(f"  identical best_params in {same} of {sum(len(e['best_params']) for e in seq.values())} folds; "
      f"median seconds {np.median([e['time'] for e in par.values()]):.0f}; "
      f"median ratio ConfOpt seed 1 / re-time {(tab['lgbm_confopt_seed1'].time / pd.Series({d: e['time'] for d, e in par.items()})).median():.1f}")
print(f"  five largest gains: {top.sum() / d[d > 0].sum():.0%} of the summed gains; five largest losses "
      f"{d.nsmallest(5).sum() / d[d < 0].sum():.0%} of the summed losses")
