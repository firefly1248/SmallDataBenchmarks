"""Block E: the foundation models against the boosters on post-release datasets, those deposited
after TabPFN-3's release (2026-05-05). Release dates are not training cutoffs, and for 7 of the 14
the data were public before. The question is whether their UCI++ lead leans on benchmark data in
training.

Pre-registered (plan of 2026-10-31): tabpfn3 against lgbm and against catboost, PR AUC, paired
Wilcoxon over datasets, Holm over the two pairs, paired bootstrap CI; the same pairs on UCI++;
the pairs per provenance stratum, no test; calibration (ECE, Brier) with raw p. Added after
review: the tabicl pairs (secondary), medians, the difference of mean gaps and a Mann-Whitney
test against UCI++, the stratum membership as fixed below, a matched UCI++ subset (binary,
catboost below 0.95 on both sides; the threshold is not pre-registered) and a regression of the gap on a
post-release indicator with dataset covariates. Difficulty is CatBoost's score, not LightGBM's:
LightGBM's own noise is in the gap.

Usage: uv run python -m scripts.post_release
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.data import datasets_to_run, load_data_df
from benchmark.results import CALIBRATION_POPULATION, compare, per_dataset
from benchmark.stats import holm

OUTPUT = "results/post_release.csv"
MODELS = ["lgbm", "catboost", "tabicl", "tabpfn3"]
PAIRS = [("tabpfn3", "lgbm"), ("tabpfn3", "catboost"), ("tabicl", "lgbm"), ("tabicl", "catboost")]
# Underlying data, or a paper on it, public before TabPFN-3's release (linked papers, repository records).
PREVIOUSLY_PUBLIC = ["dengue-mexico-deaths", "cholera-tz-result", "malpdf-fs43-binary", "biotech-y36",
                     "drrisk-retinopathy", "steroid-ocular-hypertension", "aedes-pupal-sex"]
rng = np.random.default_rng(0)


def boot_mean(x, n=10_000):
    return x[rng.integers(0, len(x), (n, len(x)))].mean(1)


def ci(b):
    return f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]"


names = datasets_to_run("post-release")
post_ckpt = {m: joblib.load(ckpt_path(m, "post-release")) for m in MODELS}
uci_ckpt = {m: joblib.load(ckpt_path(m)) for m in MODELS}
uci_cal = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
post_rows = {m: per_dataset(post_ckpt[m], set(names)) for m in MODELS}
pd.concat({m: post_rows[m] for m in MODELS}, names=["model", "dataset"]).to_csv(OUTPUT)

post = pd.DataFrame({m: post_rows[m]["prauc"].astype(float) for m in MODELS}).reindex(names)
uci = pd.DataFrame({m: per_dataset(uci_ckpt[m], set())["prauc"].astype(float) for m in MODELS})
uci = uci.loc[[d for d in datasets_to_run() if d in uci.index]].dropna()
pd.set_option("display.width", 250)
print(f"post-release {len(post)} datasets, UCI++ {len(uci)} scored by all four")
print("mean PR AUC, post-release:", post.mean().round(4).to_dict())
print("mean PR AUC, UCI++:", uci.mean().round(4).to_dict())

rows = []
for a, b in PAIRS:
    dp, du = (post[a] - post[b]).dropna(), uci[a] - uci[b]
    rows.append({"pair": f"{a} - {b}", "post_mean": dp.mean(), "post_ci": ci(boot_mean(dp.values)),
                 "post_median": dp.median(), "ahead": int((dp > 0).sum()), "behind": int((dp < 0).sum()),
                 "wilcoxon_p": wilcoxon(dp).pvalue, "uci_mean": du.mean(), "uci_median": du.median(),
                 "gap_diff": dp.mean() - du.mean(), "gap_diff_ci": ci(boot_mean(dp.values) - boot_mean(du.values)),
                 "mann_whitney_p": mannwhitneyu(dp, du).pvalue})
table = pd.DataFrame(rows)
primary = holm(dict(zip(table.pair[:2], table.wilcoxon_p[:2])))
table["holm_p"] = table.pair.map(primary)
print("\nPR AUC pairs (Holm over the two tabpfn3 pairs; the tabicl pairs are secondary):")
print(table.round(4).to_string(index=False))

new = [d for d in names if d not in PREVIOUSLY_PUBLIC]
print(f"\nStrata, mean / median gap: previously public (n {len(PREVIOUSLY_PUBLIC)}) and new (n {len(new)}):")
for a, b in PAIRS + [("tabpfn3", "tabicl")]:
    g = post[a] - post[b]
    print(f"  {a} - {b}: {g[PREVIOUSLY_PUBLIC].mean():+.4f} / {g[PREVIOUSLY_PUBLIC].median():+.4f}, "
          f"{g[new].mean():+.4f} / {g[new].median():+.4f}")

balance = lambda ckpts, cal: np.mean([p["balance_probabilities"] for d, e in ckpts["tabpfn3"].items()
                                       if d in cal for p in e["best_params"]])
print(f"\ntabpfn3 picks balance_probabilities=True in {balance(post_ckpt, set(names)):.0%} of the post-release folds, "
      f"{balance(uci_ckpt, uci_cal):.0%} of the UCI++ calibration folds")
print("Calibration (raw p):")
for label, ckpts, cal in (("post-release", post_ckpt, set(names)), ("UCI++, 108 calibration datasets", uci_ckpt, uci_cal)):
    out, _ = compare({f"{a} vs {b}": (ckpts[a], ckpts[b]) for a, b in PAIRS}, cal)
    print(f"  {label}")
    print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
          .round(4).to_string(index=False))
    if label == "post-release":
        tests = {f"{m} {c}": p for c, e, b in zip(out.comparison, out.ece_p, out.brier_p)
                 for m, p in (("ECE", e), ("Brier", b))}
        print("  Holm over the eight: " + ", ".join(f"{k} {v:.2g}" for k, v in holm(tests).items()))
out, _ = compare({f"tabpfn3 vs {b}": (post_ckpt["tabpfn3"], post_ckpt[b]) for b in ("lgbm", "catboost")},
                 set(names) - {"dengue-mexico-deaths"})
print("  post-release without dengue-mexico-deaths")
print(out[["comparison", "ece_base", "ece_variant", "ece_p", "brier_base", "brier_variant", "brier_p"]]
      .round(4).to_string(index=False))

# tabpfn3 - lgbm against dataset covariates: a matched UCI++ subset and a regression over both pools.
meta = pd.read_csv("results/datasets.csv").set_index("dataset")
cov = [{"post": 0, "gap": uci.loc[d, "tabpfn3"] - uci.loc[d, "lgbm"], "level": uci.loc[d, "catboost"],
        "n": meta.loc[d, "n_rows"], "p": meta.loc[d, "n_features"], "classes": meta.loc[d, "n_classes"],
        "minority": meta.loc[d, "minority_share"]} for d in uci.index]
for d in names:
    X, y, _ = load_data_df(d)
    cov.append({"post": 1, "gap": post.loc[d, "tabpfn3"] - post.loc[d, "lgbm"], "level": post.loc[d, "catboost"],
                "n": len(X), "p": X.shape[1], "classes": len(set(y)), "minority": min(np.mean(y == c) for c in set(y))})
cov = pd.DataFrame(cov)
matched = cov[(cov.post == 0) & (cov.classes == 2) & (cov.level < 0.95)].gap
pooled = cov[(cov.post == 1) & (cov.level < 0.95)].gap
binary = cov[(cov.post == 0) & (cov.classes == 2)].gap
print(f"\ntabpfn3 - lgbm, binary UCI++ (n {len(binary)}): mean {binary.mean():+.4f}, median {binary.median():+.4f}")
print(f"tabpfn3 - lgbm, matched UCI++ (binary, catboost < 0.95, n {len(matched)}): mean {matched.mean():+.4f}, "
      f"median {matched.median():+.4f}; post-release mean {pooled.mean():+.4f}, median {pooled.median():+.4f}; "
      f"difference {matched.mean() - pooled.mean():+.4f} {ci(boot_mean(matched.values) - boot_mean(pooled.values))}; "
      f"post-release n {len(pooled)}")
uci_gap = (uci.tabpfn3 - uci.lgbm).sort_values()
hill = [d for d in uci.index if d.startswith("hill-valley")]
cov["dataset"] = list(uci.index) + names
no_hill = cov[(cov.post == 0) & (cov.classes == 2) & (cov.level < 0.95) & ~cov.dataset.isin(hill)].gap
print(f"  matched without {', '.join(hill)}: n {len(no_hill)}, mean {no_hill.mean():+.4f}, median {no_hill.median():+.4f}")
for k in (5, 10):
    print(f"  UCI++ mean gap without its {k} largest: {uci_gap.iloc[:-k].mean():+.4f} "
          f"({uci_gap.iloc[-k:].sum() / uci_gap.sum():.0%} of the summed gap in them: {', '.join(uci_gap.index[-k:])})")
design = np.column_stack([np.ones(len(cov)), cov.post, np.log(cov.n), np.log(cov.p), cov.classes == 2,
                          cov.minority, cov.level]).astype(float)
fit = lambda i: np.linalg.lstsq(design[i], cov.gap.values[i], rcond=None)[0]
coef = fit(np.arange(len(cov)))
boots = np.array([fit(rng.integers(0, len(cov), len(cov))) for _ in range(2000)])
print(f"Regression of the gap over {len(cov)} datasets, bootstrap CI over datasets:")
for i, label in enumerate(["const", "post-release", "log n", "log p", "binary", "minority", "catboost level"]):
    print(f"  {label:12s} {coef[i]:+.4f} {ci(boots[:, i])}")
keep = np.flatnonzero(~cov.dataset.isin(hill).values)
print(f"  post-release without hill-valley: {fit(keep)[1]:+.4f} "
      f"{ci(np.array([fit(rng.choice(keep, len(keep)))[1] for _ in range(2000)]))}")
print(f"Saved {OUTPUT}")
