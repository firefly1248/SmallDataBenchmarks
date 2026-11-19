"""es_time_analyse: delta step and analysis of the es_time run. Plan: early_stopping_time_plan.md.

delta     reads only T's final fits of the thin E2 results, writes results/es_time/delta.json (run once, first)
analyse   the plan's four hypotheses with labels, sensitivities and diagnostics; needs delta.json, never
          recomputes it; run once after all seeds finish. Pipe to results/es_time_analysis.txt
selftest  both steps on synthetic fits with a planted effect; reads no results

Usage: python -m scripts.es_time_analyse delta | analyse | selftest
"""
import json
import sys
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from scripts.es_time import CAPPED, DATASETS, OUT, PREPARED, SEEDS, layout, load, sha256

LEVEL, TIE = 0.9875, 1e-4
HYP = {"H1": ("d_t", "dr_t"), "H2": ("d", "ef"), "H3": ("t", "d_own"), "H4": ("t", "er")}
DESCRIPTIVE = {"D - DR": ("d", "dr"), "T - EF": ("t", "ef"), "D - D_own": ("d", "d_own"), "DR - ER": ("dr", "er"),
               "D_T9 - D_T": ("d_t9", "d_t")}
PAIRS = [("d_own_es", "ef"), ("er", "ef"), ("t", "ef"), ("t", "er")]
KEY = ["pop", "scheme", "dataset", "seed", "cut"]


def half_width(values, level=LEVEL):
    n = len(values)
    return t_dist.ppf(1 - (1 - level) / 2, n - 1) * np.std(values, ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- delta

def split_halves(s):
    """Half-width of the pooled interval of T's seed-pair contrast, for the 15 splits of five seeds into two
    disjoint pairs. s: dataset x seed frame of T's mean test score over cuts."""
    out = []
    for left_out in SEEDS:
        rest = [k for k in SEEDS if k != left_out]
        for j in (1, 2, 3):
            a = [rest[0], rest[j]]
            b = [k for k in rest if k not in a]
            out.append(half_width((s[a].mean(axis=1) - s[b].mean(axis=1)).to_numpy()))
    return out


def delta():
    files = sorted((OUT / "thin").glob("*.joblib"))
    keep = [("arm", "t"), ("stage", "final"), ("scheme", "e2")]
    rows = pd.concat([x[(x.arm == "t") & (x.stage == "final") & (x.scheme == "e2")][["dataset", "seed", "cut", "ap"]]
                      for x in map(joblib.load, files)])
    assert rows.dataset.nunique() == len(DATASETS["thin"]) and rows.seed.nunique() == len(SEEDS) and rows.ap.notna().all()
    halves = split_halves(rows.groupby(["dataset", "seed"]).ap.mean().unstack("seed"))
    value = float(np.sqrt(np.mean(np.square(halves))))
    json.dump({"delta": value, "half_widths": halves, "level": LEVEL, "rows_read": f"{keep}, {len(rows)} rows",
               "inputs": {f.name: sha256(f) for f in files}}, open(OUT / "delta.json", "w"), indent=1)
    print(f"delta {value:.5f} from {len(halves)} splits (half-widths {min(halves):.5f} to {max(halves):.5f})")


# ---------------------------------------------------------------- labels

def label(lo, hi, delta_):
    direction = "first fit ahead" if lo > 0 else "second fit ahead" if hi < 0 else "no difference found"
    if lo > delta_ or hi < -delta_:
        return direction + ", beyond delta"
    if lo > -delta_ and hi < delta_:
        return direction + ", equivalent within delta"
    return direction


def flagged(per, delta_):
    """Datasets whose removal flips the sign of the pooled mean or moves it across +delta or -delta."""
    full = per.mean()
    out = []
    for d in per.index:
        loo = per.drop(d).mean()
        if (np.sign(full) != np.sign(loo) or (full > delta_) != (loo > delta_) or (full < -delta_) != (loo < -delta_)):
            out.append(d)
    return out


def pooled(per):
    """Mean over datasets with its t-interval; None for fewer than two datasets."""
    if len(per) < 2:
        return per.mean(), np.nan, np.nan
    h = half_width(per.to_numpy())
    return per.mean(), per.mean() - h, per.mean() + h


def fmt(per):
    m, lo, hi = pooled(per)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(per)})"


# ---------------------------------------------------------------- analysis

def cells(fits):
    """One row per population, scheme, dataset, seed and cut; columns (value, arm) of the final fits."""
    f = fits[fits.stage == "final"]
    return f.set_index(KEY + ["arm"])[["ap", "trees", "best", "cand"]].unstack("arm")


def diff(w, a, b):
    return w["ap"][a] - w["ap"][b]


def per_dataset(d, pop, scheme):
    return d.xs((pop, scheme), level=("pop", "scheme")).groupby("dataset").mean()


def check_complete(fits):
    print("Before analysis:")
    per_unit = fits.groupby(["pop", "dataset", "seed"]).size()
    want = {"thin": 2 * 3 * 732, "dense": 3 * 732}
    short = [(k, n) for k, n in per_unit.items() if n != want[k[0]]]
    expected = {p: len(DATASETS[p]) * len(SEEDS) for p in DATASETS}
    print(f"  units per population: {per_unit.groupby('pop').size().to_dict()} (expected {expected}); "
          f"units with a wrong row count: {short or 'none'}")
    failed = fits[(fits.error != "") | fits.ap.isna()]
    print(f"  failed fits or NaN scores: {len(failed)}")
    for r in failed.head(20).itertuples():
        print(f"    {r.pop} {r.dataset} seed {r.seed} {r.scheme} cut {r.cut} {r.stage} {r.arm} cand {r.cand}: {r.error}")
    assert not short and per_unit.groupby("pop").size().to_dict() == expected, "incomplete run"


def report(fits, delta_, summary):
    w = cells(fits)
    thin = summary["thin"]
    low = list(thin.index[thin.low_signal])
    lag = list(thin.index[thin.c3_ef_max > 0.05])
    final = fits[(fits.stage == "final") & (fits["pop"] == "thin") & (fits.scheme == "e2")]
    share = {a: final[final.arm == a].groupby("dataset").best.apply(lambda b: (b > CAPPED).mean())
             for a in ("er_t", "ef_t")}
    capped = list(share["er_t"].index[np.maximum(share["er_t"], share["ef_t"]) > 0.25])
    equal = (w["trees"]["d_t"] == w["trees"]["dr_t"]).astype(float)

    print(f"\nPrimary: thin, E2 (differences of test average precision, mean over 3 cuts and 5 seeds per dataset; "
          f"pooled over {w.xs(('thin', 'e2'), level=('pop', 'scheme')).index.get_level_values('dataset').nunique()} "
          f"datasets; t-interval at {LEVEL:.2%}; delta {delta_:.5f})")
    for h, (a, b) in HYP.items():
        d = diff(w, a, b)
        per = per_dataset(d, "thin", "e2")
        m, lo, hi = pooled(per)
        ahead, behind = int((per >= TIE).sum()), int((per <= -TIE).sum())
        bad = flagged(per, delta_)
        print(f" {h} {a.upper()} - {b.upper()}: {fmt(per)}, half-width {(hi - lo) / 2:.4f}\n"
              f"    label: {label(lo, hi, delta_)} (delta {delta_:.5f}); ahead {ahead} / behind {behind} / tied "
              f"{len(per) - ahead - behind}; leave-one-out: {'not robust, flagged by ' + ', '.join(bad) if bad else 'robust'}; "
              f"cells without a value: {int(d.xs(('thin', 'e2'), level=('pop', 'scheme')).isna().sum())}")
        print(f"    without low-signal datasets ({', '.join(low) or 'none'}): {fmt(per.drop(low))}")
        if h == "H1":
            print(f"    without datasets above 5 % EF stopping-set label lag ({', '.join(lag) or 'none'}): "
                  f"{fmt(per.drop(lag))}")
            print(f"    without datasets with over 25 % capped ER_T or EF_T final fits ({', '.join(capped) or 'none'}): "
                  f"{fmt(per.drop(capped))}")
            print(f"    cells with equal tree counts for D_T and DR_T, per dataset: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in per_dataset(equal, "thin", "e2").items()))
        for k in range(3):
            pc = d.xs(("thin", "e2", k), level=("pop", "scheme", "cut")).groupby("dataset").mean()
            print(f"    cut {k + 1}: {fmt(pc)}")
    print("\nSensitivity analyses, no labels: same estimates and intervals")
    for pop, scheme, name in (("thin", "e1", "E1 (expanding window, thin)"), ("dense", "e2", "dense, E2")):
        print(f" {name}")
        for h, (a, b) in HYP.items():
            print(f"  {h}: {fmt(per_dataset(diff(w, a, b), pop, scheme))}")

    print("\nDescriptive differences, thin E2 (no label):")
    for name, (a, b) in DESCRIPTIVE.items():
        print(f"  {name}: {fmt(per_dataset(diff(w, a, b), 'thin', 'e2'))}")
    ratio = (w["best"]["er_t"] / w["best"]["ef_t"]).xs(("thin", "e2"), level=("pop", "scheme")).groupby("dataset").median()
    print("  median ER_T / EF_T best iteration per dataset: " + ", ".join(f"{k} {v:.2f}" for k, v in ratio.items()))
    print("  share of cells where two arms pick the same candidate (mean over datasets):")
    for a, b in PAIRS:
        same = (w["cand"][a] == w["cand"][b]).astype(float)
        print(f"    {a} = {b}: {per_dataset(same, 'thin', 'e2').mean():.2f}")

    print("\nDiagnostics, thin E2:")
    f = fits[(fits["pop"] == "thin") & (fits.scheme == "e2")]
    print(" median trees per arm and cut (inner, final):")
    print(f.groupby(["stage", "arm", "cut"]).trees.median().unstack("cut").round(0).to_string())
    es = f[f.arm.isin(["er", "ef", "er_t", "ef_t", "d_own_es"])]
    print(" capped share of ES fits (best iteration above 480):")
    print(es.groupby(["stage", "arm"]).best.apply(lambda b: (b > CAPPED).mean()).round(3).to_string())
    fin = f[(f.stage == "final") & f.arm.isin(["er", "ef"])].assign(
        ll_gap=lambda d: d.tgt_ll - d.stop_ll, ap_gap=lambda d: d.ap - d.stop_ap)
    print(" stopping-set optimism, final ER and EF fits (test minus stopping set), unseen-level shares, best iterations:")
    print(fin.groupby("arm")[["ll_gap", "ap_gap", "unseen_stop", "unseen_tgt", "best", "best_ap_iter"]]
          .agg(lambda x: x.mean() if x.name not in ("best", "best_ap_iter") else x.median()).round(4).to_string())
    print(" thread seconds per arm and stage, all populations:")
    print(fits.groupby(["stage", "arm"]).sec.sum().round(0).to_string())


def stopping_set_gaps(w, summary):
    """H1 and H2 per dataset and cut with the minority-share gaps and persistence they are plotted against."""
    rows = []
    for pop in DATASETS:
        for name in DATASETS[pop]:
            _, y, t, _ = load(name, pop)
            g, lay = layout(t)
            for k in range(3):
                (a, b), (ta, tb) = lay[("e2", k)]["final"]
                s = g["stop_start"](a, b)
                cell = w.xs((pop, "e2", name, k), level=("pop", "scheme", "dataset", "cut"))
                rows.append(dict(pop=pop, dataset=name, cut=k + 1, h1=(cell["ap"]["d_t"] - cell["ap"]["dr_t"]).mean(),
                                 h2=(cell["ap"]["d"] - cell["ap"]["ef"]).mean(),
                                 gap_test=y[s:b].mean() - y[ta:tb].mean(), gap_older=y[s:b].mean() - y[a:s].mean(),
                                 persistence=summary[pop].persistence[name]))
    return pd.DataFrame(rows)


def per_dataset_table(w):
    """Per dataset: the four differences per cut and seed, mean and 95 % t-interval over the five seeds."""
    rows = []
    for h, (a, b) in HYP.items():
        d = diff(w, a, b).unstack("seed")
        for (pop, scheme, name, cut), v in d.iterrows():
            rows.append((pop, scheme, name, h, cut + 1) + tuple(v))
    out = pd.DataFrame(rows, columns=["pop", "scheme", "dataset", "hyp", "cut"] + [f"seed{s}" for s in SEEDS])
    allcuts = out.groupby(["pop", "scheme", "dataset", "hyp"])[[f"seed{s}" for s in SEEDS]].mean().reset_index().assign(cut=0)
    out = pd.concat([out, allcuts], ignore_index=True)
    v = out[[f"seed{s}" for s in SEEDS]].to_numpy()
    out["mean"] = np.nanmean(v, axis=1)
    out["lo95"] = out["mean"] - t_dist.ppf(0.975, len(SEEDS) - 1) * np.nanstd(v, axis=1, ddof=1) / np.sqrt(len(SEEDS))
    out["hi95"] = 2 * out["mean"] - out["lo95"]
    return out


def analyse():
    meta = json.load(open(OUT / "delta.json"))
    files = sorted((OUT / "thin").glob("*.joblib"))
    assert {f.name: sha256(f) for f in files} == meta["inputs"], "thin results changed since delta was computed"
    fits = pd.concat([joblib.load(f) for pop in DATASETS for f in sorted((OUT / pop).glob("*.joblib"))],
                     ignore_index=True)
    check_complete(fits)
    summary = {pop: pd.read_csv(PREPARED / f"_summary_{pop}.csv").set_index("dataset") for pop in DATASETS}
    report(fits, meta["delta"], summary)
    w = cells(fits)
    per_dataset_table(w).round(6).to_csv(OUT / "per_dataset.csv", index=False)
    stopping_set_gaps(w, summary).round(6).to_csv(OUT / "stopping_set_gaps.csv", index=False)


# ---------------------------------------------------------------- self test

def fake_fits(rng, effect):
    """Synthetic fits with the real schema: D_T is `effect` ahead of DR_T, everything else is noise."""
    rows = []
    for pop, schemes in (("thin", ("e2", "e1")), ("dense", ("e2",))):
        for name in DATASETS[pop][:4]:
            for seed in SEEDS:
                for scheme in schemes:
                    for cut in range(3):
                        base = 0.3 + 0.05 * rng.standard_normal()
                        for arm in ("t", "er", "ef", "d", "dr", "d_own_es", "d_own", "er_t", "ef_t", "dr_t", "d_t", "d_t9"):
                            es = arm in ("er", "ef", "er_t", "ef_t", "d_own_es")
                            rows.append(dict(pop=pop, scheme=scheme, dataset=name, seed=seed, cut=cut, stage="final",
                                             arm=arm, block=-1, cand=int(rng.integers(3)), ap=base + 0.003 * rng.standard_normal()
                                             + (effect if arm == "d_t" else 0.0), trees=100, best=100 if es else np.nan,
                                             best_ap_iter=80 if es else np.nan, stop_ll=0.5, stop_ap=0.4, tgt_ll=0.55,
                                             unseen_stop=0.0, unseen_tgt=0.0, sec=1.0, error=""))
    return pd.DataFrame(rows)


def selftest():
    assert label(0.001, 0.003, 0.005) == "first fit ahead, equivalent within delta"
    assert label(-0.002, 0.003, 0.005) == "no difference found, equivalent within delta"
    assert label(0.006, 0.01, 0.005) == "first fit ahead, beyond delta"
    assert label(-0.02, -0.008, 0.005) == "second fit ahead, beyond delta"
    assert label(-0.02, 0.02, 0.005) == "no difference found"
    assert label(-0.001, 0.02, 0.005) == "no difference found"
    per = pd.Series([0.01, 0.011, 0.012, 0.0, 0.2], index=list("abcde"))
    assert flagged(per, 0.01) == ["e"]
    rng = np.random.default_rng(0)
    fits = fake_fits(rng, effect=0.02)
    s = fits[(fits.arm == "t") & (fits["pop"] == "thin") & (fits.scheme == "e2")].groupby(["dataset", "seed"]).ap.mean().unstack()
    halves = split_halves(s)
    assert len(halves) == 15 and all(h > 0 for h in halves)
    summary = {p: pd.DataFrame({"low_signal": [False, False, False, True], "c3_ef_max": [0.0, 0.1, 0.0, 0.0],
                                "persistence": 1.0}, index=DATASETS[p][:4]) for p in DATASETS}
    report(fits, float(np.sqrt(np.mean(np.square(halves)))), summary)
    w = cells(fits)
    h1 = per_dataset(diff(w, "d_t", "dr_t"), "thin", "e2")
    assert abs(h1.mean() - 0.02) < 0.005 and (per_dataset(diff(w, "d", "ef"), "thin", "e2").abs() < 0.01).all()
    assert len(per_dataset_table(w)) == 3 * 4 * len(HYP) * 4  # populations and schemes x datasets x hypotheses x (3 cuts + all)
    print("selftest passed: planted H1 effect +0.02 recovered, labels and leave-one-out as specified")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    {"delta": delta, "analyse": analyse, "selftest": selftest}.get(command, lambda: print(__doc__))()
