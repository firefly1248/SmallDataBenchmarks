"""es_time_explore: two exploratory additions to the es_time run, outside the plan (labelled exploratory).

curve   test AP at every tree count of T's pick on the E2 final window (thin and dense), the same curve on
        the three inner validation blocks, and the count read off them (a count tuned on time splits)
dr      plain refit of every inner candidate with the ER count of the same inner fit (configuration-level H1)
report  both analyses from the stored files

Usage: python -m scripts.es_time_explore curve | dr | report
Reads results/es_time/{thin,dense}/*.joblib, writes results/es_time/explore/. Changes no planned result.
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.es_time import DATASETS, MAX_TREES, OUT, SEEDS, THREADS, ap, candidates, layout, load, make, plain
from scripts.es_time_analyse import TIE, fmt, per_dataset

EXP = OUT / "explore"


def curve(ctx, params, fit, target):
    """AP of the target block after every tree count 1..MAX_TREES of one fit on `fit` (ranking by raw score)."""
    (a, b), (ta, tb) = fit, target
    X, y = ctx["X"], ctx["y"]
    model = make(params, ctx["seed"], MAX_TREES).fit(X.iloc[a:b], y[a:b])
    Xt = X.iloc[ta:tb]
    built = model.booster_.current_iteration()
    raw = np.cumsum([model.predict(Xt, raw_score=True, start_iteration=i, num_iteration=1) for i in range(built)], axis=0)
    out = np.array([ap(y[ta:tb], raw[min(k, built) - 1]) for k in range(1, MAX_TREES + 1)])
    return out


def curve_unit(pop, name, seed):
    X, y, t, cats = load(name, pop)
    g, lay = layout(t)
    ctx = dict(X=X, y=y, seed=seed)
    cands = candidates(name, seed)
    stored = joblib.load(OUT / pop / f"{name}_seed{seed}.joblib")
    fin = stored[(stored.scheme == "e2") & (stored.stage == "final")]
    jobs = []
    for k in range(3):
        pick = int(fin[(fin.cut == k) & (fin.arm == "t")].cand.iloc[0])
        cut = lay[("e2", k)]
        jobs.append((k, "test", delayed(curve)(ctx, cands[pick], *cut["final"])))
        jobs += [(k, f"val{j}", delayed(curve)(ctx, cands[pick], fit, val)) for j, (fit, val) in enumerate(cut["inner"])]
    res = Parallel(n_jobs=THREADS, prefer="threads")(j for _, _, j in jobs)
    rows = []
    for k in range(3):
        r = {role: c for (kk, role, _), c in zip(jobs, res) if kk == k}
        val = np.mean([r[f"val{j}"] for j in range(3)], axis=0)
        cell = fin[fin.cut == k].set_index("arm")
        rows.append(dict(pop=pop, dataset=name, seed=seed, cut=k, test=r["test"], val=val, k_val=int(np.argmax(val)) + 1,
                         k_t=int(cell.trees["t"]), k_er=int(cell.best["er_t"]), k_ef=int(cell.best["ef_t"]),
                         stored={a: float(cell.ap[a]) for a in ("t", "dr_t", "d_t")}))
    return rows


def dr_unit(pop, name, seed):
    X, y, t, cats = load(name, pop)
    g, lay = layout(t)
    ctx = dict(X=X, y=y, seed=seed)
    cands = candidates(name, seed)
    stored = joblib.load(OUT / pop / f"{name}_seed{seed}.joblib")
    inner = stored[stored.stage == "inner"]
    er = inner[inner.arm == "er"].set_index(["scheme", "cut", "block", "cand"]).best
    tasks, keys = [], []
    for s in (("e2", "e1") if pop == "thin" else ("e2",)):
        for k in range(3):
            for j, (fit, val) in enumerate(lay[(s, k)]["inner"]):
                for c in range(len(cands)):
                    keys.append((s, k, j, c))
                    tasks.append(delayed(plain)(ctx, cands[c], fit, val, trees=int(er[(s, k, j, c)])))
    res = Parallel(n_jobs=THREADS, prefer="threads")(tasks)
    return pd.DataFrame([dict(pop=pop, dataset=name, seed=seed, scheme=s, cut=k, block=j, cand=c, ap=r["ap"])
                         for (s, k, j, c), r in zip(keys, res)])


def run(kind):
    (EXP / kind).mkdir(parents=True, exist_ok=True)
    units = [(pop, name, seed) for pop in DATASETS for seed in SEEDS for name in DATASETS[pop]]
    for i, (pop, name, seed) in enumerate(units, 1):
        path = EXP / kind / f"{pop}_{name}_seed{seed}.joblib"
        if path.exists():
            continue
        start = time.time()
        out = (curve_unit if kind == "curve" else dr_unit)(pop, name, seed)
        joblib.dump(out, f"{path}.tmp")
        path.with_name(path.name + ".tmp").replace(path)
        print(f"[{kind} {i}/{len(units)}] {pop} {name} seed {seed}: wall {time.time() - start:.0f} s", flush=True)
    print("Done.", flush=True)


# ---------------------------------------------------------------- report

def cell_frame(rows, rule_cols):
    """One row per cell: AP at each count rule, the oracle AP and count."""
    out = []
    for r in rows:
        at = lambda k: r["test"][k - 1]
        out.append(dict(pop=r["pop"], dataset=r["dataset"], seed=r["seed"], cut=r["cut"], t=at(r["k_t"]),
                        er=at(r["k_er"]), ef=at(r["k_ef"]), val=at(r["k_val"]), oracle=r["test"].max(),
                        k_oracle=int(np.argmax(r["test"])) + 1, k_val=r["k_val"], k_t=r["k_t"], k_er=r["k_er"],
                        k_ef=r["k_ef"], **{f"chk_{a}": abs(v - at({"t": r["k_t"], "dr_t": r["k_er"], "d_t": r["k_ef"]}[a]))
                                           for a, v in r["stored"].items()}))
    return pd.DataFrame(out).set_index(["pop", "dataset", "seed", "cut"])


def report():
    rows = [r for f in sorted((EXP / "curve").glob("*.joblib")) for r in joblib.load(f)]
    d = cell_frame(rows, None)
    chk = d[[c for c in d.columns if c.startswith("chk_")]].max()
    print(f"cells {len(d)}; largest gap between the curve and the stored T, DR_T, D_T scores: "
          + ", ".join(f"{k[4:]} {v:.1e}" for k, v in chk.items()))
    for pop in ("thin", "dense"):
        x = d.xs(pop, level="pop")
        print(f"\nCurve at T's pick, {pop} E2 ({x.index.get_level_values('dataset').nunique()} datasets, "
              f"{len(x)} cells); rules: t = T's drawn count, er = ER_T count (DR_T), ef = EF_T count (D_T), "
              f"val = count that maximises the mean AP of the three inner validation blocks")
        print(" regret = oracle AP minus AP at the rule's count (mean over datasets of cell means):")
        for rule in ("t", "er", "ef", "val"):
            print(f"  {rule}: {fmt((x.oracle - x[rule]).groupby('dataset').mean())}")
        print(" differences of test AP (positive: first rule ahead):")
        for a, b in (("val", "t"), ("val", "er"), ("val", "ef"), ("t", "er"), ("er", "ef")):
            per = (x[a] - x[b]).groupby("dataset").mean()
            print(f"  {a} - {b}: {fmt(per)}; ahead {int((per >= TIE).sum())} / behind {int((per <= -TIE).sum())}")
        med = x.groupby("dataset")[["k_oracle", "k_val", "k_t", "k_er", "k_ef"]].median()
        print(" median tree count per dataset (oracle, val, t, er, ef):")
        print(med.round(0).astype(int).to_string())
    dr = pd.concat([joblib.load(f) for f in sorted((EXP / "dr").glob("*.joblib"))], ignore_index=True)
    fits = pd.concat([joblib.load(f) for pop in DATASETS for f in sorted((OUT / pop).glob("*.joblib"))], ignore_index=True)
    inner = fits[fits.stage == "inner"].pivot_table(index=["pop", "scheme", "dataset", "seed", "cut", "block", "cand"],
                                                    columns="arm", values="ap")
    m = inner.join(dr.set_index(["pop", "scheme", "dataset", "seed", "cut", "block", "cand"]).ap.rename("dr"))
    assert m.dr.notna().all()
    print("\nInner fits, same candidate and block in both arms (mean over datasets of dataset means; "
          "positive: first ahead); refit = plain fit on all rows with the count of the ES fit:")
    for pop, scheme in (("thin", "e2"), ("thin", "e1"), ("dense", "e2")):
        x = m.xs((pop, scheme), level=("pop", "scheme"))
        print(f" {pop} {scheme} ({len(x)} pairs, {x.index.get_level_values('dataset').nunique()} datasets)")
        for name, (a, b) in {"d_own - dr (configuration-level H1: EF count vs ER count, both refit)": ("d_own", "dr"),
                             "d_own - ef (all rows vs older 90 %, EF count)": ("d_own", "ef"),
                             "dr - er (all rows vs random 90 %, ER count)": ("dr", "er"),
                             "ef - er (older 90 % vs random 90 %)": ("ef", "er")}.items():
            per = (x[a] - x[b]).groupby("dataset").mean()
            print(f"  {name}: {fmt(per)}; ahead {int((per >= TIE).sum())} / behind {int((per <= -TIE).sum())}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in ("curve", "dr"):
        run(command)
    elif command == "report":
        report()
    else:
        print(__doc__)
