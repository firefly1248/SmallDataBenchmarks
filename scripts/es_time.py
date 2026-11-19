"""es_time: early stopping on time-ordered data. Plan: early_stopping_time_plan.md.

LightGBM over 60 random candidates (learning rate from 0.01), scored on three time-ordered cuts by T
(tuned n_estimators), ER and EF (early stop on a random or on the freshest 10 %), D_own (EF's tree count,
refit on all rows) and the final-fit arms of the plan's table. Every fit runs on one thread and is timed
with time.thread_time(). Categoricals are native (pandas category). Inputs are the prepared files of
es_time_prep.py, checked against its manifest.

Usage: python -m scripts.es_time check                     asserts, candidates, one reduced unit
       python -m scripts.es_time run thin|dense [dataset ...]   resumable per dataset and seed
Environment: ES_THREADS (default 6) sets the number of concurrent fits.
"""
import hashlib
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from scripts.es_time_prep import geometry

SEEDS = range(5)
N_CAND = 60
MAX_TREES, PATIENCE, STOP_SHARE = 500, 20, 0.1
CAPPED = MAX_TREES - PATIENCE
THREADS = int(os.environ.get("ES_THREADS", 6))
ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "results/es_time/prepared"
OUT = ROOT / "results/es_time"
DATASETS = {"thin": ["hotels", "himalaya", "wildlife", "f1-dnf", "nba", "elec", "nebraska-weather", "nycflights",
                     "bank-full"],
            "dense": ["hotels", "wildlife", "f1-dnf", "nba", "elec", "bank-full"]}
ES_ARMS = ("er", "ef", "er_t", "ef_t", "d_own_es")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------- data, geometry, candidates

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, pop):
    """Features, label (1 = minority) and time of a prepared file, with its sha256 checked."""
    manifest = json.load(open(PREPARED / "_manifest.json"))["prepared"]
    side = json.load(open(PREPARED / f"{name}.json"))
    file = f"{name}_{pop}.csv"
    assert sha256(PREPARED / file) == manifest[file] == side["sha256"][file], f"{file} differs from its manifest"
    df = pd.read_csv(PREPARED / file)
    if df.t.dtype == object:
        df["t"] = pd.to_datetime(df.t)
    feats = [c for c in df.columns if c not in ("t", "y", "label_known")]
    assert set(feats) <= set(side["features"]), "a feature outside the sidecar list"
    assert not {"t", "label_known", "y"} & set(feats)
    X = df[feats].copy()
    cats = [c for c in side["categorical"] if c in feats]
    for c in cats:
        X[c] = X[c].astype("category")
    return X, df.y.to_numpy(), df.t.to_numpy(), cats


def space():
    """lgbm_space of the benchmark with the learning rate from 0.01."""
    return {"learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "n_estimators": IntDistribution(50, 500),
            "num_leaves": IntDistribution(8, 256),
            "min_child_samples": IntDistribution(5, 100),
            "subsample": FloatDistribution(0.4, 1.0),
            "colsample_bytree": FloatDistribution(0.4, 1.0),
            "reg_alpha": FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": FloatDistribution(1e-8, 10.0, log=True),
            "class_weight": CategoricalDistribution(["balanced", None])}


def candidates(dataset, seed):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=zlib.crc32(dataset.encode()) + seed))
    out = []
    for _ in range(N_CAND):
        trial = study.ask(space())
        out.append(trial.params)
        study.tell(trial, 0.0)
    assert all(c["learning_rate"] >= 0.01 for c in out)
    return out


def layout(t):
    """Per scheme and cut: the three inner (window, validation block) pairs and the final (window, test block)."""
    g = geometry(t)
    lay = {(s, k): dict(inner=[(g[s][k][j], g["val"][3 * k + j]) for j in range(3)], final=(g[s][k][3], g["test"][k]))
           for s in ("e2", "e1") for k in range(3)}
    return g, lay


def check_geometry(t, g, lay):
    n = len(t)
    assert np.all(t[1:] >= t[:-1]), "rows not sorted by time"
    points = set()
    tie_run = int(pd.Series(t).value_counts().max())
    for (scheme, k), cut in lay.items():
        for (wa, wb), (va, vb) in cut["inner"] + [cut["final"]]:
            assert wb == va and wa < wb <= vb <= n, "a window must end where its block starts"
            s = g["stop_start"](wa, wb)
            assert wa < s < wb
            points |= {wa, wb, va, vb, s}
            if scheme == "e2":
                assert abs((wb - wa) - round(0.36 * n)) <= 2 * tie_run + 1, "E2 windows differ by more than the shift"
    for p in points - {0, n}:
        assert t[p] != t[p - 1], f"timestamp {t[p]} on both sides of position {p}"


# ---------------------------------------------------------------- fits

def make(params, seed, n_estimators=None):
    p = {**params, "n_estimators": n_estimators or params["n_estimators"]}
    return lightgbm.LGBMClassifier(**p, boosting_type="gbdt", objective="binary", subsample_freq=1,
                                   random_state=seed, n_jobs=1, verbose=-1)


def ap(y, p):
    return float(average_precision_score(y, p))


def unseen_share(train, other, cats):
    """Share of rows of `other` with a categorical level absent from `train`."""
    bad = np.zeros(len(other), bool)
    for c in cats:
        bad |= (other[c].notna() & ~other[c].isin(train[c].dropna().unique())).to_numpy()
    return float(bad.mean())


def plain(ctx, params, fit, target, trees=None):
    """No evaluation set: all rows of `fit`, scored on `target`. `trees` overrides n_estimators (refits)."""
    (a, b), (ta, tb) = fit, target
    X, y = ctx["X"], ctx["y"]
    assert np.unique(y[a:b]).size == 2 and np.unique(y[ta:tb]).size == 2
    model = make(params, ctx["seed"], trees).fit(X.iloc[a:b], y[a:b])
    return {"ap": ap(y[ta:tb], model.predict_proba(X.iloc[ta:tb])[:, 1]),
            "trees": trees or params["n_estimators"], "built": model.booster_.current_iteration()}


def stopped(ctx, params, fit, target, how, diag=False):
    """ER (how = "random") or EF (how = "time"): the model on 90 % of `fit`, early stopped on log loss over
    the other 10 %, at most MAX_TREES trees, scored on `target` at the best iteration."""
    (a, b), (ta, tb) = fit, target
    X, y, seed = ctx["X"], ctx["y"], ctx["seed"]
    if how == "time":
        s = ctx["stop_start"](a, b)
        keep, stop = np.arange(a, s), np.arange(s, b)
    else:
        keep, stop = train_test_split(np.arange(a, b), test_size=STOP_SHARE, stratify=y[a:b], random_state=seed)
        keep, stop = np.sort(keep), np.sort(stop)
    assert np.unique(y[keep]).size == 2 and np.unique(y[ta:tb]).size == 2
    assert 5 <= y[stop].sum() < len(stop), "a stopping set needs 5 minority rows and both classes"
    weights = {}
    if params["class_weight"] == "balanced":
        weights["eval_class_weight"] = [dict(zip([0, 1], compute_class_weight("balanced", classes=np.array([0, 1]),
                                                                              y=y[keep])))]
    model = make({**params, "n_estimators": MAX_TREES}, seed)
    model.fit(X.iloc[keep], y[keep], eval_set=[(X.iloc[stop], y[stop])],
              eval_metric=["binary_logloss", "average_precision"],
              callbacks=[lightgbm.early_stopping(PATIENCE, first_metric_only=True, verbose=False)], **weights)
    p_stop, p_target = model.predict_proba(X.iloc[stop])[:, 1], model.predict_proba(X.iloc[ta:tb])[:, 1]
    out = {"ap": ap(y[ta:tb], p_target), "best": int(model.best_iteration_), "trees": int(model.best_iteration_),
           "built": model.booster_.current_iteration(),
           "best_ap_iter": int(np.argmax(model.evals_result_["valid_0"]["average_precision"])) + 1,
           "stop_ll": float(log_loss(y[stop], p_stop, labels=[0, 1])), "stop_ap": ap(y[stop], p_stop),
           "tgt_ll": float(log_loss(y[ta:tb], p_target, labels=[0, 1]))}
    if diag:
        cats = ctx["cats"]
        out["unseen_stop"] = unseen_share(X.iloc[keep], X.iloc[stop], cats)
        out["unseen_tgt"] = unseen_share(X.iloc[keep], X.iloc[ta:tb], cats)
    return out


def attempt(arm, base, fn):
    start = time.thread_time()
    try:
        r = {"error": "", **fn()}
    except Exception as error:
        r = {"ap": np.nan, "error": repr(error)}
    return {**base, "arm": arm, **r, "sec": time.thread_time() - start}


def inner_task(ctx, scheme, cut, block, cand, fit, val):
    base = dict(scheme=scheme, cut=cut, stage="inner", block=block, cand=cand)
    params = ctx["cands"][cand]
    out = [attempt("t", base, lambda: plain(ctx, params, fit, val))]
    out.append(attempt("er", base, lambda: stopped(ctx, params, fit, val, "random")))
    ef = attempt("ef", base, lambda: stopped(ctx, params, fit, val, "time"))
    out.append(ef)
    out.append(attempt("d_own", base, lambda: plain(ctx, params, fit, val, trees=ef["best"])))  # NaN best -> error row
    return out


def ranked(scores):
    """Candidate positions, best first; NaN last; ties by position."""
    s = np.where(np.isnan(scores), np.inf, -np.asarray(scores, dtype=float))
    return np.lexsort((np.arange(len(s)), s))


def final_task(ctx, scheme, cut, inner, fit, test):
    """The 12 final fits of the plan's table on one cut, from the picks of the inner fits."""
    inner = inner[(inner.scheme == scheme) & (inner.cut == cut)]
    picks = {}
    for arm in ("t", "er", "ef", "d_own"):
        d = inner[inner.arm == arm].pivot(index="cand", columns="block", values="ap")
        picks[arm] = int(ranked(d.reindex(range(len(ctx["cands"]))).mean(axis=1, skipna=False).to_numpy())[0])
    base = dict(scheme=scheme, cut=cut, stage="final", block=-1)
    c = ctx["cands"]
    rec = {}

    def run(arm, cand, fn):
        rec[arm] = attempt(arm, {**base, "cand": cand}, fn)

    run("t", picks["t"], lambda: plain(ctx, c[picks["t"]], fit, test))
    run("er", picks["er"], lambda: stopped(ctx, c[picks["er"]], fit, test, "random", diag=True))
    run("ef", picks["ef"], lambda: stopped(ctx, c[picks["ef"]], fit, test, "time", diag=True))
    run("d", picks["ef"], lambda: plain(ctx, c[picks["ef"]], fit, test, trees=rec["ef"]["best"]))
    run("dr", picks["er"], lambda: plain(ctx, c[picks["er"]], fit, test, trees=rec["er"]["best"]))
    run("d_own_es", picks["d_own"], lambda: stopped(ctx, c[picks["d_own"]], fit, test, "time", diag=True))
    run("d_own", picks["d_own"], lambda: plain(ctx, c[picks["d_own"]], fit, test, trees=rec["d_own_es"]["best"]))
    run("er_t", picks["t"], lambda: stopped(ctx, c[picks["t"]], fit, test, "random", diag=True))
    run("ef_t", picks["t"], lambda: stopped(ctx, c[picks["t"]], fit, test, "time", diag=True))
    run("dr_t", picks["t"], lambda: plain(ctx, c[picks["t"]], fit, test, trees=rec["er_t"]["best"]))
    run("d_t", picks["t"], lambda: plain(ctx, c[picks["t"]], fit, test, trees=rec["ef_t"]["best"]))
    run("d_t9", picks["t"], lambda: plain(ctx, c[picks["t"]], fit, test, trees=round(rec["ef_t"]["best"] / 0.9)))
    for refit, source in (("d", "ef"), ("dr", "er"), ("d_own", "d_own_es"), ("dr_t", "er_t"), ("d_t", "ef_t")):
        assert rec[refit].get("trees") == rec[source].get("best"), f"{refit} did not use the best iteration of {source}"
    return list(rec.values())


def run_unit(name, pop, seed, data, n_cand=N_CAND, cuts=(0, 1, 2)):
    """Every fit of one dataset, population and seed, as one long table."""
    X, y, t, cats = data
    g, lay = layout(t)
    check_geometry(t, g, lay)
    ctx = dict(X=X, y=y, cats=cats, seed=seed, cands=candidates(name, seed)[:n_cand], stop_start=g["stop_start"])
    schemes = ("e2", "e1") if pop == "thin" else ("e2",)
    pool = Parallel(n_jobs=THREADS, prefer="threads")
    tasks = [delayed(inner_task)(ctx, s, k, j, c, fit, val)
             for s in schemes for k in cuts for j, (fit, val) in enumerate(lay[(s, k)]["inner"])
             for c in range(n_cand)]
    inner = pd.DataFrame([r for rows in pool(tasks) for r in rows])
    finals = pool(delayed(final_task)(ctx, s, k, inner, *lay[(s, k)]["final"]) for s in schemes for k in cuts)
    df = pd.concat([inner, pd.DataFrame([r for rows in finals for r in rows])], ignore_index=True)
    return df.assign(dataset=name, pop=pop, seed=seed)


# ---------------------------------------------------------------- run and check

def run_log(pop):
    versions = {m.__name__: m.__version__ for m in (np, pd, lightgbm, scipy, sklearn, optuna)}
    files = {f: sha256(ROOT / f) for f in ("early_stopping_time_plan.md", "scripts/es_time.py",
                                           "scripts/es_time_prep.py", "results/es_time/prepared/_manifest.json")}
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "pop": pop, "threads": THREADS,
                            "versions": versions, "sha256": files}) + "\n")


def run(pop, only):
    (OUT / pop).mkdir(parents=True, exist_ok=True)
    run_log(pop)
    names = only or DATASETS[pop]
    units = [(seed, name) for seed in SEEDS for name in names]
    for i, (seed, name) in enumerate(units, 1):
        path = OUT / pop / f"{name}_seed{seed}.joblib"
        if path.exists():
            continue
        start = time.time()
        df = run_unit(name, pop, seed, load(name, pop))
        joblib.dump(df, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
        es = df[df.arm.isin(ES_ARMS)]
        capped = es.groupby("stage").best.apply(lambda b: (b > CAPPED).mean())
        print(f"[{pop} {i}/{len(units)}] {name} seed {seed}: wall {time.time() - start:.0f} s, thread s "
              f"{df.sec.sum():.0f}, fits {len(df)}, errors {(df.error != '').sum()}, capped ES fits "
              f"inner {capped.get('inner', np.nan):.2f} final {capped.get('final', np.nan):.2f}", flush=True)
    print("Done.")


def check():
    """Manifest hashes, geometry asserts and candidate bounds for every prepared file; one reduced unit."""
    for pop, names in DATASETS.items():
        for name in names:
            X, y, t, cats = load(name, pop)
            g, lay = layout(t)
            check_geometry(t, g, lay)
            print(f"{name} {pop}: {len(X)} rows, {X.shape[1]} features ({len(cats)} categorical), minority "
                  f"{y.mean():.3f}: sha256 and geometry asserts pass")
    for name in DATASETS["thin"]:
        for seed in SEEDS:
            candidates(name, seed)
    print("candidates: learning rate at least 0.01 for every dataset and seed")
    # The log-loss stop agrees with a manual argmin, and a reduced unit (4 candidates, cut 1) passes its asserts.
    data = load("nba", "thin")
    X, y, t, cats = data
    g, lay = layout(t)
    params = next(c for c in candidates("nba", 0) if c["class_weight"] is None)
    ctx = dict(X=X, y=y, cats=cats, seed=0, cands=[params], stop_start=g["stop_start"])
    (a, b), (ta, tb) = lay[("e2", 0)]["final"]
    r = stopped(ctx, params, (a, b), (ta, tb), "time")
    s = g["stop_start"](a, b)
    model = make({**params, "n_estimators": MAX_TREES}, 0).fit(X.iloc[a:s], y[a:s])
    ll = [log_loss(y[s:b], model.predict_proba(X.iloc[s:b], num_iteration=k)[:, 1], labels=[0, 1])
          for k in range(1, r["best"] + PATIENCE + 1)]
    assert int(np.argmin(ll)) + 1 == r["best"], "early stopping did not stop on log loss"
    print(f"early stop agrees with a manual log-loss argmin (best iteration {r['best']}, AP-best {r['best_ap_iter']})")
    df = run_unit("nba", "dense", 0, load("nba", "dense"), n_cand=4, cuts=(0,))
    assert (df.error == "").all() and df.ap.notna().all()
    print(f"reduced unit: {len(df)} fits, arms {sorted(df.arm.unique())}, no errors")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check()
    elif command == "run":
        run(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__)
