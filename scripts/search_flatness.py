"""search_flatness: is the hyperparameter search bigger than it needs to be? Plan: search_flatness_plan.md.

LightGBM over lgbm_space(), 60 random candidates per dataset and candidate seed, scored by
three arms in one runner: R60 (inner CV and an outer refit of every candidate; Rk is computed
from it), H60 (successive halving 60 -> 20 -> 7 on nested stratified subsets of each inner
split, round 3 read from R60) and ES (early stopping on a stratified 10 % carved inside each
fit, at most 500 trees). Every fit and predict_proba runs on one thread and is timed with
time.thread_time().

Usage: uv run python -m scripts.search_flatness check     pre-run checks (2 small datasets, no endpoints)
       uv run python -m scripts.search_flatness smoke     seconds per arm on 3 datasets, scores not read
       uv run python -m scripts.search_flatness run       both candidate seeds, resumable per dataset
       uv run python -m scripts.search_flatness analyse | tee results/search_flatness.txt   after both seeds
"""
import os
import sys
import time
import zlib
from math import comb

import joblib
import lightgbm
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t as t_dist, trim_mean, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from benchmark.checkpoints import atomic_dump, ckpt_path
from benchmark.data import datasets_to_run, load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models.build import build_lgbm
from benchmark.models.objectives import lgbm_objective, lgbm_space
from config import N_INNER_FOLDS, N_JOBS, N_OUTER_FOLDS, RANDOM_STATE, TRIVIAL_PR_AUC

OUT_DIR = "results/search_flatness"
ARMS = ("r60", "h60", "es")
SEEDS = (0, 1)
N_CAND = 60
ROUNDS = {1: 9, 2: 3}       # halving round -> subset denominator (1/9, then 1/3); round 3 is R60
KEEP = {1: 20, 2: 7}        # candidates kept after each round
MAX_TREES, PATIENCE, STOP_SHARE = 500, 20, 0.1
DELTA, LEVEL, N_BOOT = 0.0025, 0.975, 10_000
LARGE_ROWS, LARGEST_ROWS = 3200, 8000
CHECK_DATASETS = ("blogger", "robot-failure-lp4")
SMOKE_DATASETS = ("blogger", "mfeat-fourier", "letter")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------- population and candidates

def population() -> list[str]:
    """The 105 non-duplicate datasets where the stored lgbm_rerun scores below 0.99."""
    stored = joblib.load(ckpt_path("lgbm_rerun"))
    return [d for d in datasets_to_run() if np.mean(stored[d]["scores"]) < TRIVIAL_PR_AUC]


def candidates(dataset: str, seed: int) -> list[dict]:
    sampler = optuna.samplers.RandomSampler(seed=zlib.crc32(dataset.encode()) + seed)
    study = optuna.create_study(sampler=sampler)
    out = []
    for _ in range(N_CAND):
        trial = study.ask(lgbm_space())
        out.append(trial.params)
        study.tell(trial, 0.0)
    return out


# ---------------------------------------------------------------- splits and subsets

def outer_parts(X, y, fold):
    train, test = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[fold]
    return (X.iloc[train].reset_index(drop=True), y[train], X.iloc[test].reset_index(drop=True), y[test])


def inner_splits(X, y):
    return list(StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))


def subset(y, idx, den):
    """Positions of idx kept at fraction 1/den: max(1, n_c // den) rows of each class c, from one
    permutation per class drawn with RANDOM_STATE, so a larger fraction holds every smaller one.
    Original row order is kept."""
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        keep.append(rng.permutation(members)[:max(1, len(members) // den)])
    return np.sort(np.concatenate(keep))


def carve(y):
    """ES: stratified (keep, stop) positions, 90 / 10, original row order kept."""
    keep, stop = train_test_split(np.arange(len(y)), test_size=STOP_SHARE, stratify=y, random_state=RANDOM_STATE)
    return np.sort(keep), np.sort(stop)


# ---------------------------------------------------------------- fits

def roc_auc(y, p):
    """One-vs-rest ROC AUC, weighted over the classes present in y."""
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    present = np.unique(y)
    return float(sum((y == c).mean() * roc_auc_score(y == c, p[:, c]) for c in present))


def fit_plain(params, X, y, n_classes, cat_cols):
    return build_lgbm(params, n_classes, cat_cols, n_jobs=1).fit(X, y), params["n_estimators"]


def fit_es(params, X, y, n_classes, cat_cols):
    """At most MAX_TREES trees, stopped on log loss over a carved 10 %; the encoder sees the 90 %."""
    keep, stop = carve(y)
    model = build_lgbm({**params, "n_estimators": MAX_TREES}, n_classes, cat_cols, n_jobs=1)
    encoder, lgbm = model[0], model[-1]  # the ordinal encoder alone: build_lgbm adds no other step here
    X_keep = encoder.fit_transform(X.iloc[keep], y[keep])
    weights = {}
    if params["class_weight"] == "balanced":
        classes = np.arange(n_classes)
        weights["eval_class_weight"] = [dict(zip(classes, compute_class_weight("balanced", classes=classes,
                                                                                y=y[keep])))]
    lgbm.fit(X_keep, y[keep], eval_set=[(encoder.transform(X.iloc[stop]), y[stop])],
             callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)], **weights)
    return model, lgbm.best_iteration_


def fit_predict(params, X_fit, y_fit, X_test, n_classes, cat_cols, es=False):
    model, trees = (fit_es if es else fit_plain)(params, X_fit, y_fit, n_classes, cat_cols)
    return model.predict_proba(X_test), trees


def fit_score(params, X_fit, y_fit, X_test, y_test, n_classes, cat_cols, es=False, roc=True):
    assert np.unique(y_fit).size == n_classes, "a training part lacks a class"
    start = time.thread_time()
    try:
        p, trees = fit_predict(params, X_fit, y_fit, X_test, n_classes, cat_cols, es)
        sec = time.thread_time() - start
        return {"pr": pr_auc_score(y_test, p), "roc": roc_auc(y_test, p) if roc else np.nan, "sec": sec,
                "trees": trees, "constant": float(np.ptp(p, axis=0).max() == 0), "error": ""}
    except Exception as error:
        return {"pr": np.nan, "roc": np.nan, "sec": time.thread_time() - start, "trees": np.nan,
                "constant": np.nan, "error": repr(error)}


def means(frame, cands=range(N_CAND), value="pr"):
    """Mean over the inner splits per candidate; NaN if any split is NaN, as in cross_val_score."""
    return frame.pivot(index="cand", columns="split", values=value).reindex(cands).mean(1, skipna=False).values


def ranked(scores):
    """Candidate positions, best first; NaN last; ties by position."""
    s = np.where(np.isnan(scores), np.inf, -np.asarray(scores, dtype=float))
    return np.lexsort((np.arange(len(s)), s))


def halving(r1, r2, m):
    """Round-1 and round-2 survivors and the pick, all ties by candidate index. r2 and m are indexed
    by candidate."""
    top20 = np.sort(ranked(r1)[:KEEP[1]])
    top7 = np.sort(top20[ranked(r2[top20])[:KEEP[2]]])
    return top20, top7, top7[ranked(m[top7])[0]]


def s1_gap(o, s1, best):
    """Mean outer score of the S1 members other than the argmax that have one, minus the argmax's."""
    others = s1[(s1 != best) & ~np.isnan(o[s1])]
    return o[others].mean() - o[best] if len(others) else np.nan


# ---------------------------------------------------------------- runner

def run_dataset(dataset, seed, data):
    """Every fit of the three arms for one dataset and candidate seed, as one long table."""
    X, y, cat_cols = data
    n_classes = int(np.unique(y).size)
    cands = candidates(dataset, seed)
    folds = [outer_parts(X, y, f) for f in range(N_OUTER_FOLDS)]
    splits = [inner_splits(Xtr, ytr) for Xtr, ytr, _, _ in folds]
    es_arm = n_classes < 100

    def inner(arm, fold, cand, split, den=1, rnd=0):
        Xtr, ytr, _, _ = folds[fold]
        tr, te = splits[fold][split]
        if den > 1:
            tr, te = subset(ytr, tr, den), subset(ytr, te, den)
        r = fit_score(cands[cand], Xtr.iloc[tr], ytr[tr], Xtr.iloc[te], ytr[te], n_classes, cat_cols,
                      es=arm == "es", roc=den == 1)
        return {"arm": arm, "kind": "inner", "round": rnd, "fold": fold, "cand": cand, "split": split, **r}

    def outer(arm, kind, fold, cand):
        Xtr, ytr, Xte, yte = folds[fold]
        params = {**cands[cand], "n_estimators": MAX_TREES} if kind == "outer_500" else cands[cand]
        r = fit_score(params, Xtr, ytr, Xte, yte, n_classes, cat_cols, es=kind == "outer_es")
        return {"arm": arm, "kind": kind, "round": 0, "fold": fold, "cand": cand, "split": -1, **r}

    pool = Parallel(n_jobs=N_JOBS, prefer="threads")
    # Phase 1, arms interleaved: R60 inner and outer, ES inner, halving round 1.
    tasks = []
    for fold in range(N_OUTER_FOLDS):
        for cand in range(N_CAND):
            tasks.append(delayed(outer)("r60", "outer", fold, cand))
            for split in range(N_INNER_FOLDS):
                tasks.append(delayed(inner)("r60", fold, cand, split))
                tasks.append(delayed(inner)("h60", fold, cand, split, ROUNDS[1], 1))
                if es_arm:
                    tasks.append(delayed(inner)("es", fold, cand, split))
    rows = pool(tasks)
    df = pd.DataFrame(rows)
    # Phase 2: halving round 2, the ES pick's refits.
    tasks = []
    for fold in range(N_OUTER_FOLDS):
        for cand in ranked(means(df.query("arm == 'h60' and fold == @fold")))[:KEEP[1]]:
            for split in range(N_INNER_FOLDS):
                tasks.append(delayed(inner)("h60", fold, int(cand), split, ROUNDS[2], 2))
        if es_arm:
            pick = int(ranked(means(df.query("arm == 'es' and fold == @fold")))[0])
            tasks.append(delayed(outer)("es", "outer_es", fold, pick))
            tasks.append(delayed(outer)("es", "outer_500", fold, pick))
    df = pd.concat([df, pd.DataFrame(pool(tasks))], ignore_index=True)
    return df.assign(seed=seed, dataset=dataset)


def path(arm, seed):
    return os.path.join(OUT_DIR, f"{arm}_seed{seed}.joblib")


def run(datasets):
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in SEEDS:
        stores = {arm: joblib.load(path(arm, seed)) if os.path.exists(path(arm, seed)) else {} for arm in ARMS}
        for i, dataset in enumerate(datasets, 1):
            if all(dataset in stores[arm] for arm in ARMS):
                continue
            start = time.time()
            df = run_dataset(dataset, seed, load_data_df(dataset))
            for arm in ARMS:
                stores[arm][dataset] = df[df.arm == arm].reset_index(drop=True)
                atomic_dump(stores[arm], path(arm, seed))
            print(f"[seed {seed} {i}/{len(datasets)}] {dataset}: wall {time.time() - start:.0f} s, "
                  f"errors {(df.error != '').sum()}", flush=True)
    print("Done.")


# ---------------------------------------------------------------- checks and smoke test

def check():
    """The plan's pre-run checks. Prints differences only, never an arm comparison."""
    stored = joblib.load(ckpt_path("lgbm_rerun"))
    for dataset in CHECK_DATASETS:
        X, y, cat_cols = load_data_df(dataset)
        n_classes = int(np.unique(y).size)
        worst_obj = worst_frac = worst_pred = 0.0
        for fold in range(N_OUTER_FOLDS):
            Xtr, ytr, Xte, _ = outer_parts(X, y, fold)
            splits = inner_splits(Xtr, ytr)
            for params in candidates(dataset, 0):
                per_split = [fit_score(params, Xtr.iloc[tr], ytr[tr], Xtr.iloc[te], ytr[te], n_classes, cat_cols)["pr"]
                             for tr, te in splits]
                objective = lgbm_objective(optuna.trial.FixedTrial(params), Xtr, ytr,
                                           StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                                           n_classes, cat_cols)
                worst_obj = max(worst_obj, abs(np.mean(per_split) - objective))
                frac = []
                for tr, te in splits:
                    tr, te = subset(ytr, tr, 1), subset(ytr, te, 1)
                    frac.append(fit_score(params, Xtr.iloc[tr], ytr[tr], Xtr.iloc[te], ytr[te], n_classes, cat_cols)["pr"])
                worst_frac = max(worst_frac, np.max(np.abs(np.subtract(frac, per_split))))
            p, _ = fit_predict(stored[dataset]["best_params"][fold], Xtr, ytr, Xte, n_classes, cat_cols)
            worst_pred = max(worst_pred, np.max(np.abs(p - stored[dataset]["preds"][fold])))
        print(f"{dataset}: R60 inner mean vs objective {worst_obj:.1e} (< 1e-6), subsets at fraction 1 vs R60 "
              f"{worst_frac:.1e}, outer path vs stored lgbm_rerun preds {worst_pred:.1e} (atol 1e-12)")
    # ES carves every fitting set; the plan excludes the two 100-class datasets for this.
    failed = []
    for dataset in population():
        X, y, _ = load_data_df(dataset)
        if np.unique(y).size >= 100:
            continue
        for fold in range(N_OUTER_FOLDS):
            Xtr, ytr, _, _ = outer_parts(X, y, fold)
            for part in [ytr] + [ytr[tr] for tr, _ in inner_splits(Xtr, ytr)]:
                try:
                    keep, _ = carve(part)
                    assert np.unique(part[keep]).size == np.unique(y).size
                except (ValueError, AssertionError) as error:
                    failed.append((dataset, fold, repr(error)))
    print(f"ES carves failing or missing a class: {failed or 'none'}")


def smoke():
    """Seconds per arm on SMOKE_DATASETS, seed 0; nothing is saved and no score is printed."""
    stored = joblib.load(ckpt_path("lgbm_rerun"))
    total = sum(stored[d]["time"] for d in population())
    walls, threads, rerun = 0.0, pd.Series(0.0, index=list(ARMS)), 0.0
    for dataset in SMOKE_DATASETS:
        start = time.time()
        df = run_dataset(dataset, 0, load_data_df(dataset))
        wall = time.time() - start
        secs = df.groupby("arm").sec.sum().reindex(list(ARMS), fill_value=0)
        print(f"{dataset}: wall {wall:.0f} s, thread s " + ", ".join(f"{a} {v:.0f}" for a, v in secs.items())
              + f"; lgbm_rerun wall {stored[dataset]['time']:.0f} s", flush=True)
        walls, threads, rerun = walls + wall, threads + secs, rerun + stored[dataset]["time"]
    hours = walls / rerun * total / 3600
    print(f"Estimate per seed: {hours:.1f} h (smoke wall / lgbm_rerun wall on these datasets x lgbm_rerun's "
          f"{total / 3600:.2f} h over the population); by thread share " +
          ", ".join(f"{a} {hours * v / threads.sum():.1f} h" for a, v in threads.items()))


# ---------------------------------------------------------------- analysis

def load_fits():
    return pd.concat([pd.concat(joblib.load(path(arm, s)).values()) for arm in ARMS for s in SEEDS],
                     ignore_index=True)


def rk_weights(k):
    """P(the j-th ranked of N_CAND wins a uniformly random k-subset), j = 1..N_CAND."""
    return np.array([comb(N_CAND - j, k - 1) / comb(N_CAND, k) for j in range(1, N_CAND + 1)])


def cells(fits, cand_params):
    """One row per seed, dataset and outer fold with every endpoint; and the candidate table."""
    rows, cand_rows = [], []
    probs = {k: rk_weights(k) for k in (10, 20)}
    for (seed, dataset, fold), g in fits.groupby(["seed", "dataset", "fold"]):
        r = g[g.arm == "r60"]
        ri = r[r.kind == "inner"]
        inner_pr = ri.pivot(index="cand", columns="split", values="pr").reindex(range(N_CAND))
        outer = r[r.kind == "outer"].set_index("cand").reindex(range(N_CAND))
        m, sd = means(ri), inner_pr.std(1, ddof=0, skipna=False).values
        o, o_roc = outer.pr.values, outer.roc.values
        order = ranked(m)
        best = order[0]
        row = {"seed": seed, "dataset": dataset, "fold": fold, "r60": o[best],
               "r60_inner_s": r[r.kind == "inner"].sec.sum()}
        for k, p in probs.items():
            w = p > 0  # a failed candidate below the k-1 last ranks has weight 0, not NaN
            row[f"r{k}"] = float(p[w] @ o[order][w])
        # H1 and its secondaries.
        s1 = np.flatnonzero(m >= m[best] - sd[best])
        row["s1_share"] = len(s1) / N_CAND
        row["h1"] = s1_gap(o, s1, best)
        row["s1_worst"] = np.nanmin(o[s1]) - o[best]
        row["all60"] = np.nanmean(o) - o[best]
        params = cand_params[(seed, dataset)]
        lr = np.array([c["learning_rate"] for c in params])
        leaves = np.array([c["num_leaves"] for c in params])
        row.update(s1_lr_min=lr[s1].min(), s1_lr_max=lr[s1].max(), s1_leaves_min=leaves[s1].min(),
                   s1_leaves_max=leaves[s1].max())
        sub = np.flatnonzero(lr >= 0.01)
        b = sub[ranked(m[sub])[0]]
        s = sub[m[sub] >= m[b] - sd[b]]
        row["h1_lr"] = s1_gap(o, s, b)
        mr = means(ri, value="roc")
        sdr = ri.pivot(index="cand", columns="split", values="roc").reindex(range(N_CAND)).std(1, ddof=0, skipna=False).values
        b = ranked(mr)[0]
        s = np.flatnonzero(mr >= mr[b] - sdr[b])
        row["h1_roc"] = s1_gap(o_roc, s, b)
        # H60.
        h = g[g.arm == "h60"]
        r1, r2 = means(h[h["round"] == 1]), means(h[h["round"] == 2])
        top20, top7, h_pick = halving(r1, r2, m)
        row["h60"] = o[h_pick]
        row["h60_rank_inner"] = int(np.flatnonzero(order == h_pick)[0]) + 1
        row["h60_rank_outer"] = float(pd.Series(o).rank(ascending=False)[h_pick])
        row["h60_s"] = h.sec.sum() + r[(r.kind == "inner") & r.cand.isin(top7)].sec.sum()
        r1_sorted, r2_sorted = r1[ranked(r1)], r2[top20][ranked(r2[top20])]
        row["tie_cut1"] = float(r1_sorted[KEEP[1] - 1] == r1_sorted[KEEP[1]])
        row["tie_cut2"] = float(r2_sorted[KEEP[2] - 1] == r2_sorted[KEEP[2]])
        row["r1_constant"] = h[h["round"] == 1].constant.mean()
        # ES.
        e = g[g.arm == "es"]
        if len(e):
            ei = e[e.kind == "inner"]
            row["es"] = e[e.kind == "outer_es"].pr.iloc[0]
            row["es_500"] = e[e.kind == "outer_500"].pr.iloc[0]
            row["es_inner_s"] = ei.sec.sum()
            row["es_trees_share"] = (ei.trees / MAX_TREES).mean()
            # Stopping never triggered: the best iteration is within PATIENCE of the cap.
            row["es_all_trees"] = (ei.trees > MAX_TREES - PATIENCE).mean()
        # Differences per cell, so a failed endpoint drops the cell from that difference only.
        row.update(d_r20=row["r20"] - row["r60"], d_r10=row["r10"] - row["r60"], d_h60=row["h60"] - row["r60"],
                   d_es=row.get("es", np.nan) - row["r60"], d_es_500=row.get("es", np.nan) - row.get("es_500", np.nan))
        rows.append(row)
        es_inner = means(e[e.kind == "inner"]) if len(e) else np.nan
        cand_rows.append(pd.DataFrame({
            "seed": seed, "dataset": dataset, "fold": fold, "cand": range(N_CAND),
            **{f"inner{s}": inner_pr[s].values for s in range(N_INNER_FOLDS)},
            "inner_mean": m, "inner_sd": sd, "inner_roc_mean": mr, "outer": o, "outer_roc": o_roc,
            "h60_round1": r1, "es_inner_mean": es_inner,
            **{k: [c[k] for c in params] for k in params[0]}}))
    return pd.DataFrame(rows), pd.concat(cand_rows, ignore_index=True)


def interval(diff, level=LEVEL, families=None):
    """Percentile bootstrap of the mean, RNG seed 0; by family when given."""
    rng = np.random.default_rng(0)
    alpha = (1 - level) / 2 * 100
    if families is None:
        boot = diff.values[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(1)
    else:
        grouped = diff.groupby(families)
        sums, counts = grouped.sum().values, grouped.size().values
        idx = rng.integers(0, len(sums), (N_BOOT, len(sums)))
        boot = sums[idx].sum(1) / counts[idx].sum(1)
    return tuple(np.percentile(boot, [alpha, 100 - alpha]))


def verdict(lo, hi, delta=DELTA):
    if lo > -delta:
        return "supported, worse within the margin" if hi < 0 else "supported"
    if hi < -delta:
        return "refuted"
    return "inconclusive"


def house(label, diff):
    """Mean with a 97.5 % percentile interval, median, trimmed mean, ahead / behind, Wilcoxon p."""
    diff = diff.dropna()
    nonzero = diff[diff != 0]
    p = wilcoxon(nonzero).pvalue if len(nonzero) > 1 else np.nan
    lo, hi = interval(diff)
    print(f"  {label:44s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")
    return lo, hi


def hypothesis(label, diff):
    """The decision rule, its checks and its sensitivities, for one per-dataset difference."""
    diff = diff.dropna()
    lo, hi = house(label, diff)
    half = t_dist.ppf(1 - (1 - LEVEL) / 2, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    t_verdict = verdict(diff.mean() - half, diff.mean() + half)
    decided = verdict(lo, hi)
    final = decided if t_verdict.split(",")[0] == decided.split(",")[0] else "inconclusive (t-interval disagrees)"
    print(f"    verdict: {final}; t-interval [{diff.mean() - half:+.4f}, {diff.mean() + half:+.4f}] -> {t_verdict}; "
          f"|difference| < 1e-4 on {(diff.abs() < 1e-4).sum()}")
    print(f"    delta 0.005: {verdict(lo, hi, 0.005)}; by family ({diff.index.str.split('-').str[0].nunique()}): "
          f"{verdict(*interval(diff, families=diff.index.str.split('-').str[0]))}")
    return final


def analyse():
    fits = load_fits()
    pop = population()
    meta = pd.read_csv("results/datasets.csv").set_index("dataset").loc[pop]
    cand_params = {(s, d): candidates(d, s) for s in SEEDS for d in pop}
    print("Before analysis:")
    expected = {"r60": N_CAND * (N_INNER_FOLDS + 1), "h60": N_CAND * N_INNER_FOLDS + KEEP[1] * N_INNER_FOLDS,
                "es": N_CAND * N_INNER_FOLDS + 2}
    counts = fits.groupby(["arm", "seed", "dataset", "fold"]).size()
    short = [(k, n) for k, n in counts.items() if n != expected[k[0]]]
    es_pop = list(meta.index[meta.n_classes < 100])
    print(f"  datasets per arm and seed: {fits.groupby(['arm', 'seed']).dataset.nunique().to_dict()} "
          f"(r60, h60: {len(pop)}; es: {len(es_pop)}); folds with a wrong row count: {short or 'none'}")
    failed = fits[(fits.error != "") | fits.pr.isna()]
    print(f"  failed fits or NaN scores: {len(failed)}")
    for r in failed.itertuples():
        print(f"    {r.arm} seed {r.seed} {r.dataset} fold {r.fold} cand {r.cand} {r.kind} split {r.split}: {r.error}")

    cell, cand = cells(fits, cand_params)
    cand.round(6).to_csv("results/search_flatness_candidates.csv", index=False)
    cell.round(6).to_csv("results/search_flatness_picks.csv", index=False)
    per = cell.groupby("dataset").mean(numeric_only=True)
    large = list(meta.index[meta.n_rows >= LARGE_ROWS])
    largest = list(meta.index[meta.n_rows >= LARGEST_ROWS])
    rare = list(meta.loc[es_pop].index[meta.loc[es_pop].eval("n_rows * minority_share") * 9 / 16 * STOP_SHARE < 2])
    print(f"  populations: all {len(pop)}, large {len(large)}, 8,000+ rows {len(largest)}, ES {len(es_pop)}, "
          f"rare class in the stopping set {len(rare)}")

    print("  cells without a value, per endpoint (dropped from it): "
          + ", ".join(f"{c} {n}" for c, n in cell[["r60", "r20", "r10", "h60", "es", "es_500", "h1"]].isna().sum().items()))

    print("\nPrimary (outer PR AUC, mean over fold x seed cells per dataset; 97.5 % intervals, delta 0.0025):")
    h1 = per.h1.reindex(pop)
    dropped = int(h1.isna().sum())
    print(f" H1 (C1a): {dropped} of {len(pop)} datasets have no cell (|S1| = 1 on every fold)")
    h1_verdict = hypothesis("H1, mean S1 without argmax - argmax", h1)
    if dropped > 0.1 * len(pop):
        h1_verdict = "inconclusive (more than 10 % of datasets dropped)"
    share = per.s1_share.reindex(pop)
    rng = np.random.default_rng(0)
    medians = np.median(share.values[rng.integers(0, len(share), (N_BOOT, len(share)))], axis=1)
    lo, hi = np.percentile(medians, [1.25, 98.75])
    premise = share.median() * N_CAND >= 20
    print(f"    median share |S1| / 60: {share.median() * N_CAND:.1f} of 60 [{lo * N_CAND:.1f}, {hi * N_CAND:.1f}]"
          f" (article 39); premise {'holds' if premise else 'fails'}")
    filled = cell.assign(h1=cell.h1.fillna(0)).groupby("dataset").h1.mean()
    print(f"    sensitivity, verdict {verdict(*house('H1, folds with |S1| = 1 scored 0', filled.reindex(pop)))}")
    print(f" C1a: {h1_verdict if premise else 'premise fails'}")
    print(f" H2 (C1b): C1b {hypothesis('H2, R20 - R60 pick', per.d_r20.reindex(pop))}")
    print(f" H3 (C2): C2 {hypothesis('H3, H60 pick - R60 pick (large 25)', per.d_h60.reindex(large))}")
    ratio = (per.h60_s / per.r60_inner_s).reindex(large)
    print(f"    cost, H60 / R60 summed inner thread s: median {ratio.median():.2f} over {len(large)} (article 0.60)")
    print(f" H4 (C3): C3 {hypothesis('H4, ES pick - R60 pick', per.d_es.reindex(es_pop))}")
    print("\nEach candidate seed alone:")
    for s in SEEDS:
        ps = cell[cell.seed == s].groupby("dataset").mean(numeric_only=True)
        for label, d in [("H1", ps.h1.reindex(pop)), ("H2", ps.d_r20.reindex(pop)),
                         ("H3", ps.d_h60.reindex(large)), ("H4", ps.d_es.reindex(es_pop))]:
            d = d.dropna()
            lo, hi = interval(d)
            print(f"  seed {s} {label}: mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict(lo, hi)}")

    print("\nSecondary, descriptive, raw p:")
    print(" C1a")
    tercile = pd.qcut(meta.n_rows.rank(method="first"), 3, labels=["small", "mid", "large"])
    for name, groups in [("row tercile", tercile), ("classes", np.where(meta.n_classes == 2, "binary", "multiclass"))]:
        shares = share.groupby(pd.Series(groups, index=meta.index), observed=True).median() * N_CAND
        print(f"  median |S1| of 60 by {name}: " + ", ".join(f"{k} {v:.1f}" for k, v in shares.items()))
    print(f"  within S1, median over cells: learning rate {cell.s1_lr_min.median():.4f} to {cell.s1_lr_max.median():.4f}, "
          f"num_leaves {cell.s1_leaves_min.median():.0f} to {cell.s1_leaves_max.median():.0f}")
    house("worst S1 member - argmax", per.s1_worst.reindex(pop))
    house("mean of all 60 - argmax", per.all60.reindex(pop))
    print(f"  H1 as a fraction of mean of all 60 - argmax: {h1.mean() / per.all60.reindex(h1.dropna().index).mean():.2f}")
    house("H1, candidates with learning rate >= 0.01", per.h1_lr.reindex(pop))
    house("H1 on ROC AUC", per.h1_roc.reindex(pop))
    print(" C1b")
    house("R10 - R60 pick", per.d_r10.reindex(pop))
    print(" C2")
    print(f"  rank of the H60 pick within R60 (large 25): inner median {per.h60_rank_inner.reindex(large).median():.1f}, "
          f"outer median {per.h60_rank_outer.reindex(large).median():.1f} (article: 4th)")
    print(f"  round-1 fits with constant predictions: {per.r1_constant.reindex(large).mean():.1%} (large 25), "
          f"{per.r1_constant.reindex(pop).mean():.1%} (all); cells with a tie at the cut to 20: "
          f"{cell.tie_cut1.mean():.1%} (large 25: {cell[cell.dataset.isin(large)].tie_cut1.mean():.1%}), to 7: "
          f"{cell.tie_cut2.mean():.1%} (large 25: {cell[cell.dataset.isin(large)].tie_cut2.mean():.1%})")
    house(f"H3 on the {len(largest)} with 8,000+ rows", per.d_h60.reindex(largest))
    house(f"H3 on all {len(pop)}", per.d_h60.reindex(pop))
    print(" C3")
    house("ES pick - same configuration at 500 trees", per.d_es_500.reindex(es_pop))
    es_tercile = pd.qcut(meta.loc[es_pop].n_rows.rank(method="first"), 3, labels=["small", "mid", "large"])
    for t in ("small", "mid", "large"):
        house(f"H4, row tercile {t}", per.d_es.reindex(list(es_tercile.index[es_tercile == t])))
    house(f"H4 on the large {len(large)}", per.d_es.reindex([d for d in large if d in es_pop]))
    house(f"H4 on the {len(rare)} with a rare class in the stopping set", per.d_es.reindex(rare))
    es_cost = (per.es_inner_s / per.r60_inner_s).reindex(es_pop)
    print(f"  ES / R60 summed inner thread s: median {es_cost.median():.2f}; stopped trees / 500 per fit: mean "
          f"{per.es_trees_share.reindex(es_pop).mean():.2f}; fits reaching the 500-tree cap without stopping: {per.es_all_trees.reindex(es_pop).mean():.1%}")
    print(" H1 and H2 on the large 25")
    house("H1", per.h1.reindex(large))
    house("H2", per.d_r20.reindex(large))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check()
    elif command == "smoke":
        smoke()
    elif command == "run":
        run(population())
    elif command == "analyse":
        analyse()
    else:
        print(__doc__)
