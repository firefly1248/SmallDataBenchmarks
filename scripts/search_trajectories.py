"""Search trajectories on the 30 datasets of scripts/split_seeds.py, from the Optuna studies of
the 100-trial runs kept under --run traj (KEEP_STUDIES=1): TPE and ConfOpt at sampler seed 0
(block F) and ConfOpt with the DtACI fix at seeds 0 and 1 (block G).

The searches are deterministic, so each study's first 50 trials are the 50-trial run's; the
picks after 50 and 100 trials are checked against the stored checkpoints. Per (dataset, fold,
trial) the objective value and the best value so far go to results/search_trajectories.csv.
The pick after 10, 33, 50 and 100 trials is re-scored on a fresh inner split (seed
RANDOM_STATE + 1) and refitted on the outer fold. Not pre-registered.

Usage: uv run python -m scripts.search_trajectories
"""
import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import trim_mean, wilcoxon
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models.registry import MODELS
from benchmark.nested_cv import _storage
from benchmark.results import CALIBRATION_POPULATION, per_dataset
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/search_trajectories.csv"
RUN = "traj"
MARKS = (10, 33, 50, 100)
# Model under --run traj: (sampler label, its 50-trial checkpoint, its 100-trial checkpoint).
MODELS_TRAJ = {"lgbm_trials100": ("TPE", "lgbm_rerun", "lgbm_trials100"),
               "lgbm_confopt_trials100": ("ConfOpt", "lgbm_confopt", "lgbm_confopt_trials100"),
               "lgbm_confopt_dtaci_trials100": ("ConfOpt DtACI", None, None),
               "lgbm_confopt_dtaci_trials100_seed1": ("ConfOpt DtACI seed 1", None, None)}
calibration = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
rng = np.random.default_rng(0)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def gap(label, diff):
    """As in scripts/confopt_controls.py."""
    diff = diff.dropna()
    p = wilcoxon(diff[diff != 0]).pvalue if (diff != 0).any() else 1.0
    boot = diff.values[rng.integers(0, len(diff), (10_000, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:44s} n {len(diff)}, mean {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {diff.median():+.4f}, "
          f"trimmed {trim_mean(diff, 0.1):+.4f}, ahead {(diff > 0).sum()} / behind {(diff < 0).sum()}, p {p:.3g}")


traj, picks = [], {}
for model in MODELS_TRAJ:
    storage = _storage(model, RUN)
    for name in optuna.study.get_all_study_names(storage):
        dataset, fold = name.rsplit("/fold", 1)
        trials = sorted((t for t in optuna.load_study(study_name=name, storage=storage).trials
                         if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: t.number)
        best = -np.inf
        for t in trials:
            best = max(best, t.value)
            traj.append({"model": model, "dataset": dataset, "fold": int(fold), "trial": t.number + 1,
                         "value": t.value, "best": best})
        for k in MARKS:
            winner = max((t for t in trials if t.number < k), key=lambda t: (t.value, -t.number))
            picks[(model, dataset, int(fold), k)] = (winner.params, winner.number + 1)
traj = pd.DataFrame(traj)
traj.round(6).to_csv(OUTPUT, index=False)
print(f"{len(traj)} trials over {traj.groupby(['model', 'dataset', 'fold']).ngroups} studies")

print("\nPicks against the stored checkpoints (default run):")
for model, (label, c50, c100) in MODELS_TRAJ.items():
    stored = {k: joblib.load(ckpt_path(c)) for k, c in ((50, c50), (100, c100)) if c}
    stored[100] = stored.get(100) or joblib.load(ckpt_path(model, RUN))
    for k, ckpt in stored.items():
        keys = [key for key in picks if key[0] == model and key[3] == k]
        same = sum(picks[key][0] == ckpt[key[1]]["best_params"][key[2]] for key in keys)
        print(f"  {label}, {k} trials: {same} of {len(keys)} folds pick the stored configuration")

print("\nMean best inner score after k trials (dataset mean over folds, then over datasets):")
for model, (label, *_) in MODELS_TRAJ.items():
    t = traj[traj.model == model]
    print(f"  {label:22s} " + ", ".join(
        f"{k}: {t[t.trial == k].groupby('dataset').best.mean().mean():.4f}" for k in MARKS))
print("Trial that found the 100-trial winner:")
for model, (label, *_) in MODELS_TRAJ.items():
    n = pd.Series([v[1] for key, v in picks.items() if key[0] == model and key[3] == 100])
    print(f"  {label:22s} median {n.median():.0f}, in trials 1-10 {np.mean(n <= 10):.0%}, 11-33 "
          f"{np.mean((n > 10) & (n <= 33)):.0%}, 34-50 {np.mean((n > 33) & (n <= 50)):.0%}, 51-100 {np.mean(n > 50):.0%}")

data = {d: load_data_df(d) for d in {key[1] for key in picks}}


def rescore(model, dataset, fold, params):
    X, y, cat_cols = data[dataset]
    train, test = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[fold]
    X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
    n_classes = int(np.unique(y).size)
    fresh = MODELS[model].objective(optuna.trial.FixedTrial(params), X_train, y_train,
                                    StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1),
                                    n_classes, cat_cols)
    est = MODELS[model].build(params, n_classes, cat_cols).fit(X_train, y_train)
    pred = est.predict_proba(X.iloc[test].reset_index(drop=True))
    return fresh, pr_auc_score(y[test], pred), pred, y[test]


unique = {}
for key, (params, _) in picks.items():
    unique.setdefault((key[0], key[1], key[2], repr(sorted(params.items()))), params)
done = dict(zip(unique, Parallel(n_jobs=N_OUTER_FOLDS, prefer="threads")(
    delayed(rescore)(m, d, f, p) for (m, d, f, _), p in unique.items())))
scored = {key: done[(key[0], key[1], key[2], repr(sorted(p.items())))] for key, (p, _) in picks.items()}

print("\nPick after k trials: best inner score, fresh inner split, outer fold (dataset means):")
for model, (label, *_) in MODELS_TRAJ.items():
    for k in MARKS:
        keys = [key for key in scored if key[0] == model and key[3] == k]
        frame = pd.DataFrame([{"dataset": key[1], "search": traj.loc[(traj.model == model) & (traj.dataset == key[1])
                                                                     & (traj.fold == key[2]) & (traj.trial == k), "best"].iloc[0],
                               "fresh": scored[key][0], "outer": scored[key][1]} for key in keys])
        m = frame.groupby("dataset").mean()
        print(f"  {label:22s} {k:3d}: search {m.search.mean():.4f}, fresh {m.fresh.mean():.4f}, outer {m.outer.mean():.4f}")


def entries(model, k):
    """The refitted picks after k trials as checkpoint entries, for per_dataset."""
    out = {}
    for key in sorted(k_ for k_ in scored if k_[0] == model and k_[3] == k):
        e = out.setdefault(key[1], {"scores": [], "preds": [], "labels": [], "time": np.nan})
        e["scores"].append(scored[key][1])
        e["preds"].append(scored[key][2])
        e["labels"].append(scored[key][3])
    return out


print("\nG. ConfOpt with the DtACI fix against the published ConfOpt and TPE, 30 datasets:")
REF = {(0, 50): ("lgbm_confopt", "lgbm_rerun"), (0, 100): ("lgbm_confopt_trials100", "lgbm_trials100"),
       (1, 50): ("lgbm_confopt_seed1", "lgbm_seed1"), (1, 100): ("lgbm_confopt_trials100_seed1", "lgbm_trials100_seed1")}
for (seed, k), (conf, tpe) in REF.items():
    model = "lgbm_confopt_dtaci_trials100" + ("_seed1" if seed else "")
    fixed = per_dataset(entries(model, k), calibration)
    ds = list(fixed.index)
    for ref_label, ref in (("ConfOpt", conf), ("TPE", tpe)):
        other = per_dataset({d: e for d, e in joblib.load(ckpt_path(ref)).items() if d in ds}, calibration)
        gap(f"seed {seed}, {k} trials, PR AUC: fixed - {ref_label}", (fixed.prauc - other.prauc).astype(float))
        cal = fixed.ece.dropna().index
        gap(f"seed {seed}, {k} trials, ECE: fixed - {ref_label}", (fixed.ece[cal] - other.ece[cal]).astype(float))
for seed in (0, 1):
    model = "lgbm_confopt_dtaci_trials100" + ("_seed1" if seed else "")
    fixed = joblib.load(ckpt_path(model, RUN))
    ref = joblib.load(ckpt_path("lgbm_confopt_trials100" + ("_seed1" if seed else "")))
    print(f"  seed {seed}, 100 trials, hours over the 30 datasets: fixed {sum(e['time'] for e in fixed.values()) / 3600:.2f}, "
          f"published {sum(ref[d]['time'] for d in fixed) / 3600:.2f}")
print(f"Saved {OUTPUT}")
