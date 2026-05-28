"""
Benchmark: SVC, LogReg, TabPFN, TabICL, RandomForest, XGBoost, SGD, CatBoost,
LightGBM, LightGBM-linear, HistGradientBoosting, TabNet, FT-Transformer, ResNet.

Hyperparameter tuning strategy:
- SVC, LogReg, TabPFN, TabICL : GridSearchCV (small, well-understood HP space)
- All others                  : Optuna TPE (N_TRIALS / N_TRIALS_NN trials per outer fold)

Categorical handling:
- CatBoost, TabPFN, TabICL      : native categorical handling
- RF, XGBoost, LGBM, HGB        : CatFeaturesEncoder(ordinal) (NaN handled natively)
- SVC, LogReg, SGD              : CatFeaturesEncoder(target/...) + Imputer + Scaler
- TabNet, FT-Transformer, ResNet: ordinal-encode + impute + StandardScaler in wrapper
"""

import torch  # must precede xgboost / lightgbm / catboost to win the OpenMP init race
import argparse
import signal
import time
import joblib
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np

DATASET_TIMEOUT = 43_200  # 12 hours — above longest legit run (pendigits ~11h), below hangs (18h+)


class _DatasetTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _DatasetTimeout()

from benchmark.data import load_data_df
from benchmark.nested_cv import run_nested_cv
from config import N_JOBS, RANDOM_STATE, N_OUTER_FOLDS, MAX_DATASET_ROWS, MODELS_TO_RUN

warnings.filterwarnings("ignore")


def _timed(fn, *args):
    """Call fn(*args) and return (result, elapsed_seconds)."""
    t = time.time()
    return fn(*args), time.time() - t


CHECKPOINT   = "results/optuna_models_ckpt.joblib"
FINAL_OUTPUT = "results/optuna_models.joblib"

if __name__ == "__main__":
    ALL_MODELS = ["svc", "logreg", "tabpfn", "tabicl", "random_forest", "xgboost", "sgd",
                  "catboost", "lgbm", "lgbm_linear", "hgb",
                  "tabnet", "ft_transformer", "resnet"]

    parser = argparse.ArgumentParser(description="Optuna nested CV benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", choices=ALL_MODELS, metavar="MODEL",
                       help="Run only these models (default: all)")
    group.add_argument("--skip", nargs="+", choices=ALL_MODELS, metavar="MODEL",
                       help="Run all models except these")
    args = parser.parse_args()

    if args.models:
        MODEL_NAMES = [m for m in ALL_MODELS if m in args.models]
    elif args.skip:
        MODEL_NAMES = [m for m in ALL_MODELS if m not in args.skip]
    elif MODELS_TO_RUN != "all":
        MODEL_NAMES = [m for m in ALL_MODELS if m in MODELS_TO_RUN]
    else:
        MODEL_NAMES = ALL_MODELS

    print(f"Models to run: {MODEL_NAMES}")

    # SVC uses only 1 core internally — run it in a background thread so it
    # overlaps with the other models instead of blocking them.
    PARALLEL_MODELS = {"svc"} & set(MODEL_NAMES)
    SEQUENTIAL_MODELS = [m for m in MODEL_NAMES if m not in PARALLEL_MODELS]

    _, _, _, evaluated_datasets, _ = joblib.load("results/compare_baseline_models.joblib")

    # Granular checkpoint: {dataset_name: {model_name: {"scores": [...], "time": float}}}
    # done_set: set of (dataset_name, model_name) pairs
    ckpt_results: dict = {}
    done_set: set = set()
    try:
        ckpt_results, done_set = joblib.load(CHECKPOINT)
        n_models_done = len(done_set)
        n_datasets_done = len({ds for ds, _ in done_set})
        print(f"Resuming: {n_models_done} (dataset, model) pairs done "
              f"across {n_datasets_done} datasets.")
    except FileNotFoundError:
        pass

    def save_checkpoint():
        joblib.dump((ckpt_results, done_set), CHECKPOINT)

    for i, dataset_name in enumerate(evaluated_datasets):
        models_to_run_seq = [m for m in SEQUENTIAL_MODELS if (dataset_name, m) not in done_set]
        models_to_run_par = [m for m in PARALLEL_MODELS  if (dataset_name, m) not in done_set]
        if not models_to_run_seq and not models_to_run_par:
            print(f"[{i+1}/{len(evaluated_datasets)}] {dataset_name}  — skipping (all models done)")
            continue

        X, y, cat_cols = load_data_df(dataset_name)
        if len(y) == 0:
            continue

        rng = np.random.default_rng(RANDOM_STATE)
        if len(X) > MAX_DATASET_ROWS:
            random_idx = rng.choice(len(X), MAX_DATASET_ROWS, replace=False)
            X = X.iloc[random_idx].reset_index(drop=True)
            y = y[random_idx]

        print(f"\n[{i+1}/{len(evaluated_datasets)}] {dataset_name}  "
              f"shape={X.shape}  cat_cols={len(cat_cols)}")

        if dataset_name not in ckpt_results:
            ckpt_results[dataset_name] = {}

        executor = (ThreadPoolExecutor(max_workers=len(models_to_run_par))
                    if models_to_run_par else None)
        try:
            futures = {
                m: executor.submit(_timed, run_nested_cv, X, y, m, cat_cols)
                for m in models_to_run_par
            } if executor else {}

            for model_name in models_to_run_seq:
                start = time.time()
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(DATASET_TIMEOUT)
                try:
                    scores, preds, labels, best_params = run_nested_cv(X, y, model_name, cat_cols)
                except _DatasetTimeout:
                    print(f"  {model_name}: TIMEOUT after {DATASET_TIMEOUT // 3600}h")
                    scores = [np.nan] * N_OUTER_FOLDS
                    preds = labels = best_params = None
                except Exception as e:
                    print(f"  {model_name}: ERROR — {e}")
                    scores = [np.nan] * N_OUTER_FOLDS
                    preds = labels = best_params = None
                finally:
                    signal.alarm(0)
                elapsed = time.time() - start
                ckpt_results[dataset_name][model_name] = {
                    "scores": scores, "preds": preds, "labels": labels,
                    "best_params": best_params, "time": elapsed,
                }
                done_set.add((dataset_name, model_name))
                print(f"  {model_name}: mean={np.nanmean(scores):.4f}  time={elapsed:.1f}s")
                save_checkpoint()

            for model_name, future in futures.items():
                try:
                    (scores, preds, labels, best_params), elapsed = future.result(timeout=DATASET_TIMEOUT)
                except Exception as e:
                    print(f"  {model_name}: ERROR — {e}")
                    scores = [np.nan] * N_OUTER_FOLDS
                    preds = labels = best_params = None
                    elapsed = 0.0
                ckpt_results[dataset_name][model_name] = {
                    "scores": scores, "preds": preds, "labels": labels,
                    "best_params": best_params, "time": elapsed,
                }
                done_set.add((dataset_name, model_name))
                print(f"  {model_name}: mean={np.nanmean(scores):.4f}  time={elapsed:.1f}s (parallel)")
                save_checkpoint()
        finally:
            if executor:
                executor.shutdown(wait=True)

    # Final output: dump EVERY model that has data in the checkpoint, not just
    # models from this run. Otherwise re-running with --models X overwrites the
    # file with only X's results and loses everyone else.
    # NaN-pad datasets where a given model has no entry, so all arrays align to
    # the full evaluated_datasets list (consumed positionally by figures.ipynb).
    output_models = sorted({m for v in ckpt_results.values() for m in v.keys()})
    all_results = {name: [] for name in output_models}
    all_times   = {name: [] for name in output_models}
    nan_scores = [float("nan")] * N_OUTER_FOLDS
    for dataset_name in evaluated_datasets:
        for name in output_models:
            if dataset_name in ckpt_results and name in ckpt_results[dataset_name]:
                all_results[name].append(ckpt_results[dataset_name][name]["scores"])
                all_times[name].append(ckpt_results[dataset_name][name]["time"])
            else:
                all_results[name].append(nan_scores)
                all_times[name].append(float("nan"))
    for name in output_models:
        all_results[name] = np.array(all_results[name])
        all_times[name]   = np.array(all_times[name])

    joblib.dump((all_results, all_times, evaluated_datasets), FINAL_OUTPUT)
    print("\nDone. Results saved to", FINAL_OUTPUT)
