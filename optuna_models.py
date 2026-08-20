"""
Benchmark: SVC, LogReg, TabPFN, TabPFN-3, TabICL, TabFM, RandomForest, XGBoost,
SGD, CatBoost, LightGBM, LightGBM-linear, HistGradientBoosting, TabNet,
FT-Transformer, ResNet.

Hyperparameter tuning strategy:
- TabFM                                : none — fixed defaults, 4 outer folds only
- SVC, LogReg, TabPFN, TabPFN-3, TabICL : GridSearchCV (small, well-understood HP space)
- All others                           : Optuna TPE (N_TRIALS / N_TRIALS_NN trials per outer fold)

Categorical handling:
- CatBoost, TabPFN, TabICL, TabFM: native categorical handling
- RF, XGBoost, LGBM, HGB         : CatFeaturesEncoder(ordinal) (NaN handled natively)
- SVC, LogReg, SGD               : CatFeaturesEncoder(target/...) + Imputer + Scaler
- TabNet, FT-Transformer, ResNet : ordinal-encode + impute + StandardScaler in wrapper
"""

import torch  # must precede xgboost / lightgbm / catboost to win the OpenMP init race
import argparse
import os
import signal
import sys
import time
import joblib
import warnings
import numpy as np

# 12 hours — above longest legit run (pendigits ~11h), below hangs (18h+).
# Overridable so the two pairs that only ever recorded NaN can be measured
# for real without editing this file.
DATASET_TIMEOUT = int(os.environ.get("DATASET_TIMEOUT", 43_200))


class _DatasetTimeout(BaseException):
    """Raised by the SIGALRM handler when a (dataset, model) pair overruns.

    BaseException, not Exception: Optuna's ``study.optimize(catch=(Exception,))``
    would otherwise swallow it, mark the trial failed and keep searching. The
    alarm is one-shot, so the dataset would then run unbounded — CatBoost sat on
    plant-species-leaves-shape for 14 h that way.
    """


def _alarm_handler(signum, frame):
    raise _DatasetTimeout()

from benchmark.checkpoints import (
    CKPT_DIR, atomic_dump, available_models, ckpt_path, load_by_model,
)
from benchmark.data import load_data_df
from benchmark.nested_cv import run_nested_cv
from config import RANDOM_STATE, N_OUTER_FOLDS, MAX_DATASET_ROWS, MODELS_TO_RUN

warnings.filterwarnings("ignore")


FINAL_OUTPUT = "results/optuna_models.joblib"

if __name__ == "__main__":
    ALL_MODELS = ["svc", "logreg", "tabpfn", "tabpfn3", "tabicl", "tabfm",
                  "random_forest", "xgboost", "sgd",
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

    os.makedirs(CKPT_DIR, exist_ok=True)
    _, _, _, evaluated_datasets, _ = joblib.load("results/compare_baseline_models.joblib")

    ckpt = load_by_model(MODEL_NAMES)
    for model_name in MODEL_NAMES:
        print(f"  {model_name:16s} {len(ckpt[model_name])}/{len(evaluated_datasets)} done")

    def save_checkpoint(model_name):
        atomic_dump(ckpt[model_name], ckpt_path(model_name))

    # A model failing on several datasets in a row is failing systemically —
    # missing weights, expired license, OOM — not per-dataset. Without this the
    # loop would race through all 146 datasets recording NaN and marking each
    # pair done, which then takes manual checkpoint surgery to undo.
    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures: dict[str, int] = {}

    for i, dataset_name in enumerate(evaluated_datasets):
        models_to_run = [m for m in MODEL_NAMES if dataset_name not in ckpt[m]]
        if not models_to_run:
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

        for model_name in models_to_run:
            start = time.time()
            failed = False
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(DATASET_TIMEOUT)
            try:
                scores, preds, labels, best_params = run_nested_cv(X, y, model_name, cat_cols)
            except _DatasetTimeout:
                print(f"  {model_name}: TIMEOUT after {DATASET_TIMEOUT // 3600}h")
                scores = [np.nan] * N_OUTER_FOLDS
                preds = labels = best_params = None
                failed = True
            except Exception as e:
                print(f"  {model_name}: ERROR — {e}")
                scores = [np.nan] * N_OUTER_FOLDS
                preds = labels = best_params = None
                failed = True
            finally:
                signal.alarm(0)
            elapsed = time.time() - start
            # preds is None when the model never trained. A limit-skip costs
            # microseconds and must not enter the cost figures; a timeout or an
            # error does cost real time, and nulling it understates them.
            if preds is None:
                elapsed = float("nan")
            # Only real errors count towards the breaker. A hard-limit skip
            # also yields no result, and those datasets can be adjacent
            # (plant-species-leaves-margin / -shape), so counting them would
            # abort a perfectly healthy run.
            consecutive_failures[model_name] = (
                consecutive_failures.get(model_name, 0) + 1 if failed else 0
            )
            ckpt[model_name][dataset_name] = {
                "scores": scores, "preds": preds, "labels": labels,
                "best_params": best_params, "time": elapsed,
            }
            print(f"  {model_name}: mean={np.nanmean(scores):.4f}  time={elapsed:.1f}s")
            save_checkpoint(model_name)
            if consecutive_failures[model_name] >= MAX_CONSECUTIVE_FAILURES:
                sys.exit(f"\n{model_name} produced no result on "
                         f"{MAX_CONSECUTIVE_FAILURES} datasets in a row — "
                         f"stopping before it NaNs the whole benchmark. "
                         f"Fix the cause, drop those datasets from "
                         f"{ckpt_path(model_name)}, and rerun.")

    # Final output: dump EVERY model that has a checkpoint, not just the models
    # from this run. Otherwise re-running with --models X overwrites the file
    # with only X's results and loses everyone else.
    # NaN-pad datasets where a given model has no entry, so all arrays align to
    # the full evaluated_datasets list (consumed positionally by figures.ipynb).
    output_models = available_models()
    final_ckpt = {m: ckpt[m] if m in ckpt else load_by_model([m])[m]
                  for m in output_models}
    all_results = {name: [] for name in output_models}
    all_times   = {name: [] for name in output_models}
    nan_scores = [float("nan")] * N_OUTER_FOLDS
    for dataset_name in evaluated_datasets:
        for name in output_models:
            entry = final_ckpt[name].get(dataset_name)
            if entry is not None:
                all_results[name].append(entry["scores"])
                all_times[name].append(entry["time"])
            else:
                all_results[name].append(nan_scores)
                all_times[name].append(float("nan"))
    for name in output_models:
        all_results[name] = np.array(all_results[name])
        all_times[name]   = np.array(all_times[name])

    joblib.dump((all_results, all_times, evaluated_datasets), FINAL_OUTPUT)
    print("\nDone. Results saved to", FINAL_OUTPUT)
