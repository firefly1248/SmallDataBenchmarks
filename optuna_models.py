"""Benchmark entry point: every model over every dataset, nested CV.

Tuning: TabFM none; SVC, LogReg, TabPFN, TabPFN-3, TabICL by GridSearchCV;
the rest by Optuna TPE.

Categoricals: CatBoost, TabPFN, TabICL and TabFM handle them natively; RF,
XGBoost, LGBM and HGB get ordinal encoding; SVC, LogReg and SGD get target
encoding, imputation and scaling; the torch wrappers do their own.
"""

import torch  # must precede xgboost / lightgbm / catboost to win the OpenMP init race
import argparse
import os
import signal
import sys
import time
import warnings
import numpy as np

# 12h: above the longest legit run (pendigits ~11h), below hangs (18h+).
# Env-overridable so a suspect pair can be re-measured without an edit.
DATASET_TIMEOUT = int(os.environ.get("DATASET_TIMEOUT", 43_200))


class _DatasetTimeout(BaseException):
    """Raised by the SIGALRM handler when a (dataset, model) pair overruns.

    BaseException so Optuna's ``catch=(Exception,)`` cannot swallow it and let
    the dataset run unbounded. See Findings_notes.md.
    """


def _alarm_handler(signum, frame):
    raise _DatasetTimeout()

from benchmark.checkpoints import (
    CKPT_DIR, atomic_dump, available_models, ckpt_path, load_by_model,
)
from benchmark.data import datasets_to_run, evaluated_datasets, load_data_df
from benchmark.nested_cv import run_nested_cv
from config import RANDOM_STATE, N_OUTER_FOLDS, MAX_DATASET_ROWS, MODELS_TO_RUN

warnings.filterwarnings("ignore")


FINAL_OUTPUT = "results/optuna_models.joblib"

if __name__ == "__main__":
    ALL_MODELS = ["svc", "logreg", "tabpfn3", "tabpfn35", "tabpfn35fast", "tabicl", "tabfm",
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
    # The work loop skips the UCI++ duplicates; the aggregate below still spans
    # every dataset, so the scores already stored for them stay published.
    run_datasets = datasets_to_run()
    all_datasets = evaluated_datasets()

    ckpt = load_by_model(MODEL_NAMES)
    for model_name in MODEL_NAMES:
        done = sum(1 for d in run_datasets if d in ckpt[model_name])
        print(f"  {model_name:16s} {done}/{len(run_datasets)} done")

    def save_checkpoint(model_name):
        atomic_dump(ckpt[model_name], ckpt_path(model_name))

    # Consecutive failures mean the model is broken, not the data. Without this
    # the run marks all 146 pairs done as NaN, undone only by checkpoint surgery.
    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures: dict[str, int] = {}

    for i, dataset_name in enumerate(run_datasets):
        models_to_run = [m for m in MODEL_NAMES if dataset_name not in ckpt[m]]
        if not models_to_run:
            print(f"[{i+1}/{len(run_datasets)}] {dataset_name}  — skipping (all models done)")
            continue

        X, y, cat_cols = load_data_df(dataset_name)
        if len(y) == 0:
            continue

        rng = np.random.default_rng(RANDOM_STATE)
        if len(X) > MAX_DATASET_ROWS:
            random_idx = rng.choice(len(X), MAX_DATASET_ROWS, replace=False)
            X = X.iloc[random_idx].reset_index(drop=True)
            y = y[random_idx]

        print(f"\n[{i+1}/{len(run_datasets)}] {dataset_name}  "
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
            # A limit-skip costs microseconds and must stay out of the cost
            # figures; a timeout or error cost real time and must stay in.
            if preds is None:
                elapsed = float("nan")
            # Limit-skips also yield no result and can be adjacent, so counting
            # them towards the breaker would abort a healthy run.
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

    # Dump every model with a checkpoint, not just this run's, or --models X
    # overwrites the file with X alone. NaN-pad missing datasets so the arrays
    # stay aligned with all_datasets, which figures.ipynb reads positionally.
    output_models = available_models()
    final_ckpt = {**load_by_model(output_models), **ckpt}
    all_results = {name: [] for name in output_models}
    all_times   = {name: [] for name in output_models}
    nan_scores = [float("nan")] * N_OUTER_FOLDS
    for dataset_name in all_datasets:
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

    atomic_dump((all_results, all_times, all_datasets), FINAL_OUTPUT)
    print("\nDone. Results saved to", FINAL_OUTPUT)
