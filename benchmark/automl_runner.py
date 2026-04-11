"""Shared sequential benchmark runner for AutoML frameworks (AutoGluon, MLJAR).

Both frameworks follow the same outer loop structure:
- load evaluated_datasets from compare_baseline_models results
- positional checkpoint/resume (index-based, not name-based)
- save checkpoint after each dataset
- write final output when all datasets are done
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np

from benchmark.data import load_data
from config import MAX_DATASET_ROWS, RANDOM_STATE


def run_automl_benchmark(
    evaluate_fn: Callable[[np.ndarray, np.ndarray], list[float]],
    evaluated_datasets: np.ndarray,
    rf_results: np.ndarray,
    checkpoint_path: str | Path,
    final_output_path: str | Path,
) -> None:
    """Run an AutoML benchmark over *evaluated_datasets* with resume support.

    Parameters
    ----------
    evaluate_fn
        ``evaluate_autogluon`` or ``evaluate_mljar`` — takes ``(X, y)`` and
        returns a list of PR AUC scores (one per outer fold).
    evaluated_datasets
        Ordered array of dataset names (from compare_baseline_models output).
    rf_results
        Random-forest baseline scores, used only for the console comparison line.
    checkpoint_path
        Path to the intermediate checkpoint file (``*_ckpt.joblib``).
    final_output_path
        Path written only when all datasets are complete.
    """
    results: list = []
    times: list = []

    try:
        saved = joblib.load(checkpoint_path)
        results = list(saved[0])
        times   = list(saved[1])
        print(f"Resuming from checkpoint: {len(results)} datasets already done.")
    except FileNotFoundError:
        pass

    n_done = len(results)
    n_total = len(evaluated_datasets)

    for i, dataset_name in enumerate(evaluated_datasets):
        if i < n_done:
            print(f"[{i+1}/{n_total}] {dataset_name}  — skipping (done)")
            continue

        X, y = load_data(dataset_name)
        if len(y) == 0:
            continue

        rng = np.random.default_rng(RANDOM_STATE)
        if len(y) > MAX_DATASET_ROWS:
            idx = rng.choice(len(y), MAX_DATASET_ROWS, replace=False)
            X, y = X[idx], y[idx]

        print(f"[{i+1}/{n_total}] {dataset_name}  shape={X.shape}")
        start = time.time()
        nested_scores = evaluate_fn(X, y)
        elapsed = time.time() - start

        results.append(nested_scores)
        times.append(elapsed)
        print(
            f"  done. elapsed={elapsed:.1f}s  "
            f"PR AUC={np.mean(nested_scores):.4f}  "
            f"(RF baseline: {np.mean(rf_results[i]):.4f})"
        )
        joblib.dump((np.array(results), np.array(times)), checkpoint_path)

    joblib.dump((np.array(results), np.array(times)), final_output_path)
    print(f"\nDone. Results saved to {final_output_path}")
