"""Shared sequential runner for the AutoML frameworks (AutoGluon, MLJAR).

Both follow the same loop: read evaluated_datasets, resume from a name-keyed
checkpoint, save after each dataset, write the final output when all are done.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np

from benchmark.checkpoints import atomic_dump
from benchmark.data import load_data
from config import MAX_DATASET_ROWS, RANDOM_STATE


def _load_checkpoint(
    checkpoint_path: str | Path,
    evaluated_datasets: np.ndarray,
) -> dict[str, dict]:
    """Load the checkpoint, migrating the old positional tuple format if needed.

    New: ``{dataset_name: {'scores': list[float], 'time': float}}``
    Old: ``(ndarray of shape (n, folds), ndarray of shape (n,))``
    """
    try:
        saved = joblib.load(checkpoint_path)
    except FileNotFoundError:
        return {}

    if isinstance(saved, dict):
        return saved

    # Old positional format — migrate to name-based dict
    results_arr, times_arr = saved
    checkpoint: dict[str, dict] = {}
    for i, (scores, t) in enumerate(zip(results_arr, times_arr)):
        name = evaluated_datasets[i]
        checkpoint[name] = {"scores": list(scores), "time": float(t)}
    print(f"Migrated checkpoint from positional to name-based format ({len(checkpoint)} datasets).")
    return checkpoint


def run_automl_benchmark(
    evaluate_fn: Callable[[np.ndarray, np.ndarray], list[float]],
    evaluated_datasets: np.ndarray,
    rf_results: np.ndarray,
    checkpoint_path: str | Path,
    final_output_path: str | Path,
) -> None:
    """Run an AutoML benchmark over *evaluated_datasets* with resume support.

    ``evaluate_fn`` takes ``(X, y)`` and returns one PR AUC per outer fold.
    ``final_output_path`` is written only once every dataset is complete.
    """
    checkpoint = _load_checkpoint(checkpoint_path, evaluated_datasets)
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} datasets already done.")

    n_total = len(evaluated_datasets)

    for i, dataset_name in enumerate(evaluated_datasets):
        if dataset_name in checkpoint:
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

        checkpoint[dataset_name] = {"scores": nested_scores, "time": elapsed}
        print(
            f"  done. elapsed={elapsed:.1f}s  "
            f"PR AUC={np.mean(nested_scores):.4f}  "
            f"(RF baseline: {np.mean(rf_results[i]):.4f})"
        )
        atomic_dump(checkpoint, checkpoint_path)

    # Ship the names: a positional match silently misattributes every row as
    # soon as the dataset list grows.
    valid_datasets = [n for n in evaluated_datasets if n in checkpoint]
    results = [checkpoint[name]["scores"] for name in valid_datasets]
    times   = [checkpoint[name]["time"]   for name in valid_datasets]
    atomic_dump((np.array(results), np.array(times), valid_datasets), final_output_path)
    print(f"\nDone. Results saved to {final_output_path}")
