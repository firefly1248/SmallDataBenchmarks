"""Benchmark checkpoints, one joblib file per model.

`results/ckpt/<model>.joblib` holds `{dataset_name: {"scores", "preds",
"labels", "best_params", "time"}}`. One file per model so concurrent runs on
different models cannot clobber each other, and a save rewrites a few MB
instead of all of them.

A `run` names a subdirectory for results that must stay out of the published
aggregate: other outer-split seeds, datasets outside the UCI++ collection.
"""
import os
from pathlib import Path

import joblib

CKPT_DIR = "results/ckpt"


def atomic_dump(obj, path):
    """Write-then-rename. A checkpoint is the only copy of hundreds of hours of
    compute, and a crash mid-dump truncates it."""
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def ckpt_path(model_name, run=""):
    return os.path.join(CKPT_DIR, run, f"{model_name}.joblib")


def available_models(run=""):
    """Every model with a checkpoint on disk."""
    return sorted(p.stem for p in Path(CKPT_DIR, run).glob("*.joblib"))


def load_by_model(model_names=None, run=""):
    """{model_name: {dataset_name: entry}}. Missing checkpoints load as empty."""
    out = {}
    for model_name in (model_names if model_names is not None else available_models(run)):
        try:
            out[model_name] = joblib.load(ckpt_path(model_name, run))
        except FileNotFoundError:
            out[model_name] = {}
    return out


def load_by_dataset(model_names=None, run=""):
    """{dataset_name: {model_name: entry}} — the shape the figure notebooks use."""
    out: dict[str, dict] = {}
    for model_name, entries in load_by_model(model_names, run).items():
        for dataset_name, entry in entries.items():
            out.setdefault(dataset_name, {})[model_name] = entry
    return out
