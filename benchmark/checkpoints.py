"""Benchmark checkpoints, one joblib file per model.

`results/ckpt/<model>.joblib` holds `{dataset_name: {"scores", "preds",
"labels", "best_params", "time"}}`. One file per model rather than one shared
file so that several runs can work on different models concurrently without
clobbering each other, and so a save rewrites a few MB instead of all of them.
"""
import glob
import os

import joblib

CKPT_DIR = "results/ckpt"


def atomic_dump(obj, path):
    """Write-then-rename. A checkpoint is the only copy of hundreds of hours of
    compute, and a crash mid-dump truncates it."""
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def ckpt_path(model_name):
    return os.path.join(CKPT_DIR, f"{model_name}.joblib")


def available_models():
    """Every model with a checkpoint on disk."""
    return sorted(os.path.basename(p)[: -len(".joblib")]
                  for p in glob.glob(os.path.join(CKPT_DIR, "*.joblib")))


def load_by_model(model_names=None):
    """{model_name: {dataset_name: entry}}. Missing checkpoints load as empty."""
    out = {}
    for model_name in (model_names if model_names is not None else available_models()):
        try:
            out[model_name] = joblib.load(ckpt_path(model_name))
        except FileNotFoundError:
            out[model_name] = {}
    return out


def load_by_dataset(model_names=None):
    """{dataset_name: {model_name: entry}} — the shape the figure notebooks use."""
    out: dict[str, dict] = {}
    for model_name, entries in load_by_model(model_names).items():
        for dataset_name, entry in entries.items():
            out.setdefault(dataset_name, {})[model_name] = entry
    return out
