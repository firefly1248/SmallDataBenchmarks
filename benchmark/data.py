"""Data loading utilities for ARFF-formatted datasets."""
from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff

from config import DUPLICATE_DATASETS

BASELINE_RESULTS = "results/compare_baseline_models.joblib"


def evaluated_datasets() -> list[str]:
    """Every dataset the benchmark has scored, in the order the run fixed."""
    *_, names, _ = joblib.load(BASELINE_RESULTS)
    return list(names)


def datasets_to_run() -> list[str]:
    """What a new run should cover: the above minus the UCI++ duplicates.

    The only reader of ``DUPLICATE_DATASETS`` on the compute side — a runner
    that wants the filter calls this rather than repeating the test. Aggregates
    keep iterating ``evaluated_datasets`` so scores already on disk for the
    duplicates survive; only fresh compute is skipped.
    """
    return [name for name in evaluated_datasets() if name not in DUPLICATE_DATASETS]


def load_data(
    data_name: str,
    datasets_dir: str = "datasets",
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset as numpy arrays with one-hot encoded categoricals.

    Returns empty arrays when the file is not found.
    """
    file_path = os.path.join(datasets_dir, f"{data_name}.arff")
    if not os.path.exists(file_path):
        warnings.warn(f"Dataset not found, skipping: {file_path}", UserWarning, stacklevel=2)
        return np.array([]), np.array([])

    data, _ = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)

    unique_labels = sorted(df["Class"].unique())
    labels_dict = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = df["Class"].map(labels_dict).values.astype(int)
    X = pd.get_dummies(df.drop(columns=["Class"])).values.astype(float)

    return X, y


def load_data_df(
    data_name: str,
    datasets_dir: str = "datasets",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load dataset as a DataFrame with original types preserved.

    Categorical columns are decoded from bytes to str; columns that *look*
    numeric (>90 % parseable) are coerced. Returns ``(X, y, cat_cols)``, empty
    when the file is not found.
    """
    file_path = os.path.join(datasets_dir, f"{data_name}.arff")
    if not os.path.exists(file_path):
        warnings.warn(f"Dataset not found, skipping: {file_path}", UserWarning, stacklevel=2)
        return pd.DataFrame(), np.array([]), []

    data, _ = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)

    target = df["Class"]
    unique_labels = sorted(target.unique())
    labels_dict = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = target.map(labels_dict).values.astype(int)

    X = df.drop(columns=["Class"]).copy()

    for col in X.columns:
        if X[col].dtype == object:
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() > 0.9:
                X[col] = converted

    cat_cols = X.select_dtypes(include="object").columns.tolist()

    return X, y, cat_cols
