"""Data loading utilities for ARFF-formatted datasets."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.io import arff


def load_data(
    data_name: str,
    datasets_dir: str = "datasets",
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset as numpy arrays with one-hot encoded categoricals.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with integer labels
        Empty arrays are returned when the file is not found.
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

    unique_labels = df["Class"].unique()
    labels_dict = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = df["Class"].map(labels_dict).values.astype(int)
    X = pd.get_dummies(df.drop(columns=["Class"])).values.astype(float)

    return X, y


def load_data_df(
    data_name: str,
    datasets_dir: str = "datasets",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load dataset as a DataFrame with original types preserved.

    Categorical columns are decoded from bytes to str; numeric columns stay
    as float.  Columns that *look* numeric (>90 % parseable) are coerced.

    Returns
    -------
    X        : DataFrame of shape (n_samples, n_features)
    y        : ndarray of shape (n_samples,) with integer labels
    cat_cols : list of column names whose dtype is object (string categoricals)
        Empty structures are returned when the file is not found.
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
    unique_labels = target.unique()
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
