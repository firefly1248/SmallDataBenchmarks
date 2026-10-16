"""Dataset loading: the UCI++ ARFF collection and the external CSV datasets.

External datasets are listed in ``datasets/external.csv``, one row per dataset with
``dataset, source, url, first_public, target, positive, drop_columns, categorical,
license`` (the two list columns ``;``-separated), and stored as
``datasets/external/<dataset>.csv``. Labels are the target's values sorted as strings;
``positive`` names the value that becomes the last class, class 1 on binary data, the
class PR AUC scores. Every loader applies the ``MAX_DATASET_ROWS`` subsample the runs use.
"""
from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff

from config import DUPLICATE_DATASETS, MAX_DATASET_ROWS, RANDOM_STATE

BASELINE_RESULTS = "results/compare_baseline_models.joblib"


def evaluated_datasets() -> list[str]:
    """Every UCI++ dataset the benchmark has scored, in the order the run fixed."""
    *_, names, _ = joblib.load(BASELINE_RESULTS)
    return list(names)


def external_datasets(datasets_dir: str = "datasets") -> pd.DataFrame:
    """The external list, indexed by dataset name."""
    return pd.read_csv(os.path.join(datasets_dir, "external.csv"), dtype=str,
                       keep_default_na=False).set_index("dataset")


def datasets_to_run(collection: str = "uci++") -> list[str]:
    """What a run should cover: the UCI++ datasets minus the duplicates, or the
    external list.

    The only reader of ``DUPLICATE_DATASETS`` on the compute side — a runner
    that wants the filter calls this rather than repeating the test. Aggregates
    keep iterating ``evaluated_datasets`` so scores already on disk for the
    duplicates survive; only fresh compute is skipped.
    """
    if collection == "post-release":
        return list(external_datasets().index)
    return [name for name in evaluated_datasets() if name not in DUPLICATE_DATASETS]


def _read(data_name: str, datasets_dir: str) -> tuple[pd.DataFrame, str, str | None, list[str]] | None:
    """The decoded frame, its target column, the positive label and the declared
    categorical columns, or None if the dataset is in neither collection."""
    arff_path = os.path.join(datasets_dir, f"{data_name}.arff")
    if os.path.exists(arff_path):
        data, _ = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
        return df, "Class", None, []
    csv_path = os.path.join(datasets_dir, "external", f"{data_name}.csv")
    if os.path.exists(csv_path):
        row = external_datasets(datasets_dir).loc[data_name]
        df = pd.read_csv(csv_path).drop(columns=row["drop_columns"].split(";") if row["drop_columns"] else [])
        categorical = row["categorical"].split(";") if row["categorical"] else []
        for col in categorical:
            df[col] = df[col].astype(str)
        df[row["target"]] = df[row["target"]].astype(str)
        return df, row["target"], row["positive"] or None, categorical
    warnings.warn(f"Dataset not found, skipping: {data_name}", UserWarning, stacklevel=3)
    return None


def _subsample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_DATASET_ROWS:
        return df
    idx = np.random.default_rng(RANDOM_STATE).choice(len(df), MAX_DATASET_ROWS, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _labels(target: pd.Series, positive: str | None) -> np.ndarray:
    unique = sorted(target.unique())
    if positive is not None:
        unique.remove(positive)
        unique.append(positive)
    return target.map({lbl: i for i, lbl in enumerate(unique)}).to_numpy().astype(int)


def load_data(data_name: str, datasets_dir: str = "datasets", subsample: bool = True,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset as numpy arrays with one-hot encoded categoricals.

    Returns empty arrays when the dataset is not found.
    """
    read = _read(data_name, datasets_dir)
    if read is None:
        return np.array([]), np.array([])
    df, target, positive, _ = read
    if subsample:
        df = _subsample(df)
    y = _labels(df[target], positive)
    X = pd.get_dummies(df.drop(columns=[target])).values.astype(float)
    return X, y


def load_data_df(data_name: str, datasets_dir: str = "datasets", subsample: bool = True,
                 ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load dataset as a DataFrame with original types preserved.

    Columns that *look* numeric (>90 % parseable) are coerced, unless the external
    list declares them categorical. Returns
    ``(X, y, cat_cols)``, empty when the dataset is not found.
    """
    read = _read(data_name, datasets_dir)
    if read is None:
        return pd.DataFrame(), np.array([]), []
    df, target, positive, categorical = read
    if subsample:
        df = _subsample(df)
    y = _labels(df[target], positive)
    X = df.drop(columns=[target]).copy()
    for col in X.columns:
        if X[col].dtype == object and col not in categorical:
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() > 0.9:
                X[col] = converted
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    return X, y, cat_cols
