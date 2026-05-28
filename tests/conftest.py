"""Shared pytest fixtures for SmallDataBenchmarks test suite."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris


# Make sure tests always run from the project root so relative paths like
# "datasets/" resolve correctly.
@pytest.fixture(autouse=True, scope="session")
def project_root(tmp_path_factory):
    root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(root)


@pytest.fixture
def iris_df() -> tuple[pd.DataFrame, np.ndarray]:
    """Iris dataset as a DataFrame — no categorical columns."""
    X_arr, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X_arr, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    return X, y


@pytest.fixture
def binary_df() -> tuple[pd.DataFrame, np.ndarray]:
    """Binary version of iris (classes 0 and 1 only)."""
    X_arr, y_arr = load_iris(return_X_y=True)
    mask = y_arr < 2
    X = pd.DataFrame(X_arr[mask], columns=["f1", "f2", "f3", "f4"])
    y = y_arr[mask]
    return X, y


@pytest.fixture
def mixed_df() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic DataFrame with both numeric and categorical columns."""
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.uniform(size=n),
        "cat1": rng.choice(["a", "b", "c"], size=n),
        "cat2": rng.choice(["x", "y"], size=n),
    })
    y = rng.integers(0, 3, size=n)
    return X, y


@pytest.fixture
def numeric_only_df() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic DataFrame with no categorical columns."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.uniform(size=n),
        "c": rng.standard_t(df=3, size=n),
    })
    y = rng.integers(0, 2, size=n)
    return X, y
