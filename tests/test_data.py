"""Tests for benchmark.data (data loading utilities)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.data import load_data, load_data_df


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_warns_when_file_missing(self):
        with pytest.warns(UserWarning, match="not found"):
            X, y = load_data("__nonexistent_dataset__")

    def test_returns_empty_arrays_when_missing(self):
        with pytest.warns(UserWarning):
            X, y = load_data("__nonexistent_dataset__")
        assert X.shape == (0,)
        assert y.shape == (0,)

    @pytest.mark.integration
    def test_returns_numpy_arrays(self):
        X, y = load_data("abalone-3class")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    @pytest.mark.integration
    def test_shapes_consistent(self):
        X, y = load_data("abalone-3class")
        assert X.shape[0] == len(y)

    @pytest.mark.integration
    def test_labels_are_integers(self):
        _, y = load_data("abalone-3class")
        assert y.dtype in (np.int32, np.int64, int)

    @pytest.mark.integration
    def test_no_object_columns_after_onehot(self):
        X, _ = load_data("abalone-3class")
        # one-hot encoding produces a float array — no object dtype
        assert X.dtype != object


# ---------------------------------------------------------------------------
# load_data_df
# ---------------------------------------------------------------------------

class TestLoadDataDf:
    def test_warns_when_file_missing(self):
        with pytest.warns(UserWarning, match="not found"):
            load_data_df("__nonexistent_dataset__")

    def test_returns_empty_structures_when_missing(self):
        with pytest.warns(UserWarning):
            X, y, cat_cols = load_data_df("__nonexistent_dataset__")
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 0
        assert len(y) == 0
        assert cat_cols == []

    @pytest.mark.integration
    def test_returns_correct_types(self):
        X, y, cat_cols = load_data_df("abalone-3class")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(cat_cols, list)

    @pytest.mark.integration
    def test_shapes_consistent(self):
        X, y, _ = load_data_df("abalone-3class")
        assert len(X) == len(y)

    @pytest.mark.integration
    def test_cat_cols_have_object_dtype(self):
        X, _, cat_cols = load_data_df("abalone-3class")
        for col in cat_cols:
            assert X[col].dtype == object, f"Column {col!r} should be object dtype"

    @pytest.mark.integration
    def test_numeric_dataset_has_no_cat_cols(self):
        # iris has no categorical columns
        X, _, cat_cols = load_data_df("iris")
        assert cat_cols == []

    @pytest.mark.integration
    def test_no_class_column_in_X(self):
        X, _, _ = load_data_df("abalone-3class")
        assert "Class" not in X.columns
