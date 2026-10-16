"""Tests for benchmark.data (data loading utilities)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.data import load_data, load_data_df


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
        assert X.dtype != object


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
        X, _, cat_cols = load_data_df("iris")
        assert cat_cols == []

    @pytest.mark.integration
    def test_no_class_column_in_X(self):
        X, _, _ = load_data_df("abalone-3class")
        assert "Class" not in X.columns


class TestExternalDatasets:
    @pytest.fixture
    def external(self, tmp_path):
        (tmp_path / "external").mkdir()
        pd.DataFrame({"id": [1, 2, 3, 4], "code": [1, 2, 1, 2], "x": [0.1, 0.2, 0.3, 0.4],
                      "note": ["a", "b", "a", "b"], "outcome": ["no", "yes", "no", "yes"]}
                     ).to_csv(tmp_path / "external" / "survey.csv", index=False)
        pd.DataFrame([{"dataset": "survey", "source": "test", "url": "", "first_public": "2026-10-01",
                       "target": "outcome", "positive": "yes", "drop_columns": "id;note",
                       "categorical": "code", "license": "CC0"}]).to_csv(tmp_path / "external.csv", index=False)
        return str(tmp_path)

    def test_target_drops_and_categoricals(self, external):
        X, y, cat_cols = load_data_df("survey", datasets_dir=external)
        assert list(X.columns) == ["code", "x"]
        assert cat_cols == ["code"]
        assert y.tolist() == [0, 1, 0, 1]

    def test_positive_class_becomes_one(self, external):
        # "no" < "yes" already puts the positive last; flip the names to test the swap.
        df = pd.read_csv(f"{external}/external.csv")
        df["positive"] = "no"
        df.to_csv(f"{external}/external.csv", index=False)
        _, y, _ = load_data_df("survey", datasets_dir=external)
        assert y.tolist() == [1, 0, 1, 0]

    def test_unknown_positive_raises(self, external):
        df = pd.read_csv(f"{external}/external.csv")
        df["positive"] = "yess"
        df.to_csv(f"{external}/external.csv", index=False)
        with pytest.raises(ValueError):
            load_data_df("survey", datasets_dir=external)

    def test_onehot_view_matches(self, external):
        X, y = load_data("survey", datasets_dir=external)
        assert X.shape == (4, 3)  # x plus one-hot code
        assert y.tolist() == [0, 1, 0, 1]

    def test_listed_for_the_post_release_collection(self, external, monkeypatch):
        from benchmark import data
        listed = data.external_datasets(external)
        monkeypatch.setattr(data, "external_datasets", lambda datasets_dir="datasets": listed)
        assert data.datasets_to_run("post-release") == ["survey"]


class TestSubsample:
    def test_loader_draws_the_runners_rows(self, tmp_path):
        from config import MAX_DATASET_ROWS, RANDOM_STATE
        n = MAX_DATASET_ROWS + 500
        (tmp_path / "external").mkdir()
        pd.DataFrame({"x": np.arange(n), "y": np.arange(n) % 2}).to_csv(tmp_path / "external" / "big.csv", index=False)
        pd.DataFrame([{"dataset": "big", "source": "", "url": "", "first_public": "", "target": "y",
                       "positive": "", "drop_columns": "", "categorical": "", "license": ""}]
                     ).to_csv(tmp_path / "external.csv", index=False)
        X, y, _ = load_data_df("big", datasets_dir=str(tmp_path))
        idx = np.random.default_rng(RANDOM_STATE).choice(n, MAX_DATASET_ROWS, replace=False)
        assert X["x"].tolist() == idx.tolist()
        assert len(load_data_df("big", datasets_dir=str(tmp_path), subsample=False)[0]) == n
