"""Tests for benchmark.encoding (CatFeaturesEncoder, CatBoostNativeWrapper)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.encoding import (
    CAT_STRATEGIES_LINEAR,
    CAT_STRATEGY_TREE,
    CatBoostNativeWrapper,
    CatFeaturesEncoder,
)


# ---------------------------------------------------------------------------
# CatFeaturesEncoder
# ---------------------------------------------------------------------------

class TestCatFeaturesEncoderNumericPassthrough:
    """Encoder must be a no-op when there are no categorical columns."""

    def test_shape_unchanged(self, numeric_only_df):
        X, y = numeric_only_df
        enc = CatFeaturesEncoder(strategy="ordinal")
        X_out = enc.fit_transform(X, y)
        assert X_out.shape == X.shape

    def test_values_unchanged(self, numeric_only_df):
        X, y = numeric_only_df
        enc = CatFeaturesEncoder(strategy="ordinal")
        X_out = enc.fit_transform(X, y)
        pd.testing.assert_frame_equal(X_out.reset_index(drop=True), X.reset_index(drop=True))

    def test_no_object_columns_in_output(self, numeric_only_df):
        X, y = numeric_only_df
        enc = CatFeaturesEncoder()
        X_out = enc.fit_transform(X, y)
        assert list(X_out.select_dtypes(include="object").columns) == []


class TestCatFeaturesEncoderWithCategoricals:
    """Encoder must replace object columns with numeric representations."""

    def test_no_object_columns_after_transform(self, mixed_df):
        X, y = mixed_df
        enc = CatFeaturesEncoder(strategy="ordinal")
        X_out = enc.fit_transform(X, y)
        assert list(X_out.select_dtypes(include="object").columns) == []

    def test_row_count_preserved(self, mixed_df):
        X, y = mixed_df
        enc = CatFeaturesEncoder(strategy="target")
        X_out = enc.fit_transform(X, y)
        assert len(X_out) == len(X)

    def test_column_count_preserved(self, mixed_df):
        X, y = mixed_df
        enc = CatFeaturesEncoder(strategy="ordinal")
        X_out = enc.fit_transform(X, y)
        assert X_out.shape[1] == X.shape[1]

    @pytest.mark.parametrize("strategy", CAT_STRATEGIES_LINEAR)
    def test_all_strategies_produce_numeric_output(self, mixed_df, strategy):
        X, y = mixed_df
        enc = CatFeaturesEncoder(strategy=strategy)
        X_out = enc.fit_transform(X, y)
        assert list(X_out.select_dtypes(include="object").columns) == [], (
            f"Strategy {strategy!r} left object columns in output"
        )

    def test_transform_without_unknown_categories(self, mixed_df):
        """Transform on held-out data with same categories should not raise."""
        X, y = mixed_df
        split = len(X) // 2
        enc = CatFeaturesEncoder(strategy="ordinal")
        enc.fit(X.iloc[:split], y[:split])
        enc.transform(X.iloc[split:])  # must not raise

    def test_fit_returns_self(self, mixed_df):
        X, y = mixed_df
        enc = CatFeaturesEncoder()
        result = enc.fit(X, y)
        assert result is enc


class TestCatFeaturesEncoderSklearnCompat:
    """Verify get_params / set_params compatibility."""

    def test_get_params(self):
        enc = CatFeaturesEncoder(strategy="target")
        params = enc.get_params()
        assert params["strategy"] == "target"

    def test_set_params(self):
        enc = CatFeaturesEncoder(strategy="ordinal")
        enc.set_params(strategy="james_stein")
        assert enc.strategy == "james_stein"

    def test_clone_preserves_strategy(self):
        from sklearn.base import clone
        enc = CatFeaturesEncoder(strategy="m_estimate")
        cloned = clone(enc)
        assert cloned.strategy == "m_estimate"


class TestCatStrategyConstants:
    def test_cat_strategy_tree_is_ordinal(self):
        assert CAT_STRATEGY_TREE == "ordinal"

    def test_all_linear_strategies_are_valid(self):
        valid = set(CatFeaturesEncoder._ENCODER_MAP)
        for s in CAT_STRATEGIES_LINEAR:
            assert s in valid, f"{s!r} not in CatFeaturesEncoder._ENCODER_MAP"


# ---------------------------------------------------------------------------
# CatBoostNativeWrapper
# ---------------------------------------------------------------------------

class TestCatBoostNativeWrapper:
    @pytest.fixture
    def cat_data(self):
        rng = np.random.default_rng(7)
        n = 60
        X = pd.DataFrame({
            "num":  rng.normal(size=n),
            "cat":  rng.choice(["a", "b", "c"], size=n),
        })
        y = rng.integers(0, 2, size=n)
        return X, y

    def test_fit_predict_proba_shape_binary(self, cat_data):
        X, y = cat_data
        wrapper = CatBoostNativeWrapper(
            cat_cols=["cat"],
            iterations=5,
            verbose=0,
        )
        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, cat_data):
        X, y = cat_data
        wrapper = CatBoostNativeWrapper(cat_cols=["cat"], iterations=5, verbose=0)
        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_returns_integer_labels(self, cat_data):
        X, y = cat_data
        wrapper = CatBoostNativeWrapper(cat_cols=["cat"], iterations=5, verbose=0)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert set(preds).issubset({0, 1})

    def test_nan_in_cat_col_filled_with_missing(self, cat_data):
        X, y = cat_data
        X_nan = X.copy()
        X_nan.loc[0, "cat"] = None
        wrapper = CatBoostNativeWrapper(cat_cols=["cat"], iterations=5, verbose=0)
        wrapper.fit(X, y)
        wrapper.predict_proba(X_nan)  # must not raise

    def test_get_params_contains_cat_cols(self, cat_data):
        wrapper = CatBoostNativeWrapper(cat_cols=["cat"], iterations=5, verbose=0)
        params = wrapper.get_params()
        assert "cat_cols" in params
        assert params["cat_cols"] == ["cat"]

    def test_is_classifier(self, cat_data):
        """ClassifierMixin must come first in MRO for sklearn 1.7 tags API."""
        from sklearn.utils.estimator_checks import parametrize_with_checks
        from sklearn.utils import estimator_html_repr
        wrapper = CatBoostNativeWrapper(cat_cols=[], iterations=2, verbose=0)
        from sklearn.utils.validation import check_is_fitted
        # is_classifier check via tags
        from sklearn.base import is_classifier
        assert is_classifier(wrapper)
