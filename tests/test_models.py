"""Tests for benchmark.models.grid_search and the nested-CV model routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from benchmark.models.grid_search import GRID_SEARCH_MODELS, build_grid_search
from benchmark.nested_cv import _MODEL_LIMITS, run_nested_cv


class TestBuildGridSearch:
    @pytest.fixture
    def inner_cv(self):
        return StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    @pytest.mark.parametrize("model_name", ["svc", "logreg", "tabpfn", "tabpfn3", "tabicl"])
    def test_returns_grid_search_cv(self, model_name, inner_cv):
        gs = build_grid_search(model_name, inner_cv, cat_cols=[])
        assert isinstance(gs, GridSearchCV)

    def test_raises_for_unknown_model(self, inner_cv):
        with pytest.raises(ValueError, match="not a grid-search model"):
            build_grid_search("random_forest", inner_cv, cat_cols=[])

    @pytest.mark.parametrize("model_name", ["svc", "logreg"])
    def test_no_cat_strategy_without_cat_cols(self, model_name, inner_cv):
        gs = build_grid_search(model_name, inner_cv, cat_cols=[])
        grids = gs.param_grid if isinstance(gs.param_grid, list) else [gs.param_grid]
        for g in grids:
            assert "cat_enc__strategy" not in g

    @pytest.mark.parametrize("model_name", ["svc", "logreg"])
    def test_cat_strategy_present_with_cat_cols(self, model_name, inner_cv):
        gs = build_grid_search(model_name, inner_cv, cat_cols=["col1"])
        grids = gs.param_grid if isinstance(gs.param_grid, list) else [gs.param_grid]
        assert all("cat_enc__strategy" in g for g in grids)

    def test_svc_grid_has_kernel_variants(self, inner_cv):
        gs = build_grid_search("svc", inner_cv, cat_cols=[])
        kernels = {v for g in gs.param_grid for v in g.get("svc__kernel", [])}
        assert "rbf" in kernels
        assert "linear" in kernels

    def test_logreg_grid_has_elasticnet_ratio(self, inner_cv):
        gs = build_grid_search("logreg", inner_cv, cat_cols=[])
        assert "lr__l1_ratio" in gs.param_grid

    @pytest.mark.parametrize("model_name", ["svc", "logreg"])
    def test_fit_and_predict_on_iris(self, model_name, inner_cv, iris_df):
        X, y = iris_df
        gs = build_grid_search(model_name, inner_cv, cat_cols=[])
        gs.fit(X, y)
        proba = gs.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_grid_search_models_constant(self):
        assert "svc" in GRID_SEARCH_MODELS
        assert "logreg" in GRID_SEARCH_MODELS
        assert "tabpfn" in GRID_SEARCH_MODELS
        assert "tabpfn3" in GRID_SEARCH_MODELS
        assert "tabicl" in GRID_SEARCH_MODELS
        assert "random_forest" not in GRID_SEARCH_MODELS

    def test_tabfm_is_not_a_grid_search_model(self, inner_cv):
        assert "tabfm" not in GRID_SEARCH_MODELS
        with pytest.raises(ValueError, match="not a grid-search model"):
            build_grid_search("tabfm", inner_cv, cat_cols=[])

    @pytest.mark.parametrize(
        ("model_name", "version"), [("tabpfn", "v2.6"), ("tabpfn3", "v3")]
    )
    def test_tabpfn_keys_pin_their_checkpoint(self, model_name, version, inner_cv):
        gs = build_grid_search(model_name, inner_cv, cat_cols=[])
        assert gs.estimator.get_params()["model_version"] == version

    @pytest.mark.parametrize("model_name", ["tabpfn", "tabpfn3", "tabicl"])
    def test_torch_models_search_serially(self, model_name, inner_cv):
        assert build_grid_search(model_name, inner_cv, cat_cols=[]).n_jobs == 1

    @pytest.mark.parametrize(
        ("model_name", "auto_scale"), [("tabpfn", False), ("tabpfn3", True)]
    )
    def test_only_v3_auto_scales_the_ensemble(self, model_name, auto_scale, inner_cv):
        """v2.6 must match tabpfn 7.1.1, which had no auto-scaling at all."""
        gs = build_grid_search(model_name, inner_cv, cat_cols=[])
        assert gs.estimator.get_params()["auto_scale_n_estimators"] is auto_scale


class TestModelLimits:
    """Datasets past a model's hard limits are skipped, not fitted."""

    @staticmethod
    def _frame(n_features: int, n_classes: int, n_rows: int = 40):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.normal(size=(n_rows, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = np.tile(np.arange(n_classes), n_rows // n_classes + 1)[:n_rows]
        return X, y

    def test_tabfm_skips_many_class_datasets(self):
        X, y = self._frame(n_features=5, n_classes=11, n_rows=44)
        scores, preds, labels, best = run_nested_cv(X, y, "tabfm", cat_cols=[])
        assert all(np.isnan(s) for s in scores)
        assert preds is None and labels is None and best is None

    @pytest.mark.parametrize("model_name", ["tabfm", "tabicl"])
    def test_wide_datasets_are_skipped(self, model_name):
        X, y = self._frame(n_features=500, n_classes=2)
        scores, *_ = run_nested_cv(X, y, model_name, cat_cols=[])
        assert all(np.isnan(s) for s in scores)

    def test_class_limit_is_tabfm_only(self):
        """TabPFN-3 lifted the 10-class cap, and TabICL never had one."""
        assert _MODEL_LIMITS["tabfm"]["max_classes"] == 10
        assert "max_classes" not in _MODEL_LIMITS["tabicl"]
        assert "tabpfn3" not in _MODEL_LIMITS
