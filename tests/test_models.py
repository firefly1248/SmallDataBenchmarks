"""Tests for benchmark.models.grid_search."""
from __future__ import annotations

import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from benchmark.models.grid_search import GRID_SEARCH_MODELS, build_grid_search


class TestBuildGridSearch:
    @pytest.fixture
    def inner_cv(self):
        return StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    @pytest.mark.parametrize("model_name", ["svc", "logreg"])
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
        assert proba.shape == (len(X), 3)  # iris has 3 classes

    def test_grid_search_models_constant(self):
        assert "svc" in GRID_SEARCH_MODELS
        assert "logreg" in GRID_SEARCH_MODELS
        assert "random_forest" not in GRID_SEARCH_MODELS
