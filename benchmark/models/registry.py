"""Every model key the Optuna runner accepts: how it is tuned, built and limited.

A variant is its base with keyword arguments bound into the objective, the build,
or both, so adding one is one line here and nothing anywhere else.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial

from sklearn.ensemble import ExtraTreesClassifier

from benchmark.metrics import LOG_LOSS_SCORER, PR_AUC_SCORER
from benchmark.models import build as b
from benchmark.models import objectives as o
from benchmark.models.grid_search import build_grid_search
from config import N_TRIALS_NN, RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    build: Callable                     # (best_params, n_classes, cat_cols) -> unfitted model
    objective: Callable | None = None   # Optuna: (trial, X, y, inner_cv, n_classes, cat_cols) -> score
    grid: Callable | None = None        # GridSearchCV: (inner_cv, cat_cols) -> GridSearchCV
    # Datasets over these are recorded as NaN so the score arrays stay aligned. The
    # class caps raise in fit(); TabICL's feature cap is empirical, it stalls for
    # hours rather than failing. See FoundationModels_notes.md.
    limits: dict = field(default_factory=dict)
    n_trials: int | None = None         # None: config.N_TRIALS, read at run time so tests can shrink it
    sampler_seed: int = RANDOM_STATE


def build_final_model(model_name: str, best: dict, n_classes: int, cat_cols: list[str]):
    return MODELS[model_name].build(best, n_classes, cat_cols)


def _grid(key: str, **limits) -> ModelSpec:
    def build(best, n_classes, cat_cols):
        return build_grid_search(key, None, cat_cols).estimator.set_params(**best)
    return ModelSpec(build=build, grid=partial(build_grid_search, key), limits=limits)


def _lgbm(*, scoring=PR_AUC_SCORER, max_trees=500, sampler_seed=RANDOM_STATE, **model) -> ModelSpec:
    """``model`` reaches both the search and the final fit; the rest the search only."""
    return ModelSpec(build=partial(b.build_lgbm, **model),
                     objective=partial(o.lgbm_objective, scoring=scoring, max_trees=max_trees, **model),
                     sampler_seed=sampler_seed)


_xgboost = ModelSpec(build=b.build_xgboost, objective=o.xgb_objective)
_catboost = ModelSpec(build=b.build_catboost, objective=o.catboost_objective)

MODELS: dict[str, ModelSpec] = {
    "svc":           _grid("svc"),
    "logreg":        _grid("logreg"),
    "tabpfn3":       _grid("tabpfn3"),
    "tabpfn35":      _grid("tabpfn35"),
    "tabpfn35fast":  _grid("tabpfn35fast"),
    "tabicl":        _grid("tabicl", max_features=500),
    "tabfm":         ModelSpec(build=b.build_tabfm, limits={"max_features": 500, "max_classes": 10}),
    "random_forest": ModelSpec(build=b.build_forest, objective=o.rf_objective),
    "extra_trees":   ModelSpec(build=partial(b.build_forest, forest=ExtraTreesClassifier),
                               objective=partial(o.rf_objective, forest=ExtraTreesClassifier)),
    "xgboost":       _xgboost,
    "sgd":           ModelSpec(build=b.build_sgd, objective=o.sgd_objective),
    "catboost":      _catboost,
    "lgbm":          _lgbm(),
    "lgbm_linear":   _lgbm(linear_tree=True),
    "lgbm_extra":    _lgbm(extra_trees=True),
    "hgb":           ModelSpec(build=b.build_hgb, objective=o.hgb_objective),
    # The same models and search spaces, selected on log loss.
    "xgboost_logloss":  replace(_xgboost, objective=partial(o.xgb_objective, scoring=LOG_LOSS_SCORER)),
    "catboost_logloss": replace(_catboost, objective=partial(o.catboost_objective, scoring=LOG_LOSS_SCORER)),
    "lgbm_logloss":     _lgbm(scoring=LOG_LOSS_SCORER),
    # Controls for lgbm_extra: another sampler seed, a higher tree cap, and the
    # few-valued columns one-hot or as LightGBM categories.
    "lgbm_seed1":        _lgbm(sampler_seed=RANDOM_STATE + 1),
    "lgbm_2000":         _lgbm(max_trees=2000),
    "lgbm_extra_2000":   _lgbm(extra_trees=True, max_trees=2000),
    "lgbm_onehot":       _lgbm(low_levels="onehot"),
    "lgbm_cat":          _lgbm(low_levels="category"),
    "lgbm_extra_onehot": _lgbm(extra_trees=True, low_levels="onehot"),
    "lgbm_extra_cat":    _lgbm(extra_trees=True, low_levels="category"),
    "tabnet":         ModelSpec(build=b.build_tabnet, objective=o.tabnet_objective, n_trials=N_TRIALS_NN),
    "ft_transformer": ModelSpec(build=b.build_ft_transformer, objective=o.ft_transformer_objective,
                                n_trials=N_TRIALS_NN),
    "resnet":         ModelSpec(build=b.build_resnet, objective=o.resnet_objective, n_trials=N_TRIALS_NN),
}
