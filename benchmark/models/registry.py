"""Every model key the Optuna runner accepts: how it is tuned, built and limited.

A variant is its base with keyword arguments bound into the objective, the build,
or both, so adding one is one line here and nothing anywhere else.
"""
from __future__ import annotations

import random
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache, partial

import numpy as np
import optuna
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
    sampler: Callable = optuna.samplers.TPESampler   # (seed=...) -> Optuna sampler
    # Tune the outer folds in concurrent threads. Only for objectives whose search fits
    # run on one thread each (4 folds x 4 inner fits fill the cores): thread settings
    # are process-wide, so a multi-threaded fit, or HGB's threadpool limit, would
    # oversubscribe the cores or leak into the other folds.
    parallel_folds: bool = False


def build_final_model(model_name: str, best: dict, n_classes: int, cat_cols: list[str]):
    return MODELS[model_name].build(best, n_classes, cat_cols)


def _grid(key: str, **limits) -> ModelSpec:
    def build(best, n_classes, cat_cols):
        return build_grid_search(key, None, cat_cols).estimator.set_params(**best)
    return ModelSpec(build=build, grid=partial(build_grid_search, key), limits=limits)


def _lgbm(*, scoring=PR_AUC_SCORER, max_trees=500, max_lr=0.3, sampler_seed=RANDOM_STATE, **model) -> ModelSpec:
    """``model`` reaches both the search and the final fit; the rest the search only."""
    return ModelSpec(build=partial(b.build_lgbm, **model),
                     objective=partial(o.lgbm_objective, scoring=scoring, max_trees=max_trees, max_lr=max_lr, **model),
                     sampler_seed=sampler_seed, parallel_folds=True)


def _xgb(*, scoring=PR_AUC_SCORER, sampler_seed=RANDOM_STATE, **model) -> ModelSpec:
    return ModelSpec(build=partial(b.build_xgboost, **model),
                     objective=partial(o.xgb_objective, scoring=scoring, **model),
                     sampler_seed=sampler_seed, parallel_folds=True)


_catboost = ModelSpec(build=b.build_catboost, objective=o.catboost_objective)

# The OptunaHub ConfOpt sampler is fetched from GitHub once, then cached; the commit is pinned.
CONFOPT_REF = "180d82259844f17578f4c3d0f8472835c17ff8cb"
# ConfOpt's categories cannot hold None; build_lgbm reads "none" as None.
_CONFOPT_CLASS_WEIGHTS = ("balanced", "none")
_confopt_lock = threading.Lock()


@cache
def _confopt_class():
    import optunahub
    base = optunahub.load_module("samplers/confopt_sampler", ref=CONFOPT_REF).ConfOptSampler

    class Reseeded(base):
        """ConfOpt draws its candidates and fits its surrogate from the global random
        and numpy generators, seeded once at construction. Reseeding them per trial from
        the seed and the finished-trial count, under one lock, makes each draw a function
        of the stored trials alone: a resumed study and concurrent folds then search as an
        uninterrupted sequential run does."""

        def __init__(self, search_space, seed):
            super().__init__(search_space=search_space, seed=seed)
            self._seed = seed

        def sample_relative(self, study, trial, search_space):
            with _confopt_lock:
                draw = self._seed * 10_000 + sum(t.state.is_finished() for t in study.trials)
                np.random.seed(draw)
                random.seed(draw)
                return super().sample_relative(study, trial, search_space)

    return Reseeded


def _confopt(space: dict, seed: int):
    return _confopt_class()(search_space=space, seed=seed)


def _lgbm_weights(class_weights: tuple, **kw) -> ModelSpec:
    """``_lgbm`` with the class_weight choices of its search narrowed or respelled."""
    spec = _lgbm(**kw)
    return replace(spec, objective=partial(spec.objective, class_weights=class_weights))


def _lgbm_confopt(class_weights: tuple = _CONFOPT_CLASS_WEIGHTS, **kw) -> ModelSpec:
    """LightGBM's search with conformal quantile regression (ConfOpt) in place of TPE,
    same space and budget."""
    spec = _lgbm_weights(class_weights, **kw)
    return replace(spec, sampler=partial(_confopt, o.lgbm_space(class_weights=class_weights)))


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
    "xgboost":       _xgb(),
    "sgd":           ModelSpec(build=b.build_sgd, objective=o.sgd_objective),
    "catboost":      _catboost,
    "lgbm":          _lgbm(),
    "lgbm_linear":   _lgbm(linear_tree=True),
    "lgbm_extra":    _lgbm(extra_trees=True),
    "hgb":           ModelSpec(build=b.build_hgb, objective=o.hgb_objective),
    # The same models and search spaces, selected on log loss.
    "xgboost_logloss":  _xgb(scoring=LOG_LOSS_SCORER),
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
    # Regularisation block: DART with its drop rates tuned, and the XGBoost search-noise control.
    "xgboost_seed1":     _xgb(sampler_seed=RANDOM_STATE + 1),
    "lgbm_dart":         _lgbm(dart=True),
    # DART checks from the review: its own sampler seed, and a learning-rate cap of 1 for both boosters.
    "lgbm_dart_seed1":   _lgbm(dart=True, sampler_seed=RANDOM_STATE + 1),
    "lgbm_wide":         _lgbm(max_lr=1.0),
    "lgbm_dart_wide":    _lgbm(dart=True, max_lr=1.0),
    "xgboost_dart":      _xgb(dart=True),
    # DART selected on log loss: is its worse calibration the PR AUC selection or DART itself?
    "lgbm_dart_logloss": _lgbm(dart=True, scoring=LOG_LOSS_SCORER),
    "lgbm_confopt":      _lgbm_confopt(),
    # lgbm again on the current library versions and parallel folds: the TPE counterpart
    # of lgbm_confopt and the 50-trial run of lgbm_trials100.
    "lgbm_rerun":        _lgbm(),
    # ConfOpt with another sampler seed: its own search noise, as lgbm_seed1 gives TPE's.
    "lgbm_confopt_seed1": _lgbm_confopt(sampler_seed=RANDOM_STATE + 1),
    # The search budget doubled: ConfOpt's conformal step runs 67 trials instead of 17.
    "lgbm_trials100":         replace(_lgbm(), n_trials=100),
    "lgbm_confopt_trials100": replace(_lgbm_confopt(), n_trials=100),
    # Log loss with class weighting off, for both samplers: a check on ConfOpt's calibration
    # loss under PR AUC selection, chosen after those runs.
    "lgbm_logloss_unweighted":         _lgbm_weights((None,), scoring=LOG_LOSS_SCORER),
    "lgbm_confopt_logloss_unweighted": _lgbm_confopt(("none",), scoring=LOG_LOSS_SCORER),
    # The log-loss and 100-trial pairs again with sampler seed 1, as the PR AUC pair has it.
    "lgbm_logloss_unweighted_seed1":         _lgbm_weights((None,), scoring=LOG_LOSS_SCORER,
                                                           sampler_seed=RANDOM_STATE + 1),
    "lgbm_confopt_logloss_unweighted_seed1": _lgbm_confopt(("none",), scoring=LOG_LOSS_SCORER,
                                                           sampler_seed=RANDOM_STATE + 1),
    "lgbm_trials100_seed1":         replace(_lgbm(sampler_seed=RANDOM_STATE + 1), n_trials=100),
    "lgbm_confopt_trials100_seed1": replace(_lgbm_confopt(sampler_seed=RANDOM_STATE + 1), n_trials=100),
    "tabnet":         ModelSpec(build=b.build_tabnet, objective=o.tabnet_objective, n_trials=N_TRIALS_NN),
    "ft_transformer": ModelSpec(build=b.build_ft_transformer, objective=o.ft_transformer_objective,
                                n_trials=N_TRIALS_NN),
    "resnet":         ModelSpec(build=b.build_resnet, objective=o.resnet_objective, n_trials=N_TRIALS_NN),
}
