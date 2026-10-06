"""Nested cross-validation runner with Optuna / GridSearchCV tuning."""
from __future__ import annotations

import os

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from benchmark.metrics import LOG_LOSS_SCORER, pr_auc_score
from benchmark.models.build import build_final_model
from benchmark.models.grid_search import GRID_SEARCH_MODELS, build_grid_search
from benchmark.models.objectives import (
    catboost_objective,
    ft_transformer_objective,
    hgb_objective,
    lgbm_objective,
    resnet_objective,
    rf_objective,
    sgd_objective,
    tabnet_objective,
    xgb_objective,
)
from config import N_INNER_FOLDS, N_JOBS, N_OUTER_FOLDS, N_TRIALS, N_TRIALS_NN, RANDOM_STATE

_NN_MODELS: frozenset[str] = frozenset({"tabnet", "ft_transformer", "resnet"})

# Datasets over these limits are recorded as NaN so the score arrays stay
# aligned. The class caps raise in fit(); TabICL's feature cap is empirical —
# it stalls for hours rather than failing. See FoundationModels_notes.md.
_MODEL_LIMITS: dict[str, dict[str, int]] = {
    "tabicl": {"max_features": 500},
    "tabfm":  {"max_features": 500, "max_classes": 10},
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One SQLite file per model, one study per (dataset, outer fold). A study lives only
# while its dataset is unfinished: optuna_models.py drops it once the dataset is
# checkpointed, so what is left is the progress of a run that was cut off.
STUDY_DIR = "results/optuna"


def _storage(model_name: str) -> str:
    return f"sqlite:///{STUDY_DIR}/{model_name}.db"


def drop_studies(model_name: str, dataset_name: str) -> None:
    if not os.path.exists(f"{STUDY_DIR}/{model_name}.db"):
        return
    storage = _storage(model_name)
    for name in optuna.study.get_all_study_names(storage):
        if name.startswith(f"{dataset_name}/"):
            optuna.delete_study(study_name=name, storage=storage)


def run_nested_cv(
    X,
    y: np.ndarray,
    model_name: str,
    cat_cols: list[str],
    dataset_name: str | None = None,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[dict]]:
    """Run nested cross-validation for *model_name*.

    Tuning: ``"tabfm"`` none; ``GRID_SEARCH_MODELS`` by GridSearchCV;
    everything else by Optuna TPE.

    With ``dataset_name``, Optuna studies are stored in ``STUDY_DIR`` and a rerun
    resumes each outer fold from its finished trials. After a code change, delete
    the model's database first, or the old trials are reused.

    Returns ``(scores, preds, labels, best_params)``, one entry per outer fold.
    """
    n_classes = int(np.unique(y).size)

    limits = _MODEL_LIMITS.get(model_name, {})
    if (X.shape[1] >= limits.get("max_features", float("inf"))
            or n_classes > limits.get("max_classes", float("inf"))):
        nan_scores = [float("nan")] * N_OUTER_FOLDS
        return nan_scores, None, None, None

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    nested_scores: list[float] = []
    nested_preds: list[np.ndarray] = []
    nested_labels: list[np.ndarray] = []
    nested_best_params: list[dict] = []
    storage = None
    if dataset_name is not None:
        os.makedirs(STUDY_DIR, exist_ok=True)
        storage = _storage(model_name)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        y_test  = y[test_idx]

        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)

        if model_name == "tabfm":
            model = build_final_model(model_name, {}, n_classes, cat_cols)
            model.fit(X_train, y_train)
            best_params = {}
        elif model_name in GRID_SEARCH_MODELS:
            gs = build_grid_search(model_name, inner_cv, cat_cols)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = dict(gs.best_params_)
        else:
            _objectives = {
                "random_forest": lambda t: rf_objective(t, X_train, y_train, inner_cv),
                "xgboost":       lambda t: xgb_objective(t, X_train, y_train,
                                                          inner_cv, n_classes),
                "sgd":           lambda t: sgd_objective(t, X_train, y_train, inner_cv),
                "catboost":      lambda t: catboost_objective(t, X_train, y_train,
                                                               inner_cv, n_classes, cat_cols),
                "lgbm":          lambda t: lgbm_objective(t, X_train, y_train,
                                                           inner_cv, n_classes),
                "lgbm_linear":   lambda t: lgbm_objective(t, X_train, y_train,
                                                           inner_cv, n_classes, linear_tree=True),
                "hgb":           lambda t: hgb_objective(t, X_train, y_train, inner_cv),
                "tabnet":        lambda t: tabnet_objective(t, X_train, y_train,
                                                             inner_cv, cat_cols),
                "ft_transformer": lambda t: ft_transformer_objective(t, X_train, y_train,
                                                                       inner_cv, cat_cols),
                "resnet":        lambda t: resnet_objective(t, X_train, y_train,
                                                             inner_cv, cat_cols),
                # The same models and search spaces, selected on log loss.
                "xgboost_logloss":  lambda t: xgb_objective(t, X_train, y_train, inner_cv,
                                                             n_classes, scoring=LOG_LOSS_SCORER),
                "catboost_logloss": lambda t: catboost_objective(t, X_train, y_train, inner_cv,
                                                                  n_classes, cat_cols,
                                                                  scoring=LOG_LOSS_SCORER),
                "lgbm_logloss":     lambda t: lgbm_objective(t, X_train, y_train, inner_cv,
                                                              n_classes, scoring=LOG_LOSS_SCORER),
            }
            n_trials = N_TRIALS_NN if model_name in _NN_MODELS else N_TRIALS
            study = optuna.create_study(
                study_name=None if storage is None else f"{dataset_name}/fold{fold}",
                storage=storage,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            # Failed trials count, as they do in an uninterrupted run; a trial cut
            # off mid-fit stays RUNNING and is redone.
            finished = sum(t.state.is_finished() for t in study.trials)
            # One raising trial must not abort the dataset.
            study.optimize(_objectives[model_name], n_trials=max(n_trials - finished, 0),
                           catch=(Exception,))
            best_params = dict(study.best_params)
            model = build_final_model(model_name.removesuffix("_logloss"), study.best_params,
                                      n_classes, cat_cols)
            model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)
        nested_scores.append(pr_auc_score(y_test, y_pred))
        nested_preds.append(y_pred)
        nested_labels.append(y_test)
        nested_best_params.append(best_params)

    return nested_scores, nested_preds, nested_labels, nested_best_params
