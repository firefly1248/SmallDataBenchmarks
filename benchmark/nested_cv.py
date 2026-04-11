"""Nested cross-validation runner with Optuna / GridSearchCV tuning."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from benchmark.metrics import pr_auc_score
from benchmark.models.build import build_final_model
from benchmark.models.grid_search import GRID_SEARCH_MODELS, build_grid_search
from benchmark.models.objectives import (
    catboost_objective,
    lgbm_linear_tree_objective,
    lgbm_objective,
    rf_objective,
    sgd_objective,
    xgb_objective,
)
from config import N_INNER_FOLDS, N_JOBS, N_OUTER_FOLDS, N_TRIALS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_nested_cv(
    X,
    y: np.ndarray,
    model_name: str,
    cat_cols: list[str],
) -> list[float]:
    """Run nested cross-validation for *model_name* and return outer CV scores.

    Tuning strategy:
    - ``"svc"``, ``"logreg"`` → GridSearchCV (small HP space)
    - all others              → Optuna TPE (``N_TRIALS`` trials per outer fold)

    Parameters
    ----------
    X        : DataFrame of shape (n_samples, n_features)
    y        : integer label array
    model_name : one of the supported model keys
    cat_cols : categorical column names in X

    Returns
    -------
    list of float — PR AUC score for each outer fold
    """
    n_classes = int(np.unique(y).size)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    nested_scores: list[float] = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        y_test  = y[test_idx]

        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)

        if model_name in GRID_SEARCH_MODELS:
            gs = build_grid_search(model_name, inner_cv, cat_cols)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
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
                "lgbm_linear":   lambda t: lgbm_linear_tree_objective(t, X_train, y_train,
                                                                        inner_cv, n_classes),
            }
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            study.optimize(_objectives[model_name], n_trials=N_TRIALS)
            model = build_final_model(model_name, study.best_params, n_classes, cat_cols)
            model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)
        nested_scores.append(pr_auc_score(y_test, y_pred))

    return nested_scores
