"""Nested cross-validation runner with Optuna / GridSearchCV tuning."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from benchmark.metrics import pr_auc_score
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

# TabICL's docs claim up to ~2000 features, but empirically it stalls far
# earlier: arcene (10000 feats) ran 8h+ with no completion, cnae-9 (856 feats)
# stalled past 25min, both ballooning RSS to 5-10 GB. Skip wide datasets for
# TabICL — recorded as NaN. 500 catches the known offenders (madelon 500,
# multiple-features 649, har 561, cnae-9 856, micro-mass 1300, arcene 10000).
_TABICL_MAX_FEATURES = 500

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_nested_cv(
    X,
    y: np.ndarray,
    model_name: str,
    cat_cols: list[str],
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[dict]]:
    """Run nested cross-validation for *model_name*.

    Tuning strategy:
    - ``"svc"``, ``"logreg"``, ``"tabpfn"``, ``"tabicl"`` → GridSearchCV (small HP space)
    - all others                                         → Optuna TPE (``N_TRIALS`` trials per outer fold)

    Parameters
    ----------
    X        : DataFrame of shape (n_samples, n_features)
    y        : integer label array
    model_name : one of the supported model keys
    cat_cols : categorical column names in X

    Returns
    -------
    scores      : list of float — PR AUC per outer fold
    preds       : list of ndarray — predict_proba output per outer fold
    labels      : list of ndarray — true labels per outer fold
    best_params : list of dict — chosen hyperparameters per outer fold (e.g. SVC kernel)
    """
    if model_name == "tabicl" and X.shape[1] >= _TABICL_MAX_FEATURES:
        nan_scores = [float("nan")] * N_OUTER_FOLDS
        return nan_scores, None, None, None

    n_classes = int(np.unique(y).size)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    nested_scores: list[float] = []
    nested_preds: list[np.ndarray] = []
    nested_labels: list[np.ndarray] = []
    nested_best_params: list[dict] = []

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
            }
            n_trials = N_TRIALS_NN if model_name in _NN_MODELS else N_TRIALS
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            study.optimize(_objectives[model_name], n_trials=n_trials)
            best_params = dict(study.best_params)
            model = build_final_model(model_name, study.best_params, n_classes, cat_cols)
            model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)
        nested_scores.append(pr_auc_score(y_test, y_pred))
        nested_preds.append(y_pred)
        nested_labels.append(y_test)
        nested_best_params.append(best_params)

    return nested_scores, nested_preds, nested_labels, nested_best_params
