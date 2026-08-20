"""Optuna hyperparameter search objectives for each model."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from benchmark.encoding import (
    CAT_STRATEGIES_LINEAR, CAT_STRATEGY_TREE,
    CatBoostNativeWrapper, CatFeaturesEncoder,
    IMPUTER_STRATEGIES, SCALERS_LIST,
    _build_imputer, _build_scaler,
)
from benchmark.metrics import PR_AUC_SCORER
from config import INNER_FIT_THREADS, N_JOBS, RANDOM_STATE


def rf_objective(trial, X_train, y_train, inner_cv) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("rf", RandomForestClassifier(**params, random_state=RANDOM_STATE,
                                      n_jobs=INNER_FIT_THREADS)),
    ])
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def xgb_objective(trial, X_train, y_train, inner_cv, n_classes: int) -> float:
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 2, 10),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma":            trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("xgb", XGBClassifier(**params, objective=objective,
                              random_state=RANDOM_STATE, n_jobs=1, verbosity=0)),
    ])
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def sgd_objective(trial, X_train, y_train, inner_cv) -> float:
    cat_strategy = trial.suggest_categorical("cat_strategy", CAT_STRATEGIES_LINEAR)
    imputer_strategy = trial.suggest_categorical("imputer", IMPUTER_STRATEGIES)
    scaler_name = trial.suggest_categorical("scaler", SCALERS_LIST)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    params = {
        "loss":          trial.suggest_categorical("loss", ["modified_huber", "log_loss"]),
        "alpha":         trial.suggest_float("alpha", 1e-6, 1e2, log=True),
        "penalty":       penalty,
        # 'optimal' removed: produces NaN when alpha is very small
        "learning_rate": trial.suggest_categorical("learning_rate",
                                                   ["constant", "invscaling", "adaptive"]),
        "eta0":          trial.suggest_float("eta0", 1e-4, 1.0, log=True),
        "class_weight":  trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    # l1_ratio is only meaningful for elasticnet; skip for l1/l2 to avoid polluting the search space
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=cat_strategy)),
        ("imputer", _build_imputer(imputer_strategy)),
        ("scaler",  _build_scaler(scaler_name)),
        ("sgd",     SGDClassifier(**params, max_iter=1000, tol=1e-3,
                                  random_state=RANDOM_STATE, n_jobs=1)),
    ])
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def catboost_objective(trial, X_train, y_train, inner_cv, n_classes: int,
                       cat_cols: list[str]) -> float:
    params = {
        "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":              trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "n_estimators":       trial.suggest_int("n_estimators", 50, 500),
        "bootstrap_type":     "Bernoulli",  # required to use subsample
        "subsample":          trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf":   trial.suggest_int("min_data_in_leaf", 1, 20),
        "random_strength":    trial.suggest_float("random_strength", 0.0, 10.0),
    }
    loss_function = "Logloss" if n_classes == 2 else "MultiClass"
    model = CatBoostNativeWrapper(
        cat_cols=cat_cols, loss_function=loss_function,
        random_state=RANDOM_STATE, verbose=0, thread_count=INNER_FIT_THREADS, **params,
    )
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def tabnet_objective(trial, X_train, y_train, inner_cv, cat_cols: list[str]) -> float:
    from benchmark.models.torch_wrappers import TabNetNativeWrapper
    params = {
        "n_d":           trial.suggest_int("n_d", 8, 64, step=8),
        "n_a":           trial.suggest_int("n_a", 8, 64, step=8),
        "n_steps":       trial.suggest_int("n_steps", 3, 10),
        "gamma":         trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "mask_type":     trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "lr":            trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "n_independent": trial.suggest_int("n_independent", 1, 4),
        "n_shared":      trial.suggest_int("n_shared", 1, 4),
    }
    model = TabNetNativeWrapper(cat_cols=cat_cols, max_epochs=200, patience=15, **params)
    # n_jobs=1: avoid nested PyTorch thread parallelism
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                          cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=1)))


def ft_transformer_objective(trial, X_train, y_train, inner_cv,
                              cat_cols: list[str]) -> float:
    from benchmark.models.torch_wrappers import FTTransformerWrapper
    d_token = trial.suggest_categorical("d_token", [64, 96, 128, 192, 256])
    params = {
        "d_token":            d_token,
        "n_blocks":           trial.suggest_int("n_blocks", 1, 6),
        "attention_dropout":  trial.suggest_float("attention_dropout", 0.0, 0.5),
        "ffn_d_hidden":       trial.suggest_int("ffn_d_hidden", 64, 512, step=32),
        "ffn_dropout":        trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "residual_dropout":   trial.suggest_float("residual_dropout", 0.0, 0.3),
        "lr":                 trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay":       trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    model = FTTransformerWrapper(cat_cols=cat_cols, max_epochs=200, patience=16, **params)
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                          cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=1)))


def resnet_objective(trial, X_train, y_train, inner_cv, cat_cols: list[str]) -> float:
    from benchmark.models.torch_wrappers import ResNetWrapper
    params = {
        "n_blocks":        trial.suggest_int("n_blocks", 1, 8),
        "d_main":          trial.suggest_int("d_main", 64, 512, step=32),
        "d_hidden":        trial.suggest_int("d_hidden", 64, 512, step=32),
        "dropout_first":   trial.suggest_float("dropout_first", 0.0, 0.5),
        "dropout_second":  trial.suggest_float("dropout_second", 0.0, 0.3),
        "lr":              trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    model = ResNetWrapper(cat_cols=cat_cols, max_epochs=200, patience=16, **params)
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                          cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=1)))


def hgb_objective(trial, X_train, y_train, inner_cv) -> float:
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_iter":          trial.suggest_int("max_iter", 50, 500),
        "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("hgb", HistGradientBoostingClassifier(**params, random_state=RANDOM_STATE)),
    ])
    # n_jobs=1: HGB has no n_jobs parameter and uses OMP internally; joblib-parallel
    # outer folds × OMP threads deadlocks libomp on macOS. Serial folds, each fit
    # gets the full OMP thread budget.
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=1)))


def lgbm_objective(trial, X_train, y_train, inner_cv, n_classes: int,
                   linear_tree: bool = False) -> float:
    max_leaves = 128 if linear_tree else 256
    max_child  = 50  if linear_tree else 100
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "num_leaves":       trial.suggest_int("num_leaves", 8, max_leaves),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, max_child),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight":     trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    if linear_tree:
        params["linear_lambda"] = trial.suggest_float("linear_lambda", 1e-4, 10.0, log=True)
    objective = "binary" if n_classes == 2 else "multiclass"
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("lgbm", LGBMClassifier(**params, linear_tree=linear_tree, objective=objective,
                                subsample_freq=1, random_state=RANDOM_STATE,
                                n_jobs=1, verbose=-1)),
    ])
    return float(np.nanmean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))
