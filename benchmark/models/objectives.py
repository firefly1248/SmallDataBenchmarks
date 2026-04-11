"""Optuna hyperparameter search objectives for each model."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from benchmark.encoding import CatFeaturesEncoder, CatBoostNativeWrapper, CAT_STRATEGY_TREE
from benchmark.metrics import PR_AUC_SCORER
from config import N_JOBS, RANDOM_STATE

# Preprocessing choices for linear models
CAT_STRATEGIES_LINEAR: list[str] = ["ordinal", "target", "james_stein", "m_estimate", "catboost_enc"]
IMPUTER_STRATEGIES: list[str] = ["mean", "median", "most_frequent"]
SCALERS_LIST: list[str] = ["minmax", "standard", "robust", "maxabs", "power_yj"]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler,
)

_SCALERS = {
    "minmax":   MinMaxScaler,
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "maxabs":   MaxAbsScaler,
    "power_yj": lambda: PowerTransformer(method="yeo-johnson"),
}


def _build_scaler(name: str):
    return _SCALERS[name]()


def _build_imputer(strategy: str) -> SimpleImputer:
    return SimpleImputer(strategy=strategy)


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
        ("rf", RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=1)),
    ])
    return float(np.mean(cross_val_score(model, X_train, y_train,
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
    return float(np.mean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def sgd_objective(trial, X_train, y_train, inner_cv) -> float:
    cat_strategy = trial.suggest_categorical("cat_strategy", CAT_STRATEGIES_LINEAR)
    imputer_strategy = trial.suggest_categorical("imputer", IMPUTER_STRATEGIES)
    scaler_name = trial.suggest_categorical("scaler", SCALERS_LIST)
    params = {
        "loss":          trial.suggest_categorical("loss", ["modified_huber", "log_loss"]),
        "alpha":         trial.suggest_float("alpha", 1e-6, 1e2, log=True),
        "l1_ratio":      trial.suggest_float("l1_ratio", 0.0, 1.0),
        "penalty":       trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        # 'optimal' removed: produces NaN when alpha is very small
        "learning_rate": trial.suggest_categorical("learning_rate",
                                                   ["constant", "invscaling", "adaptive"]),
        "eta0":          trial.suggest_float("eta0", 1e-4, 1.0, log=True),
        "class_weight":  trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=cat_strategy)),
        ("imputer", _build_imputer(imputer_strategy)),
        ("scaler",  _build_scaler(scaler_name)),
        ("sgd",     SGDClassifier(**params, max_iter=1000, tol=1e-3,
                                  random_state=RANDOM_STATE, n_jobs=1)),
    ])
    return float(np.mean(cross_val_score(model, X_train, y_train,
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
        random_state=RANDOM_STATE, verbose=0, thread_count=1, **params,
    )
    return float(np.mean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def lgbm_objective(trial, X_train, y_train, inner_cv, n_classes: int) -> float:
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "num_leaves":       trial.suggest_int("num_leaves", 8, 256),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight":     trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    objective = "binary" if n_classes == 2 else "multiclass"
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("lgbm", LGBMClassifier(**params, objective=objective, subsample_freq=1,
                                random_state=RANDOM_STATE, n_jobs=1, verbose=-1)),
    ])
    return float(np.mean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))


def lgbm_linear_tree_objective(trial, X_train, y_train, inner_cv, n_classes: int) -> float:
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "num_leaves":       trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "linear_lambda":    trial.suggest_float("linear_lambda", 1e-4, 10.0, log=True),
        "class_weight":     trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    objective = "binary" if n_classes == 2 else "multiclass"
    model = Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("lgbm", LGBMClassifier(**params, linear_tree=True, objective=objective,
                                subsample_freq=1, random_state=RANDOM_STATE,
                                n_jobs=1, verbose=-1)),
    ])
    return float(np.mean(cross_val_score(model, X_train, y_train,
                                         cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=N_JOBS)))
