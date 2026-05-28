"""Build final (refitted) models from best Optuna hyperparameters."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from benchmark.encoding import (
    CAT_STRATEGY_TREE, CatBoostNativeWrapper, CatFeaturesEncoder,
    TabPFNNativeWrapper, _build_imputer, _build_scaler,
)
from config import N_JOBS, RANDOM_STATE


def build_final_model(model_name: str, best: dict, n_classes: int, cat_cols: list[str]):
    """Construct a fitted-ready pipeline from Optuna ``best_params``.

    Parameters
    ----------
    model_name : str
        One of ``"random_forest"``, ``"xgboost"``, ``"sgd"``,
        ``"lgbm"``, ``"lgbm_linear"``, ``"hgb"``, ``"catboost"``,
        ``"tabnet"``, ``"ft_transformer"``, ``"resnet"``.
    best : dict
        ``study.best_params`` returned by Optuna.
    n_classes : int
        Number of target classes (used to set model-specific objectives).
    cat_cols : list[str]
        Categorical column names (used by CatBoost wrapper).

    Returns
    -------
    Unfitted sklearn-compatible estimator / pipeline.
    """
    if model_name == "random_forest":
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("rf", RandomForestClassifier(**best, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ])

    if model_name == "xgboost":
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("xgb", XGBClassifier(**best, objective=objective,
                                  random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)),
        ])

    if model_name == "sgd":
        sgd_params = {k: v for k, v in best.items()
                      if k not in ("cat_strategy", "imputer", "scaler")}
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=best["cat_strategy"])),
            ("imputer", _build_imputer(best["imputer"])),
            ("scaler",  _build_scaler(best["scaler"])),
            ("sgd",     SGDClassifier(**sgd_params, max_iter=1000, tol=1e-3,
                                     random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ])

    if model_name == "lgbm":
        objective = "binary" if n_classes == 2 else "multiclass"
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("lgbm", LGBMClassifier(**best, objective=objective, subsample_freq=1,
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)),
        ])

    if model_name == "hgb":
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("hgb", HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE)),
        ])

    if model_name == "lgbm_linear":
        objective = "binary" if n_classes == 2 else "multiclass"
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("lgbm", LGBMClassifier(**best, linear_tree=True, objective=objective,
                                    subsample_freq=1, random_state=RANDOM_STATE,
                                    n_jobs=N_JOBS, verbose=-1)),
        ])

    if model_name == "tabnet":
        from benchmark.models.torch_wrappers import TabNetNativeWrapper
        return TabNetNativeWrapper(cat_cols=cat_cols, max_epochs=400, patience=20,
                                   **{k: v for k, v in best.items() if k != "cat_cols"})

    if model_name == "ft_transformer":
        from benchmark.models.torch_wrappers import FTTransformerWrapper
        return FTTransformerWrapper(cat_cols=cat_cols, max_epochs=400, patience=24,
                                    **{k: v for k, v in best.items() if k != "cat_cols"})

    if model_name == "resnet":
        from benchmark.models.torch_wrappers import ResNetWrapper
        return ResNetWrapper(cat_cols=cat_cols, max_epochs=400, patience=24,
                             **{k: v for k, v in best.items() if k != "cat_cols"})

    if model_name == "catboost":
        loss_function = "Logloss" if n_classes == 2 else "MultiClass"
        cb_params = dict(best)
        cb_params["bootstrap_type"] = "Bernoulli"  # required to use subsample
        return CatBoostNativeWrapper(
            cat_cols=cat_cols, loss_function=loss_function,
            random_state=RANDOM_STATE, verbose=0, thread_count=N_JOBS, **cb_params,
        )

    raise ValueError(f"Unknown model_name: {model_name!r}")
