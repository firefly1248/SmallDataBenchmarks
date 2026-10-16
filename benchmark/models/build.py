"""Build final (refitted) models from best Optuna hyperparameters."""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from benchmark.encoding import (
    CAT_STRATEGY_TREE, CatBoostNativeWrapper, CatFeaturesEncoder, LowLevelEncoder,
    TabFMNativeWrapper, _build_imputer, _build_scaler,
)
from config import N_JOBS, RANDOM_STATE


def lgbm_pipeline(params: dict, n_classes: int, n_jobs: int, linear_tree: bool = False,
                  extra_trees: bool = False, low_levels: str | None = None) -> Pipeline:
    """LightGBM after ordinal encoding; ``low_levels`` re-encodes the few-valued
    columns (see ``LowLevelEncoder``)."""
    steps = [("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE))]
    if low_levels is not None:
        steps.append(("low_levels", LowLevelEncoder(mode=low_levels)))
    objective = "binary" if n_classes == 2 else "multiclass"
    steps.append(("lgbm", LGBMClassifier(**params, linear_tree=linear_tree, extra_trees=extra_trees,
                                         objective=objective, subsample_freq=1,
                                         random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1)))
    return Pipeline(steps)


def build_final_model(model_name: str, best: dict, n_classes: int, cat_cols: list[str]):
    """Construct an unfitted pipeline for *model_name* from Optuna ``best_params``,
    which is empty for the models that are not tuned.
    """
    if model_name == "random_forest":
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("rf", RandomForestClassifier(**best, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ])

    if model_name == "extra_trees":
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("et", ExtraTreesClassifier(**best, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ])

    # Every LightGBM variant: lgbm_linear, lgbm_extra, and their seed, tree-cap and
    # encoding variants, whose names carry the suffixes read here.
    if model_name.startswith("lgbm"):
        low_levels = {"onehot": "onehot", "cat": "category"}.get(model_name.rsplit("_", 1)[-1])
        return lgbm_pipeline(best, n_classes, N_JOBS, linear_tree="linear" in model_name,
                             extra_trees="extra" in model_name, low_levels=low_levels)

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

    if model_name == "hgb":
        return Pipeline([
            ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
            ("hgb", HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE)),
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

    if model_name == "tabfm":
        return TabFMNativeWrapper(cat_cols=cat_cols, random_state=RANDOM_STATE, **best)

    if model_name == "catboost":
        loss_function = "Logloss" if n_classes == 2 else "MultiClass"
        cb_params = dict(best)
        cb_params["bootstrap_type"] = "Bernoulli"  # required to use subsample
        return CatBoostNativeWrapper(
            cat_cols=cat_cols, loss_function=loss_function,
            random_state=RANDOM_STATE, verbose=0, thread_count=N_JOBS, **cb_params,
        )

    raise ValueError(f"Unknown model_name: {model_name!r}")
