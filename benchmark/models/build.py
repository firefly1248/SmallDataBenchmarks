"""Unfitted final models from Optuna ``best_params``: one builder per learner, all
``(best, n_classes, cat_cols)``; variants bind keyword arguments in the registry."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from benchmark.encoding import (
    CAT_STRATEGY_TREE, CatFeaturesEncoder, LowLevelEncoder, _build_imputer, _build_scaler,
)
from benchmark.models.wrappers import CatBoostNativeWrapper, TabFMNativeWrapper
from config import N_JOBS, RANDOM_STATE


def build_forest(best, n_classes, cat_cols, forest=RandomForestClassifier):
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("rf", forest(**best, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ])


def build_xgboost(best, n_classes, cat_cols):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("xgb", XGBClassifier(**best, objective=objective,
                              random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)),
    ])


def build_lgbm(best, n_classes, cat_cols, linear_tree=False, extra_trees=False,
               low_levels=None, n_jobs=N_JOBS):
    """LightGBM after ordinal encoding; ``low_levels`` re-encodes the few-valued
    columns (see ``LowLevelEncoder``)."""
    steps = [("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE))]
    if low_levels is not None:
        steps.append(("low_levels", LowLevelEncoder(mode=low_levels)))
    objective = "binary" if n_classes == 2 else "multiclass"
    steps.append(("lgbm", LGBMClassifier(**best, linear_tree=linear_tree, extra_trees=extra_trees,
                                         objective=objective, subsample_freq=1,
                                         random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1)))
    return Pipeline(steps)


def build_sgd(best, n_classes, cat_cols):
    sgd_params = {k: v for k, v in best.items() if k not in ("cat_strategy", "imputer", "scaler")}
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=best["cat_strategy"])),
        ("imputer", _build_imputer(best["imputer"])),
        ("scaler",  _build_scaler(best["scaler"])),
        ("sgd",     SGDClassifier(**sgd_params, max_iter=1000, tol=1e-3,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ])


def build_hgb(best, n_classes, cat_cols):
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("hgb", HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE)),
    ])


def build_catboost(best, n_classes, cat_cols):
    loss_function = "Logloss" if n_classes == 2 else "MultiClass"
    return CatBoostNativeWrapper(
        cat_cols=cat_cols, loss_function=loss_function, random_state=RANDOM_STATE, verbose=0,
        thread_count=N_JOBS, **best, bootstrap_type="Bernoulli",  # required to use subsample
    )


def build_tabfm(best, n_classes, cat_cols):
    return TabFMNativeWrapper(cat_cols=cat_cols, random_state=RANDOM_STATE, **best)


# Final fits train longer than the search did.
def build_tabnet(best, n_classes, cat_cols):
    from benchmark.models.torch_wrappers import TabNetNativeWrapper
    return TabNetNativeWrapper(cat_cols=cat_cols, max_epochs=400, patience=20,
                               **{k: v for k, v in best.items() if k != "cat_cols"})


def build_ft_transformer(best, n_classes, cat_cols):
    from benchmark.models.torch_wrappers import FTTransformerWrapper
    return FTTransformerWrapper(cat_cols=cat_cols, max_epochs=400, patience=24,
                                **{k: v for k, v in best.items() if k != "cat_cols"})


def build_resnet(best, n_classes, cat_cols):
    from benchmark.models.torch_wrappers import ResNetWrapper
    return ResNetWrapper(cat_cols=cat_cols, max_epochs=400, patience=24,
                         **{k: v for k, v in best.items() if k != "cat_cols"})
