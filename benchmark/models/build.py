"""Unfitted final models from Optuna ``best_params``: one builder per learner, all
``(best, n_classes, cat_cols)``; variants bind keyword arguments in the registry."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
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


def build_xgboost(best, n_classes, cat_cols, dart=False, n_jobs=N_JOBS):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("xgb", XGBClassifier(**best, objective=objective, booster="dart" if dart else "gbtree",
                              random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0)),
    ])


def build_lgbm(best, n_classes, cat_cols, linear_tree=False, extra_trees=False,
               low_levels=None, dart=False, n_jobs=N_JOBS):
    """LightGBM after ordinal encoding; ``low_levels`` re-encodes the few-valued
    columns (see ``LowLevelEncoder``)."""
    steps = [("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE))]
    if low_levels is not None:
        steps.append(("low_levels", LowLevelEncoder(mode=low_levels)))
    objective = "binary" if n_classes == 2 else "multiclass"
    # max_drop=-1: LightGBM's default caps the drop at 50 trees per iteration, which
    # would flatten drop_rate above 50 / n_estimators.
    booster = {"boosting_type": "dart", "max_drop": -1} if dart else {"boosting_type": "gbdt"}
    # ConfOpt's categories cannot hold None, so its search spells it "none".
    if best.get("class_weight") == "none":
        best = {**best, "class_weight": None}
    steps.append(("lgbm", LGBMClassifier(**best, **booster, linear_tree=linear_tree, extra_trees=extra_trees,
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


class SingleThreadHGB(HistGradientBoostingClassifier):
    """Two OpenMP runtimes are loaded (torch's and sklearn's); with both, HGB's own
    threads stall on small data and a fit runs 30x slower. One thread, same result."""

    def fit(self, X, y, **kwargs):
        with threadpool_limits(1, "openmp"):
            return super().fit(X, y, **kwargs)

    def predict_proba(self, X):
        with threadpool_limits(1, "openmp"):
            return super().predict_proba(X)


def build_hgb(best, n_classes, cat_cols):
    return Pipeline([
        ("cat_enc", CatFeaturesEncoder(strategy=CAT_STRATEGY_TREE)),
        ("hgb", SingleThreadHGB(**best, random_state=RANDOM_STATE)),
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
