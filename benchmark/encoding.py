"""Categorical encoding utilities for tabular benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import category_encoders as ce
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler,
)


# Strategies suitable for linear models (support y in fit for target-based)
CAT_STRATEGIES_LINEAR: list[str] = [
    "ordinal",
    "target",
    "james_stein",
    "m_estimate",
    "catboost_enc",
]

# Tree models handle ordering natively — ordinal encoding is enough
CAT_STRATEGY_TREE: str = "ordinal"

IMPUTER_STRATEGIES: list[str] = ["mean", "median", "most_frequent"]
SCALERS_LIST: list[str] = ["minmax", "standard", "robust", "maxabs", "power_yj"]

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


class CatFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical (object-dtype) columns; pass numeric columns through.

    Parameters
    ----------
    strategy : str
        One of ``"ordinal"``, ``"target"``, ``"james_stein"``,
        ``"m_estimate"``, ``"catboost_enc"``.
    """

    _ENCODER_MAP: dict = {
        "ordinal":      ce.OrdinalEncoder,
        "target":       ce.TargetEncoder,
        "james_stein":  ce.JamesSteinEncoder,
        "m_estimate":   ce.MEstimateEncoder,
        "catboost_enc": ce.CatBoostEncoder,
    }

    def __init__(self, strategy: str = "ordinal") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None) -> "CatFeaturesEncoder":
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._cat_cols: list[str] = X.select_dtypes(include="object").columns.tolist()
        if self._cat_cols:
            encoder_cls = self._ENCODER_MAP[self.strategy]
            self._encoder = encoder_cls(
                cols=self._cat_cols,
                handle_missing="value",
                handle_unknown="value",
            )
            self._encoder.fit(X, y)
        else:
            self._encoder = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self._encoder is not None:
            X = self._encoder.transform(X)
        return X


class CatBoostNativeWrapper(ClassifierMixin, BaseEstimator):
    """Wrap CatBoostClassifier to use native ``cat_features`` from DataFrame cols.

    Fills NaN in categorical columns with ``"missing"`` before training.
    ``ClassifierMixin`` must precede ``BaseEstimator`` in MRO so that
    sklearn 1.7's ``get_tags()`` API correctly identifies this as a classifier.

    Parameters
    ----------
    cat_cols : list[str]
        Names of categorical columns in the input DataFrame.
    **catboost_params
        Forwarded verbatim to ``CatBoostClassifier``.
    """

    def __init__(self, cat_cols: list[str], **catboost_params) -> None:
        self.cat_cols = cat_cols
        self.catboost_params = catboost_params

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna("missing").astype(str)
        return X

    def _cat_indices(self, X: pd.DataFrame) -> list[int]:
        return [i for i, col in enumerate(X.columns) if col in self.cat_cols]

    def fit(self, X, y):
        X = self._prepare(X)
        self._model = CatBoostClassifier(**self.catboost_params)
        self._model.fit(X, y, cat_features=self._cat_indices(X))
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(self._prepare(X))

    def predict(self, X) -> np.ndarray:
        return self._model.predict(self._prepare(X))

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {"cat_cols": self.cat_cols}
        params.update(self.catboost_params)
        return params

    def set_params(self, **params) -> "CatBoostNativeWrapper":
        self.cat_cols = params.pop("cat_cols", self.cat_cols)
        self.catboost_params.update(params)
        return self
