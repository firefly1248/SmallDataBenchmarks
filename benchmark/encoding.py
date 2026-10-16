"""Categorical encoders for the tabular benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler,
)


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
    """Encode object-dtype columns via ``_ENCODER_MAP``; pass numerics through."""

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


class LowLevelEncoder(BaseEstimator, TransformerMixin):
    """Treat columns with at most ``max_levels`` distinct values as categorical:
    one-hot them (``mode="onehot"``), or give them a pandas category dtype, which
    LightGBM splits as categories (``mode="category"``). Levels are fixed at fit, so
    a level unseen in training is all zeros or NaN."""

    def __init__(self, mode: str = "onehot", max_levels: int = 8) -> None:
        self.mode = mode
        self.max_levels = max_levels

    def fit(self, X: pd.DataFrame, y=None) -> "LowLevelEncoder":
        self.levels_ = {c: np.sort(X[c].dropna().unique()) for c in X.columns
                        if X[c].nunique() <= self.max_levels}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "category":
            return X.assign(**{c: pd.Categorical(X[c], categories=levels)
                               for c, levels in self.levels_.items()})
        onehot = {f"{c}={v}": (X[c] == v).astype(float)
                  for c, levels in self.levels_.items() for v in levels}
        return pd.concat([X.drop(columns=list(self.levels_)), pd.DataFrame(onehot, index=X.index)],
                         axis=1)
