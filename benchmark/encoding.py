"""Categorical encoding utilities for tabular benchmarks."""
from __future__ import annotations

import torch  # must precede catboost to win the OpenMP init race
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


class TabPFNNativeWrapper(ClassifierMixin, BaseEstimator):
    """Wrap TabPFNClassifier to resolve categorical column names to indices at fit time.

    Parameters
    ----------
    cat_cols : list[str]
        Categorical column names (resolved to integer indices at fit time).
    n_estimators : int
        Ensemble size — higher is more accurate but slower.
    balance_probabilities : bool
        Balance predicted class probabilities.
    **tabpfn_params
        Forwarded verbatim to ``TabPFNClassifier``.
    """

    def __init__(
        self,
        cat_cols: list[str],
        n_estimators: int = 4,
        balance_probabilities: bool = False,
        **tabpfn_params,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.tabpfn_params = tabpfn_params

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna("missing").astype(str)
        return X

    def _cat_indices(self, X: pd.DataFrame) -> list[int]:
        cols = list(X.columns)
        return [cols.index(c) for c in self.cat_cols if c in cols]

    @staticmethod
    def _patch_tabpfn_auth() -> None:
        """Monkey-patch verify_token to retry on network timeout.

        api.priorlabs.ai/protected/ is occasionally slow; 3 retries with 30s
        timeout each prevents spurious TabPFNLicenseError in long benchmark runs.
        """
        import tabpfn.browser_auth as _ba  # noqa: PLC0415

        if getattr(_ba, "_verify_token_patched", False):
            return

        _orig = _ba.verify_token

        def _robust_verify(token: str, api_url: str) -> "bool | None":
            import urllib.request, urllib.error  # noqa: PLC0415
            url = f"{api_url.rstrip('/')}/protected/"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            for _ in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        return resp.status == 200
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 403):
                        return False
                    return None
                except Exception:
                    pass
            return None

        _ba.verify_token = _robust_verify
        _ba._verify_token_patched = True  # type: ignore[attr-defined]

    def fit(self, X, y):
        self._patch_tabpfn_auth()
        from tabpfn import TabPFNClassifier
        X = self._prepare(X)
        self._model = TabPFNClassifier(
            n_estimators=self.n_estimators,
            categorical_features_indices=self._cat_indices(X),
            balance_probabilities=self.balance_probabilities,
            ignore_pretraining_limits=True,
            n_preprocessing_jobs=1,
            device="cpu",
            **{k: v for k, v in self.tabpfn_params.items()
               if k not in ("n_estimators", "balance_probabilities",
                            "ignore_pretraining_limits", "n_jobs",
                            "n_preprocessing_jobs", "device")},
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(self._prepare(X))

    def predict(self, X) -> np.ndarray:
        return self._model.predict(self._prepare(X))

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {
            "cat_cols": self.cat_cols,
            "n_estimators": self.n_estimators,
            "balance_probabilities": self.balance_probabilities,
        }
        params.update(self.tabpfn_params)
        return params

    def set_params(self, **params) -> "TabPFNNativeWrapper":
        self.cat_cols = params.pop("cat_cols", self.cat_cols)
        self.n_estimators = params.pop("n_estimators", self.n_estimators)
        self.balance_probabilities = params.pop("balance_probabilities",
                                                self.balance_probabilities)
        self.tabpfn_params.update(params)
        return self


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
