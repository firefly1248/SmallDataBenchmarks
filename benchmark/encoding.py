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
    model_version : str, default ``"v2.6"``
        Checkpoint generation, one of ``ModelVersion``'s values. Pinned rather
        than left at the package default so that the ``tabpfn`` benchmark
        results stay reproducible after the package upgraded its default to
        v3; the ``tabpfn3`` model key passes ``"v3"``.
    auto_scale_n_estimators : bool, default False
        Introduced in tabpfn 8.x and on by default there: raises the *effective*
        ensemble size to cover all features on wide datasets, while leaving
        ``n_estimators`` — and therefore the reported ``best_params`` — at the
        requested value. Off by default because the param did not exist in 7.1.1
        when the ``tabpfn`` results were measured; the ``tabpfn3`` key turns it
        back on, since for a fresh measurement the library default is the
        behaviour a user would actually get.
    device : str, default ``"cpu"``
        Forwarded to ``TabPFNClassifier``.
    **tabpfn_params
        Forwarded verbatim to ``TabPFNClassifier``.
    """

    def __init__(
        self,
        cat_cols: list[str],
        n_estimators: int = 4,
        balance_probabilities: bool = False,
        model_version: str = "v2.6",
        auto_scale_n_estimators: bool = False,
        device: str = "cpu",
        **tabpfn_params,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.model_version = model_version
        self.auto_scale_n_estimators = auto_scale_n_estimators
        self.device = device
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
        from tabpfn.constants import ModelVersion
        X = self._prepare(X)
        self._model = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version),
            n_estimators=self.n_estimators,
            categorical_features_indices=self._cat_indices(X),
            balance_probabilities=self.balance_probabilities,
            ignore_pretraining_limits=True,
            auto_scale_n_estimators=self.auto_scale_n_estimators,
            n_preprocessing_jobs=1,
            device=self.device,
            **{k: v for k, v in self.tabpfn_params.items()
               if k not in ("n_estimators", "balance_probabilities",
                            "ignore_pretraining_limits", "auto_scale_n_estimators",
                            "n_jobs", "n_preprocessing_jobs", "device",
                            "model_version")},
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
            "model_version": self.model_version,
            "auto_scale_n_estimators": self.auto_scale_n_estimators,
            "device": self.device,
        }
        params.update(self.tabpfn_params)
        return params

    def set_params(self, **params) -> "TabPFNNativeWrapper":
        self.cat_cols = params.pop("cat_cols", self.cat_cols)
        self.n_estimators = params.pop("n_estimators", self.n_estimators)
        self.balance_probabilities = params.pop("balance_probabilities",
                                                self.balance_probabilities)
        self.model_version = params.pop("model_version", self.model_version)
        self.auto_scale_n_estimators = params.pop("auto_scale_n_estimators",
                                                  self.auto_scale_n_estimators)
        self.device = params.pop("device", self.device)
        self.tabpfn_params.update(params)
        return self


class TabICLNativeWrapper(ClassifierMixin, BaseEstimator):
    """Wrap TabICLClassifier.

    TabICL has its own internal preprocessor (OrdinalEncoder for cats, imputer
    for nums); we only forward the DataFrame as-is and let it auto-detect.

    Default device is CPU: MPS crashes (SIGSEGV) when the benchmark runs in a
    detached background process on Apple Silicon, even though it works fine in
    interactive sessions. Override with ``device="mps"`` from interactive use.

    Parameters
    ----------
    cat_cols : list[str]
        Unused — kept for signature parity with other native wrappers. TabICL
        auto-detects categorical columns from pandas dtype.
    n_estimators : int
        Ensemble size.
    softmax_temperature : float
        Output temperature; <1.0 sharpens, >1.0 softens predicted probabilities.
    device : str, default ``"cpu"``
        Forwarded to ``TabICLClassifier``.
    batch_size : int, default 4
        Ensemble members processed together. Lower than the TabICL default of 8
        to cap peak memory on this 24 GB machine (memory is the binding
        constraint, not compute); does not affect predictions.
    **tabicl_params
        Forwarded verbatim to ``TabICLClassifier``.
    """

    def __init__(
        self,
        cat_cols: list[str],
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        device: str = "cpu",
        batch_size: int = 4,
        **tabicl_params,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.device = device
        self.batch_size = batch_size
        self.tabicl_params = tabicl_params

    def fit(self, X, y):
        from tabicl import TabICLClassifier
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._model = TabICLClassifier(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            device=self.device,
            batch_size=self.batch_size,
            **{k: v for k, v in self.tabicl_params.items()
               if k not in ("n_estimators", "softmax_temperature", "device", "batch_size")},
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._model.predict(X)

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {
            "cat_cols": self.cat_cols,
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "device": self.device,
            "batch_size": self.batch_size,
        }
        params.update(self.tabicl_params)
        return params

    def set_params(self, **params) -> "TabICLNativeWrapper":
        self.cat_cols = params.pop("cat_cols", self.cat_cols)
        self.n_estimators = params.pop("n_estimators", self.n_estimators)
        self.softmax_temperature = params.pop("softmax_temperature",
                                              self.softmax_temperature)
        self.device = params.pop("device", self.device)
        self.batch_size = params.pop("batch_size", self.batch_size)
        self.tabicl_params.update(params)
        return self


class TabFMNativeWrapper(ClassifierMixin, BaseEstimator):
    """Wrap TabFMClassifier (Google's 1.6B-parameter tabular foundation model).

    Like TabICL, TabFM auto-detects categorical columns from pandas dtype, so
    the DataFrame is forwarded as-is.

    Default device is MPS. TabFM is ~30x larger than TabPFN / TabICL and CPU
    inference was measured at 17-36x slower than MPS on this machine, which
    puts a full benchmark run out of reach. Weights are cast to bfloat16 at
    load time and cached process-wide by the library, so repeated fits pay the
    ~20 s load cost only once.

    Parameters
    ----------
    cat_cols : list[str]
        Unused — kept for signature parity with the other native wrappers.
    n_estimators : int, default 4
        Ensemble size, well below the library default of 32. Measured PR AUC
        spread across 1 / 4 / 8 / 32 members is under 0.005 on every dataset
        tried, while wall-clock scales linearly — 4 halves the full run against
        8. Peak memory is unaffected: with ``batch_size=1`` the members run
        sequentially.
    max_num_rows : int, default 5000
        Cap on the in-context training rows. Memory is *superlinear* in context
        length — on a 10 000-row dataset the uncapped fit allocated 13.55 GB and
        drove this 24 GB machine deep into swap, which both slowed it ~5x and
        made the timings meaningless. 5000 brings that to 9.30 GB. Measured PR
        AUC across caps 2500 / 5000 / 7500 was 0.476 / 0.501 / 0.485 — flat, so
        the truncation costs nothing detectable. Affects 22 of the 126 eligible
        datasets; must be disclosed when comparing against models that see the
        full training fold.
    device : str, default ``"mps"``
        Forwarded to the checkpoint loader — ``TabFMClassifier`` itself takes no
        device argument and runs wherever the weights were placed.
    **tabfm_params
        Forwarded verbatim to ``TabFMClassifier``.
    """

    def __init__(
        self,
        cat_cols: list[str],
        n_estimators: int = 4,
        max_num_rows: int = 5000,
        device: str = "mps",
        random_state: int | None = None,
        **tabfm_params,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.device = device
        self.random_state = random_state
        self.tabfm_params = tabfm_params

    def fit(self, X, y):
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._model = TabFMClassifier(
            model=tabfm_v1_0_0.load(model_type="classification", device=self.device),
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            random_state=self.random_state,
            **{k: v for k, v in self.tabfm_params.items()
               if k not in ("model", "n_estimators", "max_num_rows",
                            "random_state")},
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._model.predict(X)

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {
            "cat_cols": self.cat_cols,
            "n_estimators": self.n_estimators,
            "max_num_rows": self.max_num_rows,
            "device": self.device,
            "random_state": self.random_state,
        }
        params.update(self.tabfm_params)
        return params

    def set_params(self, **params) -> "TabFMNativeWrapper":
        self.cat_cols = params.pop("cat_cols", self.cat_cols)
        self.n_estimators = params.pop("n_estimators", self.n_estimators)
        self.max_num_rows = params.pop("max_num_rows", self.max_num_rows)
        self.device = params.pop("device", self.device)
        self.random_state = params.pop("random_state", self.random_state)
        self.tabfm_params.update(params)
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
