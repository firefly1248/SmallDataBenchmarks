"""Sklearn-compatible wrappers for tabular neural networks.

TabNet         : uses pytorch-tabnet (its own training loop).
FT-Transformer : uses rtdl, shared _train_rtdl training loop.
ResNet         : uses rtdl, treats all features as numeric (ordinal-encodes cats).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import RANDOM_STATE


# ── preprocessing ────────────────────────────────────────────────────────────

class _Preprocessor:
    """Fit/transform: impute NaN → ordinal-encode cats → StandardScale nums."""

    def fit(self, X: pd.DataFrame, cat_cols: list[str]) -> "_Preprocessor":
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.cat_cols_ = [c for c in cat_cols if c in X.columns]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        if self.cat_cols_:
            Xc = X[self.cat_cols_].astype(str).fillna("__nan__")
            self._ord = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            ).fit(Xc)
            # cardinality = len(known cats) + 1 slot for unknown/missing (index 0)
            self.cat_cardinalities_: list[int] = [len(c) + 1 for c in self._ord.categories_]
        else:
            self.cat_cardinalities_ = []

        if self.num_cols_:
            Xn = X[self.num_cols_].values.astype(np.float32)
            self._imp = SimpleImputer(strategy="mean").fit(Xn)
            self._scaler = StandardScaler().fit(self._imp.transform(Xn))

        return self

    def transform(self, X: pd.DataFrame):
        """Return (X_num float32 ndarray, X_cat int64 ndarray)."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.num_cols_:
            Xn = X[self.num_cols_].values.astype(np.float32)
            X_num = self._scaler.transform(self._imp.transform(Xn)).astype(np.float32)
        else:
            X_num = np.empty((len(X), 0), dtype=np.float32)

        if self.cat_cols_:
            Xc = X[self.cat_cols_].astype(str).fillna("__nan__")
            enc = self._ord.transform(Xc).astype(np.int64)
            X_cat = np.clip(enc + 1, 0, None)  # shift: unknown/-1 → 0
        else:
            X_cat = np.empty((len(X), 0), dtype=np.int64)

        return X_num, X_cat

    @property
    def n_num(self) -> int:
        return len(self.num_cols_)

    @property
    def n_cat(self) -> int:
        return len(self.cat_cols_)


# ── shared rtdl training loop ─────────────────────────────────────────────────

def _train_rtdl(
    model: nn.Module,
    forward_fn,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    patience: int,
) -> nn.Module:
    """Train with AdamW + early stopping (15 % val split). Returns best model."""
    y = np.asarray(y)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    val_n = max(int(0.15 * n), min(20, n // 5))
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    def _tensors(i):
        return (
            torch.from_numpy(X_num[i]),
            torch.from_numpy(X_cat[i]),
            torch.from_numpy(y[i].astype(np.int64)),
        )

    Xn_tr, Xc_tr, y_tr = _tensors(tr_idx)
    Xn_val, Xc_val, y_val = _tensors(val_idx)
    n_tr = len(tr_idx)

    if n_classes == 2:
        loss_fn = lambda logits, tgt: nn.BCEWithLogitsLoss()(
            logits.squeeze(1), tgt.float()
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, no_improve = float("inf"), copy.deepcopy(model.state_dict()), 0

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, batch_size):
            b = perm[start : start + batch_size]
            logits = forward_fn(model, Xn_tr[b], Xc_tr[b])
            opt.zero_grad()
            loss_fn(logits, y_tr[b]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(forward_fn(model, Xn_val, Xc_val), y_val).item()

        if val_loss < best_val - 1e-6:
            best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def _predict_proba_rtdl(
    model: nn.Module,
    forward_fn,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    n_classes: int,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(X_num), batch_size):
            logits = forward_fn(
                model,
                torch.from_numpy(X_num[s : s + batch_size]),
                torch.from_numpy(X_cat[s : s + batch_size]),
            ).numpy()
            chunks.append(logits)
    logits = np.concatenate(chunks)
    if n_classes == 2:
        p = 1.0 / (1.0 + np.exp(-logits.ravel()))
        return np.column_stack([1 - p, p])
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


# ── TabNet ────────────────────────────────────────────────────────────────────

class TabNetNativeWrapper(ClassifierMixin, BaseEstimator):
    """Wraps pytorch_tabnet.TabNetClassifier with DataFrame + cat_cols support."""

    def __init__(
        self,
        cat_cols: list[str],
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        mask_type: str = "sparsemax",
        lr: float = 2e-2,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 1024,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.mask_type = mask_type
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

    def fit(self, X, y) -> "TabNetNativeWrapper":
        from pytorch_tabnet.tab_model import TabNetClassifier

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cols = list(X.columns)
        cat_present = [c for c in self.cat_cols if c in cols]
        num_cols = [c for c in cols if c not in cat_present]

        # Ordinal-encode cats and record per-col maps
        self._cat_maps: dict = {}
        cat_idxs, cat_dims = [], []
        for col in cat_present:
            X[col] = X[col].astype(str).fillna("__nan__")
            cats = sorted(X[col].unique())
            self._cat_maps[col] = {c: i for i, c in enumerate(cats)}
            X[col] = X[col].map(self._cat_maps[col]).fillna(0).astype(int)
            cat_idxs.append(cols.index(col))
            cat_dims.append(len(cats))

        # Impute numerics
        self._num_medians = {c: float(X[c].median()) for c in num_cols}
        for c in num_cols:
            X[c] = X[c].fillna(self._num_medians[c])

        self._cols = cols
        self._cat_present = cat_present
        self._num_cols = num_cols
        X_np = X[cols].values.astype(np.float32)

        # 15 % val split for early stopping
        rng = np.random.default_rng(RANDOM_STATE)
        val_n = max(int(0.15 * len(y)), min(20, len(y) // 5))
        idx = rng.permutation(len(y))
        vi, ti = idx[:val_n], idx[val_n:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf = TabNetClassifier(
                n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                gamma=self.gamma, n_independent=self.n_independent,
                n_shared=self.n_shared, lambda_sparse=self.lambda_sparse,
                mask_type=self.mask_type,
                optimizer_params={"lr": self.lr},
                cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
                seed=RANDOM_STATE, verbose=0,
            )
            self._clf.fit(
                X_np[ti], y[ti],
                eval_set=[(X_np[vi], y[vi])],
                eval_metric=["logloss"],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
            )

        self.classes_ = self._clf.classes_
        return self

    def _transform(self, X) -> np.ndarray:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self._cat_present:
            X[col] = X[col].astype(str).fillna("__nan__").map(
                self._cat_maps[col]
            ).fillna(0).astype(int)
        for col in self._num_cols:
            X[col] = X[col].fillna(self._num_medians[col])
        return X[self._cols].values.astype(np.float32)

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(self._transform(X))

    def predict(self, X) -> np.ndarray:
        return self._clf.predict(self._transform(X))

    def get_params(self, deep: bool = True) -> dict:
        return {
            "cat_cols": self.cat_cols, "n_d": self.n_d, "n_a": self.n_a,
            "n_steps": self.n_steps, "gamma": self.gamma,
            "n_independent": self.n_independent, "n_shared": self.n_shared,
            "lambda_sparse": self.lambda_sparse, "mask_type": self.mask_type,
            "lr": self.lr, "max_epochs": self.max_epochs,
            "patience": self.patience, "batch_size": self.batch_size,
        }

    def set_params(self, **params) -> "TabNetNativeWrapper":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── FT-Transformer ────────────────────────────────────────────────────────────

class FTTransformerWrapper(ClassifierMixin, BaseEstimator):
    """Wraps rtdl.FTTransformer with DataFrame + cat_cols support."""

    def __init__(
        self,
        cat_cols: list[str],
        d_token: int = 64,
        n_blocks: int = 3,
        attention_dropout: float = 0.2,
        ffn_d_hidden: int = 128,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 16,
    ) -> None:
        self.cat_cols = cat_cols
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y) -> "FTTransformerWrapper":
        import rtdl

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        self._prep = _Preprocessor().fit(X, self.cat_cols)
        X_num, X_cat = self._prep.transform(X)

        # rtdl requires n_num_features >= 1; pad with a bias column when needed
        self._num_padded = self._prep.n_num == 0
        if self._num_padded:
            X_num = np.ones((len(X), 1), dtype=np.float32)

        d_out = 1 if self._n_classes == 2 else self._n_classes
        n_num = X_num.shape[1]
        cat_cards = self._prep.cat_cardinalities_ if self._prep.n_cat > 0 else None

        self._model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_num,
            cat_cardinalities=cat_cards,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            d_out=d_out,
        )

        n_cat = self._prep.n_cat

        def fwd(model, xn, xc):
            return model(xn, xc if n_cat > 0 else None)

        self._model = _train_rtdl(
            self._model, fwd, X_num, X_cat, y,
            n_classes=self._n_classes,
            lr=self.lr, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_num, X_cat = self._prep.transform(X)
        if self._num_padded:
            X_num = np.ones((len(X_num), 1), dtype=np.float32)
        n_cat = self._prep.n_cat

        def fwd(model, xn, xc):
            return model(xn, xc if n_cat > 0 else None)

        return _predict_proba_rtdl(self._model, fwd, X_num, X_cat,
                                   self._n_classes, self.batch_size)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in (
            "cat_cols", "d_token", "n_blocks", "attention_dropout",
            "ffn_d_hidden", "ffn_dropout", "residual_dropout",
            "lr", "weight_decay", "max_epochs", "batch_size", "patience",
        )}

    def set_params(self, **params) -> "FTTransformerWrapper":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── ResNet ────────────────────────────────────────────────────────────────────

class ResNetWrapper(ClassifierMixin, BaseEstimator):
    """Wraps rtdl.ResNet. Ordinal-encodes cats and stacks with scaled numerics."""

    def __init__(
        self,
        cat_cols: list[str],
        n_blocks: int = 4,
        d_main: int = 128,
        d_hidden: int = 256,
        dropout_first: float = 0.2,
        dropout_second: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 16,
    ) -> None:
        self.cat_cols = cat_cols
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.d_hidden = d_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y) -> "ResNetWrapper":
        import rtdl

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        self._prep = _Preprocessor().fit(X, self.cat_cols)
        X_num, X_cat = self._prep.transform(X)
        # stack cats as float alongside numerics
        X_all = np.hstack([X_num, X_cat.astype(np.float32)])
        X_cat_empty = np.empty((len(X_all), 0), dtype=np.int64)

        d_out = 1 if self._n_classes == 2 else self._n_classes
        self._model = rtdl.ResNet.make_baseline(
            d_in=X_all.shape[1],
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout_first,
            dropout_second=self.dropout_second,
            d_out=d_out,
        )

        def fwd(model, xn, xc):
            return model(xn)

        self._model = _train_rtdl(
            self._model, fwd, X_all, X_cat_empty, y,
            n_classes=self._n_classes,
            lr=self.lr, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_num, X_cat = self._prep.transform(X)
        X_all = np.hstack([X_num, X_cat.astype(np.float32)])
        X_cat_empty = np.empty((len(X_all), 0), dtype=np.int64)

        def fwd(model, xn, xc):
            return model(xn)

        return _predict_proba_rtdl(self._model, fwd, X_all, X_cat_empty,
                                   self._n_classes, self.batch_size)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in (
            "cat_cols", "n_blocks", "d_main", "d_hidden",
            "dropout_first", "dropout_second",
            "lr", "weight_decay", "max_epochs", "batch_size", "patience",
        )}

    def set_params(self, **params) -> "ResNetWrapper":
        for k, v in params.items():
            setattr(self, k, v)
        return self
