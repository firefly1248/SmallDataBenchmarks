"""Nested cross-validation runner with Optuna / GridSearchCV tuning."""
from __future__ import annotations

import os

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from benchmark.metrics import pr_auc_score
from benchmark.models.registry import MODELS
from config import N_INNER_FOLDS, N_OUTER_FOLDS, N_TRIALS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One SQLite file per model, one study per (dataset, outer fold). A study lives only
# while its dataset is unfinished: optuna_models.py drops it once the dataset is
# checkpointed, so what is left is the progress of a run that was cut off.
STUDY_DIR = "results/optuna"


def _storage(model_name: str, run: str = "") -> str:
    return "sqlite:///" + os.path.join(STUDY_DIR, run, f"{model_name}.db")


def drop_studies(model_name: str, dataset_name: str, run: str = "") -> None:
    if not os.path.exists(os.path.join(STUDY_DIR, run, f"{model_name}.db")):
        return
    storage = _storage(model_name, run)
    for name in optuna.study.get_all_study_names(storage):
        if name.startswith(f"{dataset_name}/"):
            optuna.delete_study(study_name=name, storage=storage)


def run_nested_cv(
    X,
    y: np.ndarray,
    model_name: str,
    cat_cols: list[str],
    dataset_name: str | None = None,
    split_seed: int = RANDOM_STATE,
    run: str = "",
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[dict]]:
    """Run nested cross-validation for *model_name*.

    Tuning as ``MODELS[model_name]`` says: GridSearchCV, Optuna TPE, or none.

    With ``dataset_name``, Optuna studies are stored in ``STUDY_DIR`` and a rerun
    resumes each outer fold from its finished trials. After a code change, delete
    the model's database first, or the old trials are reused.

    ``split_seed`` seeds the outer split only; the inner split, the sampler and
    the models keep ``RANDOM_STATE``, so another seed measures split noise alone.
    ``run`` keeps its checkpoints and studies apart (see ``benchmark.checkpoints``).

    Returns ``(scores, preds, labels, best_params)``, one entry per outer fold.
    """
    n_classes = int(np.unique(y).size)

    spec = MODELS[model_name]
    if (X.shape[1] >= spec.limits.get("max_features", float("inf"))
            or n_classes > spec.limits.get("max_classes", float("inf"))):
        nan_scores = [float("nan")] * N_OUTER_FOLDS
        return nan_scores, None, None, None

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                               random_state=split_seed)
    nested_scores: list[float] = []
    nested_preds: list[np.ndarray] = []
    nested_labels: list[np.ndarray] = []
    nested_best_params: list[dict] = []
    storage = None
    if dataset_name is not None:
        os.makedirs(os.path.join(STUDY_DIR, run), exist_ok=True)
        storage = _storage(model_name, run)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        y_test  = y[test_idx]

        inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)

        if spec.grid is not None:
            gs = spec.grid(inner_cv, cat_cols)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = dict(gs.best_params_)
        elif spec.objective is None:
            model = spec.build({}, n_classes, cat_cols)
            model.fit(X_train, y_train)
            best_params = {}
        else:
            study = optuna.create_study(
                study_name=None if storage is None else f"{dataset_name}/fold{fold}",
                storage=storage,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=spec.sampler_seed),
            )
            # Failed trials count, as they do in an uninterrupted run; a trial cut
            # off mid-fit stays RUNNING and is redone.
            finished = sum(t.state.is_finished() for t in study.trials)
            # One raising trial must not abort the dataset.
            study.optimize(lambda t: spec.objective(t, X_train, y_train, inner_cv, n_classes, cat_cols),
                           n_trials=max((spec.n_trials or N_TRIALS) - finished, 0),
                           catch=(Exception,))
            best_params = dict(study.best_params)
            model = spec.build(best_params, n_classes, cat_cols)
            model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)
        nested_scores.append(pr_auc_score(y_test, y_pred))
        nested_preds.append(y_pred)
        nested_labels.append(y_test)
        nested_best_params.append(best_params)

    return nested_scores, nested_preds, nested_labels, nested_best_params
