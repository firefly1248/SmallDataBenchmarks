"""Nested cross-validation runner with Optuna / GridSearchCV tuning."""
from __future__ import annotations

import base64
import os
import pickle
import threading

import numpy as np
import optuna
from joblib import Parallel, delayed
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


def _finished(study) -> int:
    return sum(t.state.is_finished() for t in study.trials)


def _save_sampler(study, trial) -> None:
    """Keep the sampler state with the study, so a resumed search draws what an
    uninterrupted one would; a fresh seeded sampler would repeat its first trials."""
    state = base64.b64encode(pickle.dumps(study.sampler)).decode()
    study.set_user_attr("sampler", {"finished": _finished(study), "state": state})


def _study(spec, storage, dataset_name, fold):
    study = optuna.create_study(
        study_name=None if storage is None else f"{dataset_name}/fold{fold}",
        storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=spec.sampler_seed))
    saved = study.user_attrs.get("sampler")
    if saved is not None:
        if saved["finished"] == _finished(study):
            study.sampler = pickle.loads(base64.b64decode(saved["state"]))
        else:
            # Cut between a trial's commit and its sampler save.
            print(f"  {dataset_name}/fold{fold}: sampler state is a trial behind, resuming with a fresh sampler")
    return study


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
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    splits = list(outer_cv.split(X, y))

    def train(fold):
        return X.iloc[splits[fold][0]].reset_index(drop=True), y[splits[fold][0]]

    best = {}
    if spec.objective is not None:
        storage = None
        if dataset_name is not None:
            os.makedirs(os.path.join(STUDY_DIR, run), exist_ok=True)
            # One storage for the fold threads, with room to wait for SQLite's write lock.
            storage = optuna.storages.RDBStorage(_storage(model_name, run),
                                                 engine_kwargs={"connect_args": {"timeout": 60}})
        studies = [_study(spec, storage, dataset_name, fold) for fold in range(N_OUTER_FOLDS)]
        stop = threading.Event()

        def stop_if_asked(study, trial):
            if stop.is_set():
                study.stop()

        def search(fold):
            X_train, y_train = train(fold)
            study = studies[fold]
            # Failed trials count, as they do in an uninterrupted run; a trial cut
            # off mid-fit stays RUNNING and is redone. One raising trial must not
            # abort the dataset.
            study.optimize(lambda t: spec.objective(t, X_train, y_train, inner_cv, n_classes, cat_cols),
                           n_trials=max((spec.n_trials or N_TRIALS) - _finished(study), 0),
                           catch=(Exception,), callbacks=[_save_sampler, stop_if_asked])
            return dict(study.best_params)

        # Each fold has its own seeded study, so concurrent folds search exactly as
        # sequential ones do. Fold threads: the inner cross_val_score then runs its
        # fits on threads too (joblib nests that way), which only one-thread fits
        # tolerate; see ModelSpec.parallel_folds. On a timeout or error in this
        # thread, the fold threads stop after their current trial.
        try:
            found = Parallel(n_jobs=N_OUTER_FOLDS if spec.parallel_folds else 1, prefer="threads")(
                delayed(search)(fold) for fold in range(N_OUTER_FOLDS))
        except BaseException:
            stop.set()
            raise
        best = dict(enumerate(found))

    scores, preds, labels, params = [], [], [], []
    for fold, (_, test_idx) in enumerate(splits):
        X_train, y_train = train(fold)
        if spec.grid is not None:
            gs = spec.grid(inner_cv, cat_cols)
            gs.fit(X_train, y_train)
            model, best_params = gs.best_estimator_, dict(gs.best_params_)
        else:
            best_params = best.get(fold, {})
            model = spec.build(best_params, n_classes, cat_cols)
            model.fit(X_train, y_train)
        y_test = y[test_idx]
        y_pred = model.predict_proba(X.iloc[test_idx].reset_index(drop=True))
        scores.append(pr_auc_score(y_test, y_pred))
        preds.append(y_pred)
        labels.append(y_test)
        params.append(best_params)
    return scores, preds, labels, params
