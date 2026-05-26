import shutil
import joblib
import numpy as np
from supervised import AutoML
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from benchmark.automl_runner import run_automl_benchmark
from config import N_JOBS, RANDOM_STATE, N_OUTER_FOLDS, AUTOML_SEC


SEC = AUTOML_SEC
MLJAR_PATH = "AutoML_temp"


def evaluate_mljar(X, y):
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(np.unique(y))
    ml_task = "binary_classification" if n_classes == 2 else "multiclass_classification"
    nested_scores = []
    for train_inds, test_inds in outer_cv.split(X, y):
        X_train, y_train = X[train_inds], y[train_inds]
        X_test,  y_test  = X[test_inds],  y[test_inds]
        shutil.rmtree(MLJAR_PATH, ignore_errors=True)  # clean before fit
        automl = AutoML(
            results_path=MLJAR_PATH,
            mode="Compete",
            ml_task=ml_task,
            eval_metric="logloss",
            total_time_limit=SEC,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        automl.fit(X_train, y_train)
        y_pred = automl.predict_proba(X_test)
        y_score = y_pred[:, 1] if n_classes == 2 else y_pred
        score = average_precision_score(y_test, y_score, average="weighted")
        nested_scores.append(score)
        shutil.rmtree(MLJAR_PATH, ignore_errors=True)
    return nested_scores


CHECKPOINT   = f"results/mljar_sec_{SEC}_ckpt.joblib"
FINAL_OUTPUT = f"results/mljar_sec_{SEC}.joblib"

if __name__ == "__main__":
    _, _, random_forest_results, evaluated_datasets, _ = joblib.load(
        "results/compare_baseline_models.joblib"
    )
    run_automl_benchmark(
        evaluate_fn=evaluate_mljar,
        evaluated_datasets=evaluated_datasets,
        rf_results=random_forest_results,
        checkpoint_path=CHECKPOINT,
        final_output_path=FINAL_OUTPUT,
    )
