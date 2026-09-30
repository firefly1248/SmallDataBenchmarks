import joblib
import numpy as np
from flaml import AutoML
from sklearn.model_selection import StratifiedKFold

from benchmark.automl_runner import run_automl_benchmark
from benchmark.metrics import pr_auc_score
from config import N_JOBS, RANDOM_STATE, N_OUTER_FOLDS, AUTOML_SEC


SEC = AUTOML_SEC


def evaluate_flaml(X, y):
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    nested_scores, nested_preds, nested_labels = [], [], []
    for train_inds, test_inds in outer_cv.split(X, y):
        X_train, y_train = X[train_inds], y[train_inds]
        X_test,  y_test  = X[test_inds],  y[test_inds]
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", metric="log_loss",
                   time_budget=SEC, n_jobs=N_JOBS, seed=RANDOM_STATE, verbose=0)
        y_pred = automl.predict_proba(X_test)
        nested_scores.append(pr_auc_score(y_test, y_pred))
        nested_preds.append(y_pred)
        nested_labels.append(y_test)
    return nested_scores, nested_preds, nested_labels


CHECKPOINT   = f"results/flaml_sec_{SEC}_ckpt.joblib"
FINAL_OUTPUT = f"results/flaml_sec_{SEC}.joblib"

if __name__ == "__main__":
    _, _, random_forest_results, evaluated_datasets, _ = joblib.load(
        "results/compare_baseline_models.joblib"
    )
    run_automl_benchmark(
        evaluate_fn=evaluate_flaml,
        evaluated_datasets=evaluated_datasets,
        rf_results=random_forest_results,
        checkpoint_path=CHECKPOINT,
        final_output_path=FINAL_OUTPUT,
    )
