"""Runs in its own venv: LightAutoML pins xgboost<3 and statsmodels<=0.14.0 (see README)."""
import joblib
import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import StratifiedKFold

from benchmark.automl_runner import run_automl_benchmark
from benchmark.metrics import pr_auc_score
from config import N_JOBS, RANDOM_STATE, N_OUTER_FOLDS, AUTOML_SEC


SEC = AUTOML_SEC


def evaluate_lightautoml(X, y):
    data_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    data_df["y"] = y
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(np.unique(y))
    task = Task("binary" if n_classes == 2 else "multiclass")
    nested_scores, nested_preds, nested_labels = [], [], []
    for train_inds, test_inds in outer_cv.split(X, y):
        train_df = data_df.iloc[train_inds]
        test_df  = data_df.iloc[test_inds]
        automl = TabularAutoML(task=task, timeout=SEC, cpu_limit=N_JOBS,
                               reader_params={"n_jobs": N_JOBS, "random_state": RANDOM_STATE})
        automl.fit_predict(train_df, roles={"target": "y"}, verbose=0)
        # Labels already 0..K-1 are kept as they are; anything else is remapped in
        # frequency order, which would silently reorder the probability columns.
        assert automl.reader.class_mapping is None
        y_pred = automl.predict(test_df.drop(columns=["y"])).data
        if n_classes == 2:
            y_pred = np.column_stack([1 - y_pred[:, 0], y_pred[:, 0]])
        y_test = test_df["y"].values
        nested_scores.append(pr_auc_score(y_test, y_pred))
        nested_preds.append(y_pred)
        nested_labels.append(y_test)
    return nested_scores, nested_preds, nested_labels


CHECKPOINT   = f"results/lightautoml_sec_{SEC}_ckpt.joblib"
FINAL_OUTPUT = f"results/lightautoml_sec_{SEC}.joblib"

if __name__ == "__main__":
    _, _, random_forest_results, evaluated_datasets, _ = joblib.load(
        "results/compare_baseline_models.joblib"
    )
    run_automl_benchmark(
        evaluate_fn=evaluate_lightautoml,
        evaluated_datasets=evaluated_datasets,
        rf_results=random_forest_results,
        checkpoint_path=CHECKPOINT,
        final_output_path=FINAL_OUTPUT,
    )
