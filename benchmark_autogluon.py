import shutil
import joblib
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from benchmark.automl_runner import run_automl_benchmark
from config import RANDOM_STATE, N_OUTER_FOLDS, AUTOML_SEC


SEC = AUTOML_SEC
AG_PATH = ".autogluon_temp"


def evaluate_autogluon(X, y, random_state=RANDOM_STATE):
    data_df = pd.DataFrame(X)
    data_df["y"] = y
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y))
    problem_type = "binary" if n_classes == 2 else "multiclass"
    nested_scores = []
    for train_inds, test_inds in outer_cv.split(X, y):
        train_df = data_df.iloc[train_inds]
        test_df  = data_df.iloc[test_inds]
        shutil.rmtree(AG_PATH, ignore_errors=True)  # clean before fit
        predictor = TabularPredictor(
            label="y",
            problem_type=problem_type,
            eval_metric="log_loss",
            verbosity=0,
            path=AG_PATH,
        ).fit(train_df, time_limit=SEC, presets="best_quality", num_cpus=8)
        y_pred = predictor.predict_proba(test_df.drop(columns=["y"])).values
        y_score = y_pred[:, 1] if n_classes == 2 else y_pred
        score = average_precision_score(test_df["y"], y_score, average="weighted")
        nested_scores.append(score)
        shutil.rmtree(AG_PATH, ignore_errors=True)
    return nested_scores


CHECKPOINT   = f"results/autogluon_sec_{SEC}_ckpt.joblib"
FINAL_OUTPUT = f"results/autogluon_sec_{SEC}.joblib"

if __name__ == "__main__":
    _, _, random_forest_results, evaluated_datasets, _ = joblib.load(
        "results/compare_baseline_models.joblib"
    )
    run_automl_benchmark(
        evaluate_fn=evaluate_autogluon,
        evaluated_datasets=evaluated_datasets,
        rf_results=random_forest_results,
        checkpoint_path=CHECKPOINT,
        final_output_path=FINAL_OUTPUT,
    )
