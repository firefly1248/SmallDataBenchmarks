"""
Which baseline models are best?
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from benchmark.data import load_data  # intentionally not load_data_df: this meta-dataset has swallowed
                                      # information about what's categorical and what isn't, so we treat
                                      # every feature as continuous via one-hot encoding
from benchmark.metrics import PR_AUC_SCORER
from config import N_JOBS, RANDOM_STATE, N_OUTER_FOLDS, N_INNER_FOLDS, MAX_DATASET_ROWS


def evaluate_pipeline_helper(X, y, pipeline, param_grid, scoring=PR_AUC_SCORER, random_state=RANDOM_STATE):
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=random_state)
    # Note: both GridSearchCV and cross_val_score use n_jobs=N_JOBS (legacy script).
    # This causes N_JOBS² thread contention; acceptable since results are already saved.
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring=scoring, n_jobs=N_JOBS)
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=scoring, n_jobs=N_JOBS)
    return nested_score


def define_and_evaluate_pipelines(X, y, random_state=RANDOM_STATE):
    # LinearSVC
    pipeline1 = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "svc",
                SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=random_state),
            ),
        ]
    )
    param_grid1 = {
        "svc__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    }

    # logistic regression
    pipeline2 = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="saga", max_iter=10000, penalty="elasticnet", l1_ratio=0.1, random_state=random_state
                ),  # without elasticnet penalty, LR can get pretty bad sometimes
            ),
        ]
    )
    param_grid2 = {
        "logistic__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    }

    # random forest
    pipeline3 = RandomForestClassifier(random_state=random_state)
    param_grid3 = {
        "max_depth": [1, 2, 4, 8, 16, 32, None],
    }

    nested_scores1 = evaluate_pipeline_helper(X, y, pipeline1, param_grid1, random_state=random_state)
    nested_scores2 = evaluate_pipeline_helper(X, y, pipeline2, param_grid2, random_state=random_state)
    nested_scores3 = evaluate_pipeline_helper(X, y, pipeline3, param_grid3, random_state=random_state)

    return nested_scores1, nested_scores2, nested_scores3


CHECKPOINT = "results/compare_baseline_models_ckpt.joblib"

if __name__ == "__main__":
    database = pd.read_json("database.json").T
    database = database[database.nrow >= 50]

    # Resume from checkpoint if available
    results1, results2, results3, evaluated_datasets, times = [], [], [], [], []
    try:
        results1, results2, results3, evaluated_datasets, times = joblib.load(CHECKPOINT)
        results1 = list(results1)
        results2 = list(results2)
        results3 = list(results3)
        evaluated_datasets = list(evaluated_datasets)
        times = list(times)
        print(f"Resuming from checkpoint: {len(evaluated_datasets)} datasets already done.")
    except FileNotFoundError:
        pass

    done_set = set(evaluated_datasets)

    for i, dataset_name in enumerate(database.index.values):
        if dataset_name not in done_set:
            X, y = load_data(dataset_name)
            # datasets might have too few samples per class
            if len(y) > 0 and np.sum(pd.Series(y).value_counts() <= 15) == 0:
                rng = np.random.default_rng(RANDOM_STATE)
                if len(y) > MAX_DATASET_ROWS:
                    random_idx = rng.choice(len(y), MAX_DATASET_ROWS, replace=False)
                    X = X[random_idx, :]
                    y = y[random_idx]
                print("starting:", dataset_name, X.shape)
                start = time.time()
                nested_scores1, nested_scores2, nested_scores3 = define_and_evaluate_pipelines(X, y)
                results1.append(nested_scores1)
                results2.append(nested_scores2)
                results3.append(nested_scores3)
                elapsed = time.time() - start
                evaluated_datasets.append(dataset_name)
                done_set.add(dataset_name)
                times.append(elapsed)
                print("done. elapsed:", elapsed)
                print("scores:", np.mean(nested_scores1), np.mean(nested_scores2), np.mean(nested_scores3))
                # checkpoint after each dataset
                joblib.dump(
                    (np.array(results1), np.array(results2), np.array(results3),
                     np.array(evaluated_datasets), np.array(times)),
                    CHECKPOINT,
                )

    results1 = np.array(results1)
    results2 = np.array(results2)
    results3 = np.array(results3)
    evaluated_datasets = np.array(evaluated_datasets)
    times = np.array(times)

    # save everything to disk so we can make plots elsewhere
    joblib.dump((results1, results2, results3, evaluated_datasets, times), "results/compare_baseline_models.joblib")
