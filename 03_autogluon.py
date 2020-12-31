import time
import pickle
import numpy as np
import pandas as pd
import autogluon as ag
from autogluon import TabularPrediction as task
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import load_data


SEC = 60 * 5


def define_and_evaluate_autogluon_pipeline(X, y, random_state=0):
    # autogluon dataframes
    data_df = pd.DataFrame(X)
    data_df["y"] = y
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    nested_scores = []
    for train_inds, test_inds in outer_cv.split(X, y):
        data_df_train = data_df.iloc[train_inds, :]
        data_df_test = data_df.iloc[test_inds, :]
        if len((set(y))) == 2:
            eval_metric = "roc_auc"
            problem_type = "binary"
        else:
            eval_metric = "f1_weighted"  # no multiclass auroc in autogluon
            problem_type = "multiclass"
        predictor = task.fit(
            data_df_train,
            "y",
            time_limits=SEC,
            presets="best_quality",
            output_directory=".autogluon_temp",
            eval_metric=eval_metric,
            problem_type=problem_type,
            verbosity=0,
        )
        y_pred = predictor.predict_proba(data_df.iloc[test_inds, :])
        # same as roc_auc_ovr_weighted
        score = roc_auc_score(data_df_test["y"], y_pred, average="weighted", multi_class="ovr")
        nested_scores.append(score)
    return nested_scores


# run model on all datasets
with open("results/01_compare_baseline_models.pickle", "rb") as f:
    _, _, random_forest_results, evaluated_datasets, _ = pickle.load(f)

results = []
times = []
for i, dataset_name in enumerate(evaluated_datasets):
    X, y = load_data(dataset_name)
    np.random.seed(0)
    if len(y) > 10000:
        # subset to 10000 if too large
        random_idx = np.random.choice(len(y), 10000, replace=False)
        X = X[random_idx, :]
        y = y[random_idx]
    print("starting:", dataset_name, X.shape)
    start = time.time()
    nested_scores = define_and_evaluate_autogluon_pipeline(X, y)
    results.append(nested_scores)
    elapsed = time.time() - start
    times.append(elapsed)
    print("done. elapsed:", elapsed)
    print(f"AutoGluon score: {np.mean(nested_scores)}, Random Forest score: {np.mean(random_forest_results[i])}")

#
results = np.array(results)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open(f"results/03_autogluon_sec_{SEC}.pickle", "wb") as f:
    pickle.dump((results, times), f)
