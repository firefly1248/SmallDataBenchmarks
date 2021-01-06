import time
import pickle
import numpy as np
import pandas as pd
from supervised import AutoML
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import load_data


SEC = 60 * 5


def define_and_evaluate_mljar_pipeline(X, y, random_state=0):
    
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    nested_scores = []
    for train_inds, test_inds in outer_cv.split(X, y):
        
        X_train, y_train = X[train_inds, :], y[train_inds]
        X_test, y_test = X[test_inds, :], y[test_inds]

        binary = len((set(y))) == 2
        eval_metric = "auc" if binary else "logloss" 
        ml_task = "binary_classification" if binary else "multiclass_classifcation"

        automl = AutoML(mode="Compete", eval_metric=eval_metric, total_time_limit=SEC, ml_task=ml_task)
        automl.fit(X_train, y_train)
        y_pred = automl.predict_proba(X_test)

        # same as roc_auc_ovr_weighted
        if binary:
            score = roc_auc_score(y_test, y_pred[:, 1], average="weighted", multi_class="ovr")
        else:
            score = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovr")
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
    nested_scores = define_and_evaluate_mljar_pipeline(X, y)
    results.append(nested_scores)
    elapsed = time.time() - start
    times.append(elapsed)
    print("done. elapsed:", elapsed)
    print(f"MLJAR score: {np.mean(nested_scores)}, Random Forest score: {np.mean(random_forest_results[i])}")

#
results = np.array(results)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open(f"results/04_mljar_sec_{SEC}.pickle", "wb") as f:
    pickle.dump((results, times), f)
