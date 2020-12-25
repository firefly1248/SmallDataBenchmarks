"""
How does LightGBM compare?
"""

import time
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils import load_data

N_JOBS = 4 * 4 * 9
N_ITER = 25  # budget for hyperparam search


def evaluate_pipeline_helper(X, y, pipeline, param_grid, random_state=0):
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    clf = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER,
        cv=inner_cv,
        scoring="roc_auc_ovr_weighted",
        n_jobs=N_JOBS,
        random_state=random_state,
        verbose=-1,
    )
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS)
    return nested_score


def define_and_evaluate_lightgbm_pipeline(X, y, random_state=0):
    if len(set(y)) == 2:
        pipeline = lgb.LGBMClassifier(
            objective="binary",
            num_iterations=500,
            metric="auc",
            verbose=-1,
            tree_learner="feature",
            random_state=random_state,
            silent=True,
        )
    else:
        pipeline = lgb.LGBMClassifier(
            objective="multiclass",
            num_iterations=500,
            metric="auc_mu",
            verbose=-1,
            tree_learner="feature",
            random_state=random_state,
            silent=True,
        )
    param_grid = {
        "learning_rate": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "num_leaves": [2, 4, 8, 16, 32, 64],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [2, 4, 8, 16, 32, 64, 128, 256],
        "min_child_weight": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "reg_alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "reg_lambda": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "max_depth": [1, 2, 4, 8, 16, 32, -1],
    }
    nested_scores = evaluate_pipeline_helper(X, y, pipeline, param_grid, random_state=random_state)
    return nested_scores


# run model on all datasets
with open("results/01_compare_baseline_models.pickle", "rb") as f:
    _, _, _, evaluated_datasets, _ = pickle.load(f)


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
    nested_scores = define_and_evaluate_lightgbm_pipeline(X, y)
    results.append(nested_scores)
    elapsed = time.time() - start
    times.append(elapsed)
    print("done. elapsed:", elapsed)

#
results = np.array(results)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open(f"results/02_lightgbm_n_iter_{N_ITER}.pickle", "wb") as f:
    pickle.dump((results, times), f)
