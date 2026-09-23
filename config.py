N_JOBS = 16   # -1 = all cores; lower values cut CPU and thermal load

# cross_val_score parallelises over folds only, so each estimator can take a few
# threads. CatBoost and RandomForest gain 1.04x and 1.18x end to end; XGBoost and
# LightGBM are slower with threads and stay at 1. Scores unaffected.
INNER_FIT_THREADS = 3

RANDOM_STATE = 0

N_OUTER_FOLDS = 4   # evaluation
N_INNER_FOLDS = 4   # hyperparameter search

N_TRIALS    = 50    # Optuna trials per outer fold per model
N_TRIALS_NN = 50

# Every published AutoML number was measured at 300; a fresh run at another
# budget writes different files than the ones the figures read.
AUTOML_SEC = 300    # per outer fold, AutoGluon / MLJAR

MAX_DATASET_ROWS = 10_000  # subsample datasets larger than this

MODELS_TO_RUN = "all"  # CLI --models / --skip overrides; e.g. ["svc", "logreg"]

# UCI++ ships the same underlying data in several configurations. Every figure
# drops these, so running them spends about 10 % of the benchmark on results
# nothing reads. The scores already on disk are kept; new runs skip them.
DUPLICATE_DATASETS = frozenset({
    "statlog-german-credit-numeric",
    "thyroid-allhyper", "thyroid-allhypo", "thyroid-allrep",
    "thyroid-dis", "thyroid-hypothyroid",
    "volcanoes-a3", "volcanoes-b3", "volcanoes-b4",
    "volcanoes-b5", "volcanoes-b6", "volcanoes-d3", "volcanoes-d4",
    "waveform-v2", "wine-quality-white-5class",
})

# Datasets where even the weakest model clears this are dropped from the
# figures: they rank the whole field on noise in the fourth decimal.
TRIVIAL_PR_AUC = 0.99
