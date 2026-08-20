# ── Parallelism ───────────────────────────────────────────────────────────────
# Number of parallel jobs for cross-validation.
# -1 = all cores; lower values reduce CPU/thermal load.
N_JOBS = 16

# Threads per estimator inside the Optuna inner CV, where cross_val_score
# parallelises over folds only and so occupies at most N_INNER_FOLDS cores.
# Applied to CatBoost and RandomForest (measured 1.04x and 1.18x end to end);
# XGBoost and LightGBM are slower with threads and stay at 1. Scores unaffected.
INNER_FIT_THREADS = 3

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 0

# ── Cross-validation scheme ───────────────────────────────────────────────────
N_OUTER_FOLDS = 4   # outer CV folds (evaluation)
N_INNER_FOLDS = 4   # inner CV folds (hyperparameter search)

# ── Optuna budget ─────────────────────────────────────────────────────────────
N_TRIALS    = 50    # Optuna trials per outer fold per model
N_TRIALS_NN = 50    # Optuna trials for neural networks

# ── AutoML budget ─────────────────────────────────────────────────────────────
# 300 is the budget every published AutoML number was measured at. At 1000 a
# fresh run writes different files than the ones the figures read.
AUTOML_SEC = 300    # time limit (seconds) per outer fold for AutoGluon / MLJAR

# ── Dataset size cap ─────────────────────────────────────────────────────────
MAX_DATASET_ROWS = 10_000  # subsample datasets larger than this

# ── Models to run ─────────────────────────────────────────────────────────────
# 'all' = run all models. CLI --models / --skip takes priority over this setting.
MODELS_TO_RUN = "all"  # e.g. ["svc", "logreg", "random_forest"]
