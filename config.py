# ── Parallelism ───────────────────────────────────────────────────────────────
# Number of parallel jobs for cross-validation.
# -1 = all cores; lower values reduce CPU/thermal load.
N_JOBS = 16

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 0

# ── Cross-validation scheme ───────────────────────────────────────────────────
N_OUTER_FOLDS = 4   # outer CV folds (evaluation)
N_INNER_FOLDS = 4   # inner CV folds (hyperparameter search)

# ── Optuna budget ─────────────────────────────────────────────────────────────
N_TRIALS    = 50    # Optuna trials per outer fold per model
N_TRIALS_NN = 20    # fewer trials for neural networks (each trial is much slower)

# ── AutoML budget ─────────────────────────────────────────────────────────────
AUTOML_SEC = 1000   # time limit (seconds) per outer fold for AutoGluon / MLJAR

# ── Dataset size cap ─────────────────────────────────────────────────────────
MAX_DATASET_ROWS = 10_000  # subsample datasets larger than this

# ── Models to run ─────────────────────────────────────────────────────────────
# 'all' = run all models. CLI --models / --skip takes priority over this setting.
MODELS_TO_RUN = "all"  # e.g. ["svc", "logreg", "random_forest"]
