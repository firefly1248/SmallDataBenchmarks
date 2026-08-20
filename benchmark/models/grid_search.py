"""GridSearchCV-based hyperparameter tuning for simple linear models."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from benchmark.encoding import CatFeaturesEncoder, TabICLNativeWrapper, TabPFNNativeWrapper
from benchmark.metrics import PR_AUC_SCORER
from config import N_JOBS, RANDOM_STATE


CAT_STRATEGIES_GRID: list[str] = ["target", "james_stein", "m_estimate", "catboost_enc"]

GRID_SEARCH_MODELS: frozenset[str] = frozenset({"svc", "logreg", "tabpfn", "tabpfn3", "tabicl"})

# PyTorch-backed models: parallel GridSearchCV workers trigger OMP mutex
# conflicts on macOS, so these search serially.
_SERIAL_GRID_MODELS: frozenset[str] = frozenset({"tabpfn", "tabpfn3", "tabicl"})


def build_grid_search(
    model_name: str,
    inner_cv,
    cat_cols: list[str],
) -> GridSearchCV:
    """Build a GridSearchCV pipeline for *model_name*.

    The ``cat_enc__strategy`` parameter is only added to the search grid when
    *cat_cols* is non-empty — on purely numeric data all strategies are
    equivalent and searching them wastes compute.

    Parameters
    ----------
    model_name : ``"svc"``, ``"logreg"``, ``"tabpfn"``, ``"tabpfn3"``, or ``"tabicl"``
    inner_cv   : cross-validation splitter passed to GridSearchCV
    cat_cols   : list of categorical column names in the training data

    Returns
    -------
    GridSearchCV (not yet fitted)
    """
    base_pre = [
        ("cat_enc", CatFeaturesEncoder(strategy="target")),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
    cat_strategy_grid: dict = (
        {"cat_enc__strategy": CAT_STRATEGIES_GRID} if cat_cols else {}
    )

    if model_name == "svc":
        pipeline = Pipeline(base_pre + [
            # libsvm's SMO solver is unbounded by default and can spin for hours
            # on one pathological (config, fold) pair. Normal fits here need
            # ~1e3 iterations, so this only binds on runaways.
            ("svc", SVC(probability=True, max_iter=2_000_000,
                        random_state=RANDOM_STATE)),
        ])
        param_grid = [
            {
                **cat_strategy_grid,
                "svc__kernel":       ["rbf"],
                "svc__C":            [0.1, 1, 10, 100],
                "svc__gamma":        ["scale", "auto"],
                "svc__class_weight": ["balanced", None],
            },
            {
                **cat_strategy_grid,
                "svc__kernel":       ["linear"],
                "svc__C":            [0.1, 1, 10],
                "svc__class_weight": ["balanced", None],
            },
        ]

    elif model_name == "logreg":
        pipeline = Pipeline(base_pre + [
            ("lr", LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=10000,
                random_state=RANDOM_STATE,
            )),
        ])
        param_grid = {
            **cat_strategy_grid,
            "lr__C":            [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100],
            "lr__l1_ratio":     [0.0, 0.5, 1.0],
            "lr__class_weight": ["balanced", None],
        }

    elif model_name == "tabpfn":
        pipeline = TabPFNNativeWrapper(cat_cols=cat_cols, random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators":        [4, 8, 16, 32],
            "balance_probabilities": [True, False],
        }

    elif model_name == "tabpfn3":
        # auto_scale on: for a fresh measurement, run the model the way the
        # library ships it. On the 5 wide datasets this makes both grid points
        # collapse to the same effective ensemble size — half those inner fits
        # are duplicates — but disabling it would instead leave most features
        # unsampled there.
        pipeline = TabPFNNativeWrapper(cat_cols=cat_cols, model_version="v3",
                                       auto_scale_n_estimators=True,
                                       random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators":          [4, 8],
            "balance_probabilities": [True, False],
        }

    elif model_name == "tabicl":
        pipeline = TabICLNativeWrapper(cat_cols=cat_cols, random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators":        [4, 8, 16],
            "softmax_temperature": [0.7, 0.9, 1.1],
        }

    else:
        raise ValueError(
            f"model_name={model_name!r} is not a grid-search model. "
            f"Expected one of {sorted(GRID_SEARCH_MODELS)}."
        )

    gs_n_jobs = 1 if model_name in _SERIAL_GRID_MODELS else N_JOBS

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring=PR_AUC_SCORER,
        n_jobs=gs_n_jobs,
        refit=True,
    )
