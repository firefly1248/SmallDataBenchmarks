"""GridSearchCV-based hyperparameter tuning for simple linear models."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from benchmark.encoding import CatFeaturesEncoder
from benchmark.metrics import PR_AUC_SCORER
from config import N_JOBS, RANDOM_STATE


# Encoding strategies to search when categorical columns are present
CAT_STRATEGIES_GRID: list[str] = ["target", "james_stein", "m_estimate", "catboost_enc"]

# Models handled by GridSearchCV (well-understood, small HP spaces)
GRID_SEARCH_MODELS: frozenset[str] = frozenset({"svc", "logreg"})


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
    model_name : ``"svc"`` or ``"logreg"``
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
            ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
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

    else:
        raise ValueError(
            f"model_name={model_name!r} is not a grid-search model. "
            f"Expected one of {sorted(GRID_SEARCH_MODELS)}."
        )

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring=PR_AUC_SCORER,
        n_jobs=N_JOBS,
        refit=True,
    )
