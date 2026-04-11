"""Evaluation metrics and scorers for classification benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, make_scorer


PR_AUC_SCORER = make_scorer(
    average_precision_score,
    average="weighted",
    response_method="predict_proba",
)
"""Weighted PR AUC scorer compatible with sklearn's cross_val_score / GridSearchCV.

For binary classification ``make_scorer`` internally passes ``y_score[:, 1]``
when ``response_method="predict_proba"``, so no manual slicing is needed
inside objectives.  For multiclass the full probability matrix is used.
"""


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute weighted average precision score.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
    y_prob : array of shape (n_samples,) for binary or (n_samples, n_classes)
             for multiclass.
    """
    n_classes = len(np.unique(y_true))
    y_score = y_prob[:, 1] if (y_prob.ndim == 2 and n_classes == 2) else y_prob
    return float(average_precision_score(y_true, y_score, average="weighted"))
