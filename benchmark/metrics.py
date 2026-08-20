"""Evaluation metrics and scorers for classification benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.preprocessing import label_binarize


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

    The branch is taken on the width of ``y_prob`` (what the model was trained
    on), not on the labels present in ``y_true``: a test fold missing a class
    would otherwise score a 3-class problem as binary. Multiclass targets are
    binarised against the trained label set for the same reason — bit-identical
    to raw ``y_true`` when every class is present.
    """
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        y_score = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        return float(average_precision_score(y_true, y_score, average="weighted"))

    y_true = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
    return float(average_precision_score(y_true, y_prob, average="weighted"))
