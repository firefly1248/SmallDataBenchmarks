"""Evaluation metrics and scorers for classification benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.preprocessing import label_binarize


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Weighted average precision.

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


def pr_auc_baseline(y_true: np.ndarray, n_classes: int) -> float:
    """What a constant predictor scores: the floor this metric starts from.

    ROC AUC starts at 0.5 whatever the data, so a weak model is obvious from the
    number alone. Average precision starts at the class prevalence, which spans
    0.01 to 0.95 across these datasets — a score only means something next to
    this. A run sitting on its baseline has learned nothing, however respectable
    it looks in a column of means.
    """
    shares = np.bincount(y_true, minlength=n_classes) / len(y_true)
    if n_classes == 2:
        return float(shares[1])
    return float(np.sum(shares ** 2))


PR_AUC_SCORER = make_scorer(pr_auc_score, response_method="predict_proba")
"""``pr_auc_score`` as an sklearn scorer.

Wrapping ``average_precision_score`` directly instead raises on any inner fold
whose labels miss a trained class: sklearn calls the metric binary on the fold's
labels while passing the full probability matrix.
"""
