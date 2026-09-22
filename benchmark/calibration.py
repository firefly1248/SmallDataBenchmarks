"""Calibration metrics over stored out-of-fold probabilities.

PR AUC measures ranking only: a model that scores every positive above every
negative is perfect regardless of whether its 0.9 means 90 %. These two metrics
read the probabilities themselves.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multiclass Brier score, 0 (perfect) to 2.

    ``labels`` comes from the width of ``y_prob``, so a test fold missing a
    class still scores against the full trained label set — the same reasoning
    as ``benchmark.metrics.pr_auc_score``.
    """
    return float(brier_score_loss(y_true, y_prob,
                                  labels=np.arange(y_prob.shape[1]),
                                  scale_by_half=False))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label ECE: |confidence - accuracy| per bin, weighted by bin size.

    Bins are equal-width over the confidence range actually observed, not over
    [0, 1]: a 3-class model never predicts below 1/3, and fixed [0, 1] bins
    would leave the lower third empty and compress the rest.
    """
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(confidence.min(), confidence.max(), n_bins + 1)
    binned = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)

    # Per bin, size * |mean confidence - mean accuracy| is the difference of the
    # two sums, so the bin sizes never have to be divided out.
    gap = np.abs(np.bincount(binned, weights=confidence, minlength=n_bins)
                 - np.bincount(binned, weights=correct, minlength=n_bins))
    return float(gap.sum() / len(confidence))
