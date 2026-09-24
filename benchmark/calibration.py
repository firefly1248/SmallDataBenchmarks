"""Calibration metrics and post-hoc calibrators over stored predictions.

PR AUC measures ranking only: a model that scores every positive above every
negative is perfect regardless of whether its 0.9 means 90 %. The two metrics
here read the probabilities themselves, and the two calibrators try to repair
them without refitting anything.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

Folds = list[tuple[np.ndarray, np.ndarray]]


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
    [0, 1]: a 3-class model never predicts below 1/3, so fixed [0, 1] bins would
    spend the lower third on empty bins and read the rest at a third of the
    resolution.
    """
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(confidence.min(), confidence.max(), n_bins + 1)
    binned = np.digitize(confidence, edges[1:-1])

    # Per bin, size * |mean confidence - mean accuracy| is the difference of the
    # two sums, so the bin sizes never have to be divided out.
    gap = np.abs(np.bincount(binned, weights=confidence, minlength=n_bins)
                 - np.bincount(binned, weights=correct, minlength=n_bins))
    return float(gap.sum() / len(confidence))


def _cross_fitted(folds: Folds, fit: Callable) -> list[np.ndarray]:
    """Recalibrate each fold with a map fitted on the other folds only.

    Fitting on the fold being corrected would let the calibrator read the labels
    it is scored against, which reports a repair that does not exist out of
    sample.
    """
    out = []
    for i, (prob, _) in enumerate(folds):
        rest = folds[:i] + folds[i + 1:]
        apply = fit(np.vstack([p for p, _ in rest]),
                    np.concatenate([y for _, y in rest]))
        out.append(apply(prob))
    return out


def _normalise(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-12, None)
    return prob / prob.sum(axis=1, keepdims=True)


def temperature_scale(folds: Folds) -> list[np.ndarray]:
    """Cross-fitted temperature scaling: one parameter, fitted by NLL.

    Divides the log probabilities by a scalar, so it can only sharpen or soften
    a model's confidence — it cannot reorder the classes within a row. On binary
    problems that leaves PR AUC untouched; on multiclass the one-vs-rest column
    depends on the other classes, so it can move a little.
    """
    def fit(prob: np.ndarray, y: np.ndarray) -> Callable:
        logits = np.log(np.clip(prob, 1e-12, None))
        rows = np.arange(len(y))

        def nll(log_t: float) -> float:
            z = logits / np.exp(log_t)
            return float(np.mean(logsumexp(z, axis=1) - z[rows, y]))

        temperature = np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)
        return lambda p: _normalise(np.clip(p, 1e-12, None) ** (1 / temperature))

    return _cross_fitted(folds, fit)


def isotonic_calibrate(folds: Folds) -> list[np.ndarray]:
    """Cross-fitted isotonic regression, one-vs-rest per class then renormalised.

    Free to bend the probabilities into any monotone shape, where temperature
    scaling gets one knob. A class absent from the fitting folds is mapped to
    zero and survives only as the floor ``_normalise`` applies.
    """
    def fit(prob: np.ndarray, y: np.ndarray) -> Callable:
        models = [
            IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            .fit(prob[:, k], (y == k).astype(float))
            for k in range(prob.shape[1])
        ]
        return lambda p: _normalise(
            np.column_stack([m.predict(p[:, k]) for k, m in enumerate(models)])
        )

    return _cross_fitted(folds, fit)
