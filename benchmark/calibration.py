"""Calibration metrics and post-hoc calibrators over stored predictions.

PR AUC measures ranking only: a model that scores every positive above every
negative is perfect regardless of whether its 0.9 means 90 %. The two metrics
here read the probabilities themselves, and the three calibrators try to repair
them without refitting anything.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression, isotonic_regression
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


def confidence_gap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean top-class probability minus the share of rows whose top class is right.

    Negative: underconfident. ECE reads the size of the miscalibration; this reads
    its direction.
    """
    return float(y_prob.max(axis=1).mean() - (y_prob.argmax(axis=1) == y_true).mean())


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
    problems that leaves PR AUC untouched except where an extreme temperature
    rounds neighbouring scores together; on multiclass the one-vs-rest column
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
        # sklearn merges scores closer than its dtype's resolution: 1e-6 on the
        # float32 the torch models emit, which ties scores the model kept apart.
        prob = prob.astype(np.float64)
        models = [
            IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            .fit(prob[:, k], (y == k).astype(float))
            for k in range(prob.shape[1])
        ]
        return lambda p: _normalise(
            np.column_stack([m.predict(p[:, k]) for k, m in enumerate(models)])
        )

    return _cross_fitted(folds, fit)


def _ivap(cal_scores: np.ndarray, cal_labels: np.ndarray,
          test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inductive Venn-ABERS: the isotonic fit at a test point forced to 0 and to 1.

    This is what separates it from plain isotonic — the point being calibrated
    joins the calibration set before the fit, once with each label, so the pair
    brackets what a calibrator that already knew the answer would have said.
    Refitting once per test point is exact rather than approximated: equal
    scores merge into one weighted knot, so the fit is a weighted PAVA over the
    distinct scores and the inserted point is read straight off it.
    """
    scores, inverse = np.unique(cal_scores, return_inverse=True)
    weight = np.bincount(inverse).astype(float)
    label_sum = np.bincount(inverse, weights=cal_labels.astype(float))

    unique_test, back = np.unique(test_scores, return_inverse=True)
    at = np.searchsorted(scores, unique_test)
    merges = at < len(scores)
    merges[merges] = scores[at[merges]] == unique_test[merges]

    n = len(scores)
    bounds = np.empty((2, len(unique_test)))
    w_buf, sum_buf = np.empty(n + 1), np.empty(n + 1)
    for j, (k, merged) in enumerate(zip(at, merges)):
        if merged:
            size = n
            w_buf[:n], sum_buf[:n] = weight, label_sum
            w_buf[k] += 1.0
        else:
            size = n + 1
            w_buf[:k], sum_buf[:k] = weight[:k], label_sum[:k]
            w_buf[k] = 1.0
            w_buf[k + 1:size], sum_buf[k + 1:size] = weight[k:], label_sum[k:]
        for label in (0, 1):
            sum_buf[k] = (label_sum[k] if merged else 0.0) + label
            bounds[label, j] = isotonic_regression(
                sum_buf[:size] / w_buf[:size], sample_weight=w_buf[:size],
                y_min=0.0, y_max=1.0)[k]
    return bounds[0][back], bounds[1][back]


def venn_abers_calibrate(folds: Folds) -> list[np.ndarray]:
    """Cross-fitted Venn-ABERS, one-vs-rest then renormalised.

    Applied exactly like ``isotonic_calibrate``, on the same isotonic machinery,
    so the two differ only in construction. What that buys is resolution: plain
    isotonic maps a whole score interval onto one plateau, where
    ``p1 / (1 - p0 + p1)`` still separates the points inside it. On binary
    problems the two columns' predictions already sum to one, up to the rounding
    in the model's own output, so this is the canonical one-column method; on
    multiclass the renormalisation is a heuristic.
    """
    def fit(prob: np.ndarray, y: np.ndarray) -> Callable:
        def apply(p: np.ndarray) -> np.ndarray:
            columns = []
            for k in range(p.shape[1]):
                lower, upper = _ivap(prob[:, k], (y == k).astype(float), p[:, k])
                columns.append(upper / (1.0 - lower + upper))
            return _normalise(np.column_stack(columns))
        return apply

    return _cross_fitted(folds, fit)
