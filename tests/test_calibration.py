"""Tests for benchmark.calibration."""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.isotonic import IsotonicRegression

from benchmark.calibration import (
    _ivap,
    brier_score,
    expected_calibration_error,
    isotonic_calibrate,
    temperature_scale,
    venn_abers_calibrate,
)
from benchmark.metrics import pr_auc_score

Y = np.array([0, 0, 1, 1])
CONFIDENT_RIGHT = np.array([[1.0, 0], [1, 0], [0, 1], [0, 1]])
CONFIDENT_WRONG = CONFIDENT_RIGHT[:, ::-1]


class TestBrierScore:
    def test_bounds(self):
        assert brier_score(Y, CONFIDENT_RIGHT) == 0.0
        assert brier_score(Y, CONFIDENT_WRONG) == 2.0

    def test_uninformative(self):
        assert brier_score(Y, np.full((4, 2), 0.5)) == pytest.approx(0.5)

    def test_scores_against_the_trained_label_set(self):
        """A fold missing a class must still be scored as 3-class."""
        prob = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1],
                         [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
        expected = np.mean(np.sum((prob - np.eye(3)[Y]) ** 2, axis=1))
        assert brier_score(Y, prob) == pytest.approx(expected)


class TestExpectedCalibrationError:
    def test_confident_and_right_is_calibrated(self):
        assert expected_calibration_error(Y, CONFIDENT_RIGHT) == 0.0

    def test_confident_and_wrong_is_maximally_miscalibrated(self):
        assert expected_calibration_error(Y, CONFIDENT_WRONG) == pytest.approx(1.0)

    def test_coin_flip_predictor_is_calibrated(self):
        """Half right at 50 % confidence is honest, however useless."""
        assert expected_calibration_error(Y, np.full((4, 2), 0.5)) == 0.0

    def test_overconfidence_is_the_gap(self):
        """Claims 0.9, right half the time."""
        prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
        assert expected_calibration_error(Y, prob) == pytest.approx(0.4)

    def test_constant_confidence(self):
        """One occupied bin: the whole gap is that bin's."""
        prob = np.full((10, 2), 0.5)
        assert expected_calibration_error(np.zeros(10, dtype=int), prob) == pytest.approx(0.5)


def _overconfident(n=2000, sharpen=3.0, seed=0):
    """A binary model with real signal whose logits are inflated *sharpen* times."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(scale=0.15, size=n), 0.02, 0.98)
    p = 1 / (1 + np.exp(-np.log(p / (1 - p)) * sharpen))
    prob = np.column_stack([1 - p, p])
    return [(prob[i::4], y[i::4]) for i in range(4)]


def _mean_ece(folds, probs):
    return np.mean([expected_calibration_error(y, p) for p, (_, y) in zip(probs, folds)])


class TestPostHocCalibration:
    @pytest.mark.parametrize("calibrate", [temperature_scale, isotonic_calibrate,
                                           venn_abers_calibrate])
    def test_repairs_an_overconfident_model(self, calibrate):
        folds = _overconfident()
        before = _mean_ece(folds, [p for p, _ in folds])
        assert _mean_ece(folds, calibrate(folds)) < before / 2

    @pytest.mark.parametrize("calibrate", [temperature_scale, isotonic_calibrate,
                                           venn_abers_calibrate])
    def test_returns_distributions_of_the_same_shape(self, calibrate):
        folds = _overconfident(n=400)
        for (prob, _), out in zip(folds, calibrate(folds)):
            assert out.shape == prob.shape
            assert out.sum(axis=1) == pytest.approx(np.ones(len(out)))

    def test_temperature_scaling_keeps_the_binary_ranking(self):
        """One knob on the logits cannot reorder rows, so PR AUC must not move."""
        folds = _overconfident()
        for (prob, y), out in zip(folds, temperature_scale(folds)):
            assert pr_auc_score(y, out) == pytest.approx(pr_auc_score(y, prob))

    def test_calibrator_never_sees_the_fold_it_corrects(self):
        """Labels flipped in one fold only: a leaking fit would repair it anyway."""
        folds = _overconfident()
        poisoned = list(folds)
        prob, y = poisoned[0]
        poisoned[0] = (prob, 1 - y)
        assert _mean_ece(poisoned, temperature_scale(poisoned)) > _mean_ece(
            folds, temperature_scale(folds))

    def test_handles_a_class_absent_from_the_fitting_folds(self):
        rng = np.random.default_rng(0)
        prob = rng.dirichlet(np.ones(3), size=400)
        y = prob.argmax(axis=1)
        y[y == 2] = 0            # class 2 is predicted, never observed
        folds = [(prob[i::4], y[i::4]) for i in range(4)]
        for calibrate in (isotonic_calibrate, venn_abers_calibrate):
            for out in calibrate(folds):
                assert np.isfinite(out).all()
                assert out.sum(axis=1) == pytest.approx(np.ones(len(out)))


def test_isotonic_keeps_float32_scores_apart():
    """Scores 1e-7 apart in float32 fall inside sklearn's merge tolerance for that dtype."""
    prob = np.array([[0.5, 0.5], [0.5 - 1e-7, 0.5 + 1e-7]] * 20, dtype=np.float32)
    y = np.array([0, 1] * 20)
    folds = [(prob, y)] * 4
    for out in isotonic_calibrate(folds):
        assert len(np.unique(out[:, 1])) == 2


class TestVennAbers:
    def test_matches_a_full_refit_per_test_point(self):
        """Coarse rounding forces ties, the case the knot merging has to get right."""
        rng = np.random.default_rng(1)
        cal = np.round(rng.uniform(size=80), 1)
        labels = (rng.uniform(size=80) < cal).astype(float)
        test = np.round(rng.uniform(size=15), 1)
        lower, upper = _ivap(cal, labels, test)
        for j, t in enumerate(test):
            for label, got in ((0, lower[j]), (1, upper[j])):
                fit = IsotonicRegression(y_min=0, y_max=1).fit(
                    np.append(cal, t), np.append(labels, label))
                assert got == pytest.approx(fit.predict([t])[0], abs=1e-12)

    def test_interval_is_ordered(self):
        rng = np.random.default_rng(2)
        cal = rng.uniform(size=200)
        lower, upper = _ivap(cal, (rng.uniform(size=200) < cal).astype(float),
                             rng.uniform(size=100))
        assert np.all(lower <= upper)

    def test_binary_is_the_canonical_one_column_method(self):
        """Both columns calibrated and renormalised must equal Venn-ABERS on column 1 alone."""
        folds = _overconfident(n=400)
        for i, out in enumerate(venn_abers_calibrate(folds)):
            rest = folds[:i] + folds[i + 1:]
            lower, upper = _ivap(np.concatenate([p[:, 1] for p, _ in rest]),
                                 np.concatenate([y for _, y in rest]), folds[i][0][:, 1])
            assert out[:, 1] == pytest.approx(upper / (1 - lower + upper), abs=1e-12)

    def test_keeps_more_ranking_than_isotonic(self):
        folds = _overconfident()

        def distinct(probs):
            return sum(len(np.unique(p[:, 1])) for p in probs)

        assert distinct(venn_abers_calibrate(folds)) > 2 * distinct(isotonic_calibrate(folds))
