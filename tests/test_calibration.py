"""Tests for benchmark.calibration."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.calibration import brier_score, expected_calibration_error

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
