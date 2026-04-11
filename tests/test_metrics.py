"""Tests for benchmark.metrics."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from benchmark.metrics import PR_AUC_SCORER, pr_auc_score


class TestPrAucScore:
    def test_perfect_binary(self):
        y = np.array([0, 0, 1, 1])
        # Perfect: positive class prob = 1 where y=1, 0 where y=0
        prob = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        score = pr_auc_score(y, prob)
        assert score == pytest.approx(1.0)

    def test_binary_score_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        prob = rng.dirichlet([1, 1], size=100)
        score = pr_auc_score(y, prob)
        assert 0.0 <= score <= 1.0

    def test_multiclass_score_in_unit_interval(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=90)
        prob = rng.dirichlet([1, 1, 1], size=90)
        score = pr_auc_score(y, prob)
        assert 0.0 <= score <= 1.0

    def test_binary_uses_positive_class_column(self):
        """For binary, y_prob[:,1] must be used — not the full matrix."""
        y = np.array([0, 0, 1, 1])
        # col-1 perfectly separates — col-0 does the opposite
        prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        score = pr_auc_score(y, prob)
        assert score > 0.9


class TestPrAucScorer:
    """PR_AUC_SCORER must be usable inside cross_val_score."""

    def test_scorer_with_dummy_iris(self):
        X, y = load_iris(return_X_y=True)
        clf = DummyClassifier(strategy="prior")
        scores = cross_val_score(clf, X, y, cv=3, scoring=PR_AUC_SCORER)
        assert scores.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_scorer_binary(self, binary_df):
        X, y = binary_df
        clf = DummyClassifier(strategy="prior")
        scores = cross_val_score(clf, X, y, cv=3, scoring=PR_AUC_SCORER)
        assert all(0.0 <= s <= 1.0 for s in scores)
