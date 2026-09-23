"""Tests for benchmark.stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.stats import average_ranks, cliques, friedman_p, holm, wilcoxon_holm


def _pvalues(pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    """The shape wilcoxon_holm returns: every pair under both orientations."""
    both_ways = {}
    for pair, p in pairs.items():
        both_ways[pair] = both_ways[pair[::-1]] = {"raw": p, "holm": p}
    return pd.DataFrame.from_dict(both_ways, orient="index")


class TestHolm:
    def test_step_down_factors(self):
        adjusted = holm({"a": 0.01, "b": 0.03, "c": 0.04})
        assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.06),
                            "c": pytest.approx(0.06)}

    def test_monotone_and_capped(self):
        adjusted = holm({"a": 0.5, "b": 0.6})
        assert adjusted["a"] == pytest.approx(1.0)
        assert adjusted["b"] == pytest.approx(1.0)

    def test_less_conservative_than_bonferroni(self):
        raw = {"a": 0.01, "b": 0.02, "c": 0.03}
        assert holm(raw)["c"] < 3 * raw["c"]


class TestAverageRanks:
    def test_best_model_ranks_first(self):
        ranks = average_ranks(pd.DataFrame({"good": [0.9, 0.8], "bad": [0.1, 0.2]}))
        assert list(ranks.index) == ["good", "bad"]
        assert ranks["good"] == 1.0

    def test_ties_share_the_rank(self):
        ranks = average_ranks(pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}))
        assert ranks["a"] == ranks["b"] == 1.5


class TestWilcoxonHolm:
    def test_identical_columns_do_not_raise(self):
        """scipy raises on an all-zero difference; that is p = 1, not an error."""
        adjusted = wilcoxon_holm(pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]}))
        assert adjusted.loc[("a", "b"), "holm"] == 1.0

    def test_consistent_winner_is_significant(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.3, 0.9, size=40)
        adjusted = wilcoxon_holm(pd.DataFrame({"better": base + 0.01, "worse": base}))
        assert adjusted.loc[("better", "worse"), "holm"] < 0.05

    def test_both_orientations_are_present_and_equal(self):
        adjusted = wilcoxon_holm(pd.DataFrame({"a": [1.0, 2, 3], "b": [2.0, 1, 3],
                                               "c": [3.0, 2, 1]}))
        assert len(adjusted) == 6
        for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
            assert adjusted.loc[(a, b), "holm"] == adjusted.loc[(b, a), "holm"]

    def test_correction_only_raises_the_p_value(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(0.3, 0.9, size=30)
        adjusted = wilcoxon_holm(pd.DataFrame({"a": base + 0.01, "b": base,
                                               "c": base - 0.01}))
        assert (adjusted["holm"] >= adjusted["raw"]).all()


class TestCliques:
    def test_groups_only_the_inseparable(self):
        ranks = pd.Series({"first": 1.0, "second": 2.0, "third": 3.0})
        adjusted = _pvalues({("first", "second"): 0.01, ("first", "third"): 0.01,
                             ("second", "third"): 0.9})
        assert cliques(ranks, adjusted) == [["second", "third"]]

    def test_a_contained_group_is_dropped(self):
        ranks = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        adjusted = _pvalues({("a", "b"): 0.9, ("a", "c"): 0.9, ("b", "c"): 0.9})
        assert cliques(ranks, adjusted) == [["a", "b", "c"]]

    def test_a_group_needs_every_pair_inside_it(self):
        """The bar stops at c: a and c separate, however close their ranks."""
        ranks = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        adjusted = _pvalues({("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "c"): 0.01})
        assert cliques(ranks, adjusted) == [["a", "b"], ["b", "c"]]

    def test_all_separable_gives_no_bars(self):
        ranks = pd.Series({"a": 1.0, "b": 2.0})
        assert cliques(ranks, _pvalues({("a", "b"): 0.001})) == []


class TestFriedman:
    def test_detects_a_consistent_ordering(self):
        rng = np.random.default_rng(1)
        base = rng.uniform(0.3, 0.9, size=30)
        assert friedman_p(pd.DataFrame({"a": base + 0.05, "b": base,
                                        "c": base - 0.05})) < 0.001
