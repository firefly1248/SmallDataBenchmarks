"""Tests for the subsets, carve, ranking and candidate draws of scripts/search_flatness.py."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from benchmark.models.objectives import lgbm_space
from scripts.search_flatness import N_CAND, candidates, carve, halving, ranked, rk_weights, subset

rng = np.random.default_rng(0)
y = np.repeat([0, 1, 2], [200, 37, 5])
idx = np.sort(rng.choice(len(y), 180, replace=False))


def test_subsets_nested_stratified_and_ordered():
    parts = {den: subset(y, idx, den) for den in (9, 3, 1)}
    assert set(parts[9]) <= set(parts[3]) <= set(parts[1]) == set(idx)
    for den, part in parts.items():
        assert np.all(np.diff(part) > 0) and set(part) <= set(idx)
        for c in np.unique(y[idx]):
            assert (y[part] == c).sum() == max(1, (y[idx] == c).sum() // den)


def test_subset_keeps_a_row_of_every_class():
    assert np.unique(y[subset(y, idx, 9)]).size == np.unique(y[idx]).size


def test_carve_is_stratified_disjoint_and_ordered():
    keep, stop = carve(y)
    assert len(np.intersect1d(keep, stop)) == 0 and len(keep) + len(stop) == len(y)
    assert np.all(np.diff(keep) > 0) and np.all(np.diff(stop) > 0)
    assert len(stop) == int(np.ceil(0.1 * len(y)))
    assert abs((y[stop] == 0).mean() - (y == 0).mean()) < 0.05


def test_ranked_nan_last_ties_by_position():
    assert list(ranked([0.5, np.nan, 0.7, 0.7, 0.1])) == [2, 3, 0, 4, 1]


def test_rk_weights():
    assert np.isclose(rk_weights(20).sum(), 1)
    assert np.allclose(rk_weights(1), 1 / N_CAND)
    assert rk_weights(N_CAND)[0] == 1
    assert np.all(np.diff(rk_weights(20)) <= 0)


def test_candidates_reproducible_and_in_space():
    a, b = candidates("blogger", 0), candidates("blogger", 0)
    assert a == b and len(a) == N_CAND
    assert a != candidates("blogger", 1) and a != candidates("colon32", 0)
    space = lgbm_space()
    for params in a:
        assert set(params) == set(space)
        assert all(space[k]._contains(space[k].to_internal_repr(v)) for k, v in params.items())


def test_rk_weights_match_enumeration():
    """With N_CAND = 60, k = 3 enumerates all C(60, 3) subsets: the best-ranked member wins."""
    wins = np.zeros(N_CAND)
    for sub in combinations(range(N_CAND), 3):
        wins[min(sub)] += 1
    assert np.allclose(rk_weights(3), wins / wins.sum())


def test_halving_breaks_ties_by_candidate_index():
    r1 = np.zeros(N_CAND)
    r1[[59, 45, 3]] = 1.0  # 59 ranks first in round 1, then 45 and 3 by index
    r2 = np.zeros(N_CAND)
    r2[[59, 45]] = 0.5     # tied in round 2
    m = np.zeros(N_CAND)
    m[[59, 45]] = 0.9      # and tied in the R60 inner mean
    top20, top7, pick = halving(r1, r2, m)
    assert list(top20) == sorted(top20) and list(top7) == sorted(top7)
    assert pick == 45
    assert set(top7) == {59, 45, 0, 1, 2, 3, 4}
