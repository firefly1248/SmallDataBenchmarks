"""Significance testing for model comparisons over many datasets.

Follows Demsar (2006) as amended by Benavoli et al. (2016): Friedman as an
omnibus test, then pairwise Wilcoxon signed-rank with Holm correction. Nemenyi's
critical distance is not used because it compares average ranks, and a model's
average rank moves when an unrelated model joins the comparison. Wilcoxon reads
only the pair's own scores; the Holm multiplier still depends on how many pairs
are tested, so the family has to be chosen deliberately and stated.

The unit of observation is one dataset, never one fold: the four folds of a
dataset share their data and are not independent draws.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def average_ranks(scores: pd.DataFrame) -> pd.Series:
    """Mean rank per model, 1 = best. Rows are datasets, columns models."""
    return scores.rank(axis=1, ascending=False).mean().sort_values()


def friedman_p(scores: pd.DataFrame) -> float:
    """Omnibus p-value for "all models perform equally"."""
    return float(friedmanchisquare(*scores.values.T).pvalue)


def holm(pvalues: dict[tuple, float]) -> dict[tuple, float]:
    """Holm step-down adjustment, order preserved."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, (m - i) * p)
        adjusted[key] = min(running, 1.0)
    return adjusted


def wilcoxon_holm(scores: pd.DataFrame) -> pd.DataFrame:
    """``raw`` and ``holm`` p-values for every model pair.

    Indexed under both orientations of each pair, so no caller has to know which
    way round it was stored.
    """
    raw = {}
    for a, b in itertools.combinations(scores.columns, 2):
        diff = scores[a] - scores[b]
        # wilcoxon raises when every dataset ties; that is p = 1, not an error.
        raw[(a, b)] = 1.0 if np.all(diff == 0) else float(wilcoxon(diff).pvalue)

    adjusted = holm(raw)
    both_ways = {}
    for pair, p in raw.items():
        both_ways[pair] = both_ways[pair[::-1]] = {"raw": p, "holm": adjusted[pair]}
    return pd.DataFrame.from_dict(both_ways, orient="index").sort_index()


def cliques(ranks: pd.Series, adjusted: pd.DataFrame, alpha: float = 0.05) -> list[list[str]]:
    """Maximal groups of rank-adjacent models that are not significantly apart.

    These are the bars of a critical-difference diagram: a bar spanning two
    models means the data does not separate them. Only the order of *ranks*
    is read, not the rank values.
    """
    order = list(ranks.index)

    def differs(a: str, b: str) -> bool:
        return adjusted.loc[(a, b), "holm"] < alpha

    # Each span is the longest run starting at i, so the runs end no earlier as
    # i advances: a span is contained in its predecessor exactly when it ends at
    # the same place. Extending by one only needs the new model checked.
    spans, furthest = [], -1
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and not any(
            differs(order[k], order[j + 1]) for k in range(i, j + 1)
        ):
            j += 1
        if j > i and j > furthest:
            spans.append(order[i:j + 1])
            furthest = j
    return spans
