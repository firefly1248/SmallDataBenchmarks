"""The project's figure style, and the figures that share it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#2a78d6"
CONTEXT = "#a8a7a0"


def style_axes(ax):
    """Recessive chrome: hairline grid, no frame, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def critical_difference_diagram(
    ranks: pd.Series,
    cliques: list[list[str]],
    n_datasets: int,
    alpha: float,
):
    """Average ranks, with a bar over each group the test cannot separate.

    Demsar's layout puts the models on leader lines around a single axis, which
    collides with itself past about eight models; rows carry the same
    information and stay readable at twenty. The bars live in their own panel
    so they cannot be read as positions on the rank axis.
    """
    fig, (ax, bars) = plt.subplots(
        1, 2, figsize=(15, 0.55 * len(ranks) + 3), sharey=True,
        gridspec_kw={"width_ratios": [9, 2], "wspace": 0.04},
    )

    models = list(ranks.index)
    y = range(len(models))
    ax.hlines(y, ranks.min() - 0.4, ranks.values, color=GRID, linewidth=1.5, zorder=1)
    ax.scatter(ranks.values, y, s=90, color=ACCENT, zorder=3)
    ax.set_xlim(ranks.min() - 0.4, ranks.max() + 0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(f"Average rank across {n_datasets} datasets (1 = best)")

    for offset, group in enumerate(cliques):
        rows = [models.index(m) for m in group]
        bars.plot([offset] * 2, [min(rows), max(rows)], color=SECONDARY_INK,
                  linewidth=3, solid_capstyle="round")
    bars.set_xlim(-0.6, max(len(cliques) - 0.4, 1))
    bars.set_xlabel("not separable", color=MUTED, fontsize="x-small")

    style_axes(ax)
    style_axes(bars)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", colors=INK)
    bars.grid(False)
    bars.set_xticks([])

    fig.suptitle(
        "Critical difference: bars join models the data cannot separate\n"
        f"Wilcoxon signed-rank, Holm-adjusted, alpha = {alpha}",
        color=INK,
    )
    return fig
