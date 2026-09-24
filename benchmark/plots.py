"""The project's figure style and model names, and the figures that share them."""
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

# Two families get a hue, ordered light to dark by rank inside the family; the
# classical block stays neutral. Ramping it too would read as an ordering, and
# the whole point of the diagram is that the test cannot resolve one there.
# Steps are the validated blue and orange ordinal ramps for a light surface.
FOUNDATION_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
AUTOML_RAMP = ["#eb6834", "#a8431b"]


# The label every figure prints for a checkpoint key. Both notebooks read it,
# so a new model is named once.
MODEL_LABELS: dict[str, str] = {
    "svc":            "SVC (GridSearch)",
    "logreg":         "Logistic Regression (GridSearch)",
    "random_forest":  "Random Forest (Optuna)",
    "xgboost":        "XGBoost (Optuna)",
    "sgd":            "SGD (Optuna)",
    "catboost":       "CatBoost (Optuna)",
    "lgbm":           "LightGBM (Optuna)",
    "lgbm_linear":    "LightGBM Linear (Optuna)",
    "hgb":            "HistGradientBoosting (Optuna)",
    "resnet":         "ResNet (Optuna)",
    "tabnet":         "TabNet (Optuna)",
    "tabpfn3":        "TabPFN-3 (GridSearch)",
    "tabpfn35":       "TabPFN-3.5 (GridSearch)",
    "tabpfn35fast":   "TabPFN-3.5-fast (GridSearch)",
    "tabicl":         "TabICL (GridSearch)",
    "tabfm":          "TabFM (zero-shot)",
}

FOUNDATION_MODELS = frozenset(
    MODEL_LABELS[k] for k in ("tabfm", "tabicl", "tabpfn3", "tabpfn35", "tabpfn35fast")
)
AUTOML_MODELS = frozenset({"AutoGluon (sec=300)", "MLJAR (sec=300)"})


def _family_colours(models: list[str]) -> list[str]:
    """One colour per model, by family and by rank inside that family.

    *models* arrives in rank order, so the ramp is walked in that order and the
    darkest step goes to the family's best entry.
    """
    # Walked from the dark end, so the family's best entry is its darkest step
    # and the one that has to carry the reader's eye is not the faintest mark.
    ramps = {"foundation": FOUNDATION_RAMP[::-1], "automl": AUTOML_RAMP[::-1]}
    used = dict.fromkeys(ramps, 0)
    colours = []
    for model in models:
        family = ("foundation" if model in FOUNDATION_MODELS else
                  "automl" if model in AUTOML_MODELS else None)
        if family is None:
            colours.append(CONTEXT)
            continue
        ramp = ramps[family]
        colours.append(ramp[min(used[family], len(ramp) - 1)])
        used[family] += 1
    return colours


def style_axes(ax, grid: bool = True):
    """Recessive chrome: hairline grid, no frame, muted ticks."""
    ax.set_facecolor(SURFACE)
    # matplotlib enables the grid anyway if line properties come with grid=False.
    if grid:
        ax.grid(True, color=GRID, linewidth=1)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


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

    Colour carries the model family, shaded by rank inside it. The classical
    block stays one neutral: shading it would assert an ordering the bars
    covering it say the test cannot resolve.
    """
    fig, (ax, bars) = plt.subplots(
        1, 2, figsize=(15, 0.55 * len(ranks) + 3), sharey=True,
        gridspec_kw={"width_ratios": [9, 2], "wspace": 0.04},
    )

    models = list(ranks.index)
    colours = _family_colours(models)
    y = range(len(models))
    ax.hlines(y, ranks.min() - 0.4, ranks.values, color=GRID, linewidth=1.5, zorder=1)
    ax.scatter(ranks.values, y, s=90, color=colours, zorder=3)
    ax.set_xlim(ranks.min() - 0.4, ranks.max() + 0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(f"Average rank across {n_datasets} datasets (1 = best)")

    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                                  color=colour, label=name)
                       for name, colour in (("Foundation model", FOUNDATION_RAMP[2]),
                                            ("AutoML", AUTOML_RAMP[0]),
                                            ("Classical", CONTEXT))],
              # The marks run top-left to bottom-right, so this corner is the
              # empty one.
              loc="upper right", frameon=False, labelcolor=SECONDARY_INK)

    for offset, group in enumerate(cliques):
        rows = [models.index(m) for m in group]
        bars.plot([offset] * 2, [min(rows), max(rows)], color=SECONDARY_INK,
                  linewidth=3, solid_capstyle="round")
    bars.set_xlim(-0.6, max(len(cliques) - 0.4, 1))
    bars.set_xlabel("not separable", color=MUTED, fontsize="x-small")

    style_axes(ax)
    style_axes(bars, grid=False)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", colors=INK)
    bars.set_xticks([])

    fig.suptitle(
        "Critical difference: bars join models the data cannot separate\n"
        f"Wilcoxon signed-rank, Holm-adjusted, alpha = {alpha}",
        color=INK,
    )
    return fig
