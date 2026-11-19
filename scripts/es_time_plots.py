"""es_time_plots: the plan's three scatter plots (thin E2) and the unseen-level table, exploratory.

H1 against label persistence; H2 against the EF stopping set's minority-share gap to the test block and to the
older 90 %. Persistence is one value per dataset, so the Spearman values use the nine dataset means. The table
lists, per dataset, the share of stopping-set and test rows with a categorical level absent from the model's
training part (final ER and EF fits) beside H1.

Usage: python -m scripts.es_time_plots   (after es_time_analyse analyse; writes results/es_time/es_time_scatter.png)
"""
import joblib
import matplotlib
import pandas as pd
from scipy.stats import spearmanr

from scripts.es_time import OUT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("persistence", "h1", "label persistence", "H1 (D_T - DR_T)"),
          ("gap_test", "h2", "EF stop share minus test share", "H2 (D - EF)"),
          ("gap_older", "h2", "EF stop share minus older 90 % share", "H2 (D - EF)")]


def main():
    g = pd.read_csv(OUT / "stopping_set_gaps.csv").query("pop == 'thin'")
    means = g.groupby("dataset")[["persistence", "gap_test", "gap_older", "h1", "h2"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (x, y, xlabel, ylabel) in zip(axes, PANELS):
        for name, d in g.groupby("dataset"):
            ax.scatter(d[x], d[y], s=18, label=name)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        r = spearmanr(means[x], means[y])
        print(f"{y} vs {x}: Spearman over the 9 dataset means {r.statistic:+.2f} (p {r.pvalue:.2f})")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "es_time_scatter.png", dpi=130)

    fits = pd.concat([joblib.load(f) for f in sorted((OUT / "thin").glob("*.joblib"))])
    f = fits[(fits.scheme == "e2") & (fits.stage == "final")]
    unseen = f[f.arm.isin(["er", "ef"])].groupby(["dataset", "arm"])[["unseen_stop", "unseen_tgt"]].mean().unstack("arm").round(3)
    unseen.columns = ["_".join(c) for c in unseen.columns]
    h1 = f[f.arm.isin(["d_t", "dr_t"])].pivot_table(index="dataset", columns="arm", values="ap")
    print(unseen.join((h1.d_t - h1.dr_t).round(4).rename("H1")).to_string())


if __name__ == "__main__":
    main()
