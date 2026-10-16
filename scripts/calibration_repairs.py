"""Post-hoc repairs per (model, dataset): the long table behind the calibration table,
the calibration figure and the calibration findings.

Writes ``results/calibration_datasets.csv``, the datasets every model with stored
predictions scores among the figure population, and ``results/calibration_repairs.csv``
with PR AUC, Brier and ECE per model, dataset and variant (raw, temperature, isotonic,
Venn-ABERS). ``benchmark.results.calibration_table`` reads the latter. Also runs the
dataset-level tests of Venn-ABERS against isotonic that Findings_notes.md quotes.

A backup checkpoint passed as ``model=path`` is also scored, and the tests are
repeated with it in place of that model, e.g. the uniform-start CatBoost run:

Usage: uv run python -m scripts.calibration_repairs \\
           [catboost=results/ckpt/catboost.joblib.bak-uniform-start]
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

from benchmark.calibration import (brier_score, expected_calibration_error, isotonic_calibrate,
                                   temperature_scale, venn_abers_calibrate)
from benchmark.checkpoints import available_models, ckpt_path
from benchmark.metrics import pr_auc_score
from benchmark.plots import AUTOML_LABELS, MODEL_LABELS
from benchmark.results import CALIBRATION_POPULATION, CALIBRATION_REPAIRS, automl_path, scores

sources = [(MODEL_LABELS[k], ckpt_path(k)) for k in available_models() if k in MODEL_LABELS]
sources += [(label, automl_path(fw, checkpoint=True)) for fw, label in AUTOML_LABELS.items()]
backup = dict(arg.split("=", 1) for arg in sys.argv[1:])
sources += [(f"{MODEL_LABELS[m]} backup", path) for m, path in backup.items()]


def rows_for(label, path, population):
    # AutoML stores float32 probabilities whose rows sum to one only within 6e-7.
    warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
    rows = []
    for dataset, entry in joblib.load(path).items():
        if dataset not in population:
            continue
        folds = list(zip(entry["preds"], entry["labels"]))
        y = np.concatenate(entry["labels"])
        for variant, probs in (("raw", entry["preds"]), ("temperature", temperature_scale(folds)),
                               ("isotonic", isotonic_calibrate(folds)),
                               ("venn_abers", venn_abers_calibrate(folds))):
            prob = np.vstack(probs)
            # PR AUC is the mean over folds, as everywhere else; Brier and ECE pool the
            # folds, because a 300-row fold spread over 15 bins is mostly noise.
            rows.append(dict(model=label, dataset=dataset, variant=variant, n_classes=prob.shape[1],
                             prauc=np.mean([pr_auc_score(yf, pf) for pf, (_, yf) in zip(probs, folds)]),
                             brier=brier_score(y, prob), ece=expected_calibration_error(y, prob)))
    return rows


def report(long: pd.DataFrame) -> None:
    w = long.pivot_table(index=["model", "dataset", "n_classes"], columns="variant",
                         values=["prauc", "brier", "ece"]).reset_index()
    gain = pd.DataFrame({"model": w.model, "dataset": w.dataset, "binary": w.n_classes == 2,
                         "prauc": w[("prauc", "venn_abers")] - w[("prauc", "isotonic")],
                         "brier": w[("brier", "isotonic")] - w[("brier", "venn_abers")],
                         "ece": w[("ece", "isotonic")] - w[("ece", "venn_abers")]})
    by_ds = gain.groupby("dataset").prauc.mean()
    print(f"{w.model.nunique()} models, {w.dataset.nunique()} datasets. PR AUC from raw: "
          f"Venn-ABERS {np.mean(w[('prauc', 'venn_abers')] - w[('prauc', 'raw')]):+.4f}, "
          f"isotonic {np.mean(w[('prauc', 'isotonic')] - w[('prauc', 'raw')]):+.4f}; Venn-ABERS ahead "
          f"for {(gain.groupby('model').prauc.mean() > 0).sum()} models and on {(by_ds > 0).sum()} "
          f"datasets (p = {wilcoxon(by_ds).pvalue:.1g})")
    for binary, task in ((True, "binary"), (False, "multiclass")):
        g = gain[gain.binary == binary]
        for metric in ("brier", "ece"):
            ds = g.groupby("dataset")[metric].mean()
            print(f"  {task:10s} {metric:5s}: Venn-ABERS ahead for {(g.groupby('model')[metric].mean() > 0).sum()} "
                  f"models and on {(ds > 0).sum()} of {len(ds)} datasets (p = {wilcoxon(ds).pvalue:.1g})")
    raw = long[(long.variant == "raw") & (long.n_classes == 2)]
    raw_brier = raw.groupby("model").brier.mean()
    edge = gain[gain.binary].groupby("model").brier.mean()
    print(f"  binary Brier edge against raw Brier, Spearman over models: "
          f"{spearmanr(raw_brier, edge.loc[raw_brier.index]).statistic:.2f}")


if __name__ == "__main__":
    # The figure population, then the datasets every model with predictions scores.
    # Checkpoints are read twice, here for the names and in rows_for for the
    # predictions: holding all of them at once is ~470 MB.
    figure_datasets = set(scores(automl=True)["dataset"])
    scored = [set(d for d, e in joblib.load(path).items() if e.get("preds") is not None)
              for label, path in sources if not label.endswith(" backup")]
    population = sorted(figure_datasets.intersection(*scored))
    pd.Series(population, name="dataset").to_csv(CALIBRATION_POPULATION, index=False)
    rows = Parallel(n_jobs=8)(delayed(rows_for)(label, path, set(population)) for label, path in sources)
    long = pd.DataFrame([r for rs in rows for r in rs])
    long.to_csv(CALIBRATION_REPAIRS, index=False)
    print(f"Saved {CALIBRATION_POPULATION} ({len(population)} datasets) and {CALIBRATION_REPAIRS}\n")
    backups = {f"{MODEL_LABELS[m]} backup": MODEL_LABELS[m] for m in backup}
    print("Published family:")
    report(long[~long.model.isin(backups)])
    for label, replaced in backups.items():
        print(f"\nWith {label} in place of {replaced}:")
        report(long[long.model != replaced])
