"""Post-hoc repairs per (model, dataset): the long table behind the calibration findings.

``figures.ipynb`` computes the same rows to draw the calibration table and figure;
this script saves them, parallel over models, and adds the dataset-level tests of
Venn-ABERS against isotonic that Findings_notes.md quotes. The population is
``results/calibration_datasets.csv``, the datasets every model scores.

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
                                   venn_abers_calibrate)
from benchmark.checkpoints import available_models, ckpt_path
from benchmark.metrics import pr_auc_score
from benchmark.plots import AUTOML_LABELS, MODEL_LABELS

OUTPUT = "results/calibration_repairs.csv"
population = set(pd.read_csv("results/calibration_datasets.csv")["dataset"])
sources = [(MODEL_LABELS[k], ckpt_path(k)) for k in available_models() if k in MODEL_LABELS]
sources += [(label, f"results/{fw}_sec_300_ckpt.joblib") for fw, label in AUTOML_LABELS.items()]
backup = dict(arg.split("=", 1) for arg in sys.argv[1:])
sources += [(f"{MODEL_LABELS[m]} backup", path) for m, path in backup.items()]


def rows_for(label, path):
    warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
    rows = []
    for dataset, entry in joblib.load(path).items():
        if dataset not in population:
            continue
        folds = list(zip(entry["preds"], entry["labels"]))
        y = np.concatenate(entry["labels"])
        for variant, probs in (("raw", entry["preds"]), ("isotonic", isotonic_calibrate(folds)),
                               ("venn_abers", venn_abers_calibrate(folds))):
            prob = np.vstack(probs)
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
    rows = Parallel(n_jobs=8)(delayed(rows_for)(label, path) for label, path in sources)
    long = pd.DataFrame([r for rs in rows for r in rs])
    long.round(6).to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}\n")
    backups = {f"{MODEL_LABELS[m]} backup": MODEL_LABELS[m] for m in backup}
    print("Published family:")
    report(long[~long.model.isin(backups)])
    for label, replaced in backups.items():
        print(f"\nWith {label} in place of {replaced}:")
        report(long[long.model != replaced])
