"""How the ECE ranking of the calibration table depends on the binning: top-label ECE over
the 108 calibration datasets at 5 to 50 equal-width bins and at 15 equal-mass bins, with
each setting's Spearman correlation against the table's 15 equal-width bins.

Usage: uv run python -m scripts.ece_bins
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmark.calibration import expected_calibration_error
from benchmark.checkpoints import available_models, ckpt_path
from benchmark.plots import AUTOML_LABELS, MODEL_LABELS
from benchmark.results import CALIBRATION_POPULATION

SETTINGS = [5, 10, 15, 20, 30, 50, "equal-mass"]
population = set(pd.read_csv(CALIBRATION_POPULATION)["dataset"])
sources = [(MODEL_LABELS[k], ckpt_path(k)) for k in available_models() if k in MODEL_LABELS]
sources += [(label, f"results/{fw}_sec_300_ckpt.joblib") for fw, label in AUTOML_LABELS.items()]


def equal_mass(y, p, n_bins=15):
    confidence, correct = p.max(axis=1), (p.argmax(axis=1) == y).astype(float)
    bins = np.array_split(np.argsort(confidence), n_bins)
    return sum(abs(confidence[b].sum() - correct[b].sum()) for b in bins) / len(confidence)


table = {}
for label, path in sources:
    ece = {s: [] for s in SETTINGS}
    for d, e in joblib.load(path).items():
        if d in population:
            y, p = np.concatenate(e["labels"]), np.vstack(e["preds"])
            for s in SETTINGS:
                ece[s].append(equal_mass(y, p) if s == "equal-mass" else expected_calibration_error(y, p, s))
    table[label.split(" (")[0]] = {s: np.mean(v) for s, v in ece.items()}
table = pd.DataFrame(table).T
for s in SETTINGS:
    print(f"{str(s):10s} Spearman with 15 bins {spearmanr(table[s], table[15]).statistic:.3f}, "
          f"first five {list(table[s].sort_values().index[:5])}")
print(table.round(4).sort_values(15).to_string())
