"""The published results as frames: what the figures and the analysis scripts read.

Scores come from the tracked aggregates, so a fresh clone reproduces the figures.
Calibration and per-dataset detail come from the checkpoints, which hold the
predictions and are not tracked.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.calibration import brier_score, confidence_gap, expected_calibration_error
from benchmark.plots import AUTOML_LABELS, MODEL_LABELS
from benchmark.stats import holm
from config import AUTOML_SEC, DUPLICATE_DATASETS, MAX_DATASET_ROWS, TRIVIAL_PR_AUC

BASELINE = "results/compare_baseline_models.joblib"
OPTUNA = "results/optuna_models.joblib"
CALIBRATION_POPULATION = "results/calibration_datasets.csv"
CALIBRATION_REPAIRS = "results/calibration_repairs.csv"
# The untuned GridSearch entries of compare_baseline_models.py, in its file order.
BASELINE_LABELS = ["SVC", "Logistic Regression", "Random Forest"]
FOLD_COLS = [f"prauc_split_{i}" for i in range(1, 5)]
DELTA_COLS = [f"delta_split_{i}" for i in range(1, 5)]


def automl_path(framework: str, checkpoint: bool = False) -> str:
    return f"results/{framework}_sec_{AUTOML_SEC}{'_ckpt' if checkpoint else ''}.joblib"


def _runs(automl: bool):
    """(label, fold scores, fold times, dataset names) per model of the roster."""
    *baseline, datasets, times = joblib.load(BASELINE)
    for label, arr in zip(BASELINE_LABELS, baseline):
        yield label, arr, times, datasets
    results, times, datasets = joblib.load(OPTUNA)
    for key, label in MODEL_LABELS.items():
        if key in results:
            yield label, results[key], times[key], datasets
    if automl:
        for framework, label in AUTOML_LABELS.items():
            results, times, datasets = joblib.load(automl_path(framework))
            yield label, results, times, datasets


def scores(automl: bool) -> pd.DataFrame:
    """One row per (model, dataset): the four outer-fold PR AUCs, their mean, min, max
    and std, the gap to the GridSearch Random Forest on every fold, and the dataset's
    size, features and imbalance from database.json. The rows are the UCI++ datasets
    minus the duplicates and minus the ones every model of this roster finds trivial.
    """
    frames = []
    for label, arr, _, names in _runs(automl):
        frame = pd.DataFrame(arr, columns=FOLD_COLS, index=names)
        frame["model"] = label
        frames.append(frame)
    df = pd.concat(frames)
    df["dataset"] = df.index
    database = pd.read_json("database.json").T
    database["nrow"] = np.minimum(MAX_DATASET_ROWS, database["nrow"])
    # map, not join: the index repeats once per model per dataset.
    for col in database.columns:
        df[col] = df.index.map(database[col]).to_numpy()
    df["mean_prauc"] = df[FOLD_COLS].mean(axis=1)
    df["min_prauc"] = df[FOLD_COLS].min(axis=1)
    df["max_prauc"] = df[FOLD_COLS].max(axis=1)
    df["std_prauc"] = df[FOLD_COLS].std(axis=1)
    rf = df[df.model == "Random Forest"].set_index("dataset")[FOLD_COLS]
    for delta, fold in zip(DELTA_COLS, FOLD_COLS):
        df[delta] = df[fold].to_numpy() - df.index.map(rf[fold]).to_numpy()
    df["mean_delta"] = df[DELTA_COLS].mean(axis=1)
    df["std_delta"] = df[DELTA_COLS].std(axis=1)
    # Integer index: duplicate dataset labels break seaborn's internals.
    df = df.reset_index(drop=True)
    df = df.groupby("dataset").filter(lambda x: x["mean_prauc"].min() < TRIVIAL_PR_AUC)
    return df[~df["dataset"].isin(DUPLICATE_DATASETS)].reset_index(drop=True)


def mean_times(automl: bool) -> dict[str, float]:
    """Mean seconds per dataset, over every dataset a model ran, duplicates included."""
    return {label: float(np.nanmean(times)) for label, _, times, _ in _runs(automl)
            if label not in BASELINE_LABELS}


def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """dataset x model mean PR AUC over the datasets every model of the frame scores."""
    return df.pivot_table(index="dataset", columns="model", values="mean_prauc").dropna()


def calibration_table() -> pd.DataFrame:
    """The calibration table of the README, from scripts/calibration_repairs.py's output:
    raw PR AUC, Brier and ECE per model, ECE after each repair, PR AUC after the two
    isotonic ones, sorted by raw ECE."""
    long = pd.read_csv(CALIBRATION_REPAIRS)
    long = long[~long["model"].str.endswith(" backup")]
    table = long.pivot_table(index="model", columns="variant", values=["prauc", "brier", "ece"])
    return pd.DataFrame({
        "PR AUC": table[("prauc", "raw")], "Brier": table[("brier", "raw")],
        "ECE": table[("ece", "raw")], "ECE temp": table[("ece", "temperature")],
        "ECE iso": table[("ece", "isotonic")], "ECE VA": table[("ece", "venn_abers")],
        "PR AUC iso": table[("prauc", "isotonic")], "PR AUC VA": table[("prauc", "venn_abers")],
    }).sort_values("ECE")


def per_dataset(entries: dict, calibration: set[str]) -> pd.DataFrame:
    """Per dataset of a checkpoint: mean PR AUC over the folds, seconds, and on the
    ``calibration`` datasets ECE, Brier and |confidence gap| over the pooled folds."""
    rows = {}
    for dataset, e in entries.items():
        if e.get("scores") is None or np.isnan(np.mean(e["scores"])):
            continue
        row = {"prauc": np.mean(e["scores"]), "time": e["time"]}
        if dataset in calibration:
            y, p = np.concatenate(e["labels"]), np.vstack(e["preds"])
            row |= {"ece": expected_calibration_error(y, p), "brier": brier_score(y, p),
                    "gap": abs(confidence_gap(y, p))}
        rows[dataset] = row
    return pd.DataFrame(rows).T


def compare(pairs: dict[str, tuple[dict, dict]], calibration: set[str]
            ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Each variant against its base over every dataset both score: mean PR AUC,
    wins, Wilcoxon with Holm over the given pairs as one family, hours, and the
    calibration deltas over the calibration datasets. Also the per-dataset PR AUC
    differences, keyed like ``pairs``."""
    rows, diffs = [], {}
    for label, (variant, base) in pairs.items():
        v, b = per_dataset(variant, calibration), per_dataset(base, calibration)
        both = v.index.intersection(b.index)
        diff = v.loc[both, "prauc"] - b.loc[both, "prauc"]
        row = {"comparison": label, "datasets": len(both),
               "prauc_base": b.loc[both, "prauc"].mean(), "prauc_variant": v.loc[both, "prauc"].mean(),
               "ahead": int((diff > 0).sum()), "behind": int((diff < 0).sum()),
               "p_raw": wilcoxon(diff[diff != 0]).pvalue,
               "hours_base": b.loc[both, "time"].sum() / 3600,
               "hours_variant": v.loc[both, "time"].sum() / 3600}
        cal = both.intersection(sorted(calibration))
        for m in ("ece", "brier", "gap"):
            row[f"{m}_base"], row[f"{m}_variant"] = b.loc[cal, m].mean(), v.loc[cal, m].mean()
            row[f"{m}_p"] = wilcoxon(v.loc[cal, m].astype(float) - b.loc[cal, m].astype(float)).pvalue
        rows.append(row)
        diffs[label] = diff
    adjusted = holm({r["comparison"]: r["p_raw"] for r in rows})
    for r in rows:
        r["p_holm"] = adjusted[r["comparison"]]
    return pd.DataFrame(rows), diffs
