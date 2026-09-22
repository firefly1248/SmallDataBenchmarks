"""Flag (model, dataset) pairs that score no better than predicting the prior.

A default batch size once left TabNet taking zero gradient steps on 82 datasets.
It still produced a full column of scores, and those scores read as ordinary
because weighted PR AUC starts at the class prevalence, not at 0.5: 0.4698 looks
like a weak result and was in fact the floor. Comparing every stored score with
its own floor is what would have caught it.

This prints a report rather than failing: some datasets genuinely defeat every
model, so sitting on the baseline is evidence to look at, not proof of a bug.

Usage: uv run python -m scripts.check_prevalence_baseline [margin]
"""
import sys

import joblib
import numpy as np

from benchmark.checkpoints import available_models, ckpt_path
from benchmark.metrics import pr_auc_baseline

MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01

suspects = []
for model in available_models():
    for dataset, entry in joblib.load(ckpt_path(model)).items():
        if entry.get("preds") is None:
            continue
        score = float(np.mean(entry["scores"]))
        labels = np.concatenate(entry["labels"])
        baseline = pr_auc_baseline(labels, entry["preds"][0].shape[1])
        if score - baseline < MARGIN:
            suspects.append((score - baseline, model, dataset, score, baseline))

print(f"{len(suspects)} (model, dataset) pairs within {MARGIN} of their baseline\n")
print(f"{'model':16s} {'dataset':34s} {'score':>8s} {'baseline':>9s} {'margin':>8s}")
for margin, model, dataset, score, baseline in sorted(suspects):
    print(f"{model:16s} {dataset:34s} {score:8.4f} {baseline:9.4f} {margin:+8.4f}")

by_model = {}
for _, model, *_ in suspects:
    by_model[model] = by_model.get(model, 0) + 1
print("\nper model:")
for model, count in sorted(by_model.items(), key=lambda kv: -kv[1]):
    print(f"  {model:16s} {count}")
