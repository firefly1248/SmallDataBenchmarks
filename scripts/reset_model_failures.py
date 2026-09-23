"""Remove all-NaN entries from a model's checkpoint so they get re-run.

Usage: uv run python -m scripts.reset_model_failures <model>
"""
import sys

import joblib
import numpy as np

from benchmark.checkpoints import atomic_dump, ckpt_path

MODEL = sys.argv[1]

entries = joblib.load(ckpt_path(MODEL))
reset = [ds for ds, v in entries.items()
         if v["scores"] and all(np.isnan(s) for s in v["scores"])]
for ds in reset:
    del entries[ds]

atomic_dump(entries, ckpt_path(MODEL))

print(f"Reset {len(reset)} datasets:")
for ds in reset:
    print(f"  {ds}")
print(f"Now re-run: uv run python -u optuna_models.py --models {MODEL}")
