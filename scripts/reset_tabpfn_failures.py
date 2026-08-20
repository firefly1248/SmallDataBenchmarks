"""Remove all-NaN TabPFN entries from the checkpoint so they get re-run."""
import os

import joblib
import numpy as np

from benchmark.checkpoints import ckpt_path

MODEL = "tabpfn"

entries = joblib.load(ckpt_path(MODEL))
reset = [ds for ds, v in entries.items()
         if v["scores"] and all(np.isnan(s) for s in v["scores"])]
for ds in reset:
    del entries[ds]

tmp = ckpt_path(MODEL) + ".tmp"
joblib.dump(entries, tmp)
os.replace(tmp, ckpt_path(MODEL))

print(f"Reset {len(reset)} datasets:")
for ds in reset:
    print(f"  {ds}")
print(f"Now re-run: uv run python -u optuna_models.py --models {MODEL}")
