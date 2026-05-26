"""Remove all-NaN TabPFN entries from checkpoint so they get re-run."""
import joblib, numpy as np

CHECKPOINT = "results/optuna_models_ckpt.joblib"

ckpt, done_set = joblib.load(CHECKPOINT)

reset = []
for ds, models in ckpt.items():
    if "tabpfn" in models:
        scores = models["tabpfn"]["scores"]
        if scores and all(np.isnan(s) for s in scores):
            reset.append(ds)

for ds in reset:
    done_set.discard((ds, "tabpfn"))
    del ckpt[ds]["tabpfn"]

joblib.dump((ckpt, done_set), CHECKPOINT)
print(f"Reset {len(reset)} datasets:")
for ds in reset:
    print(f"  {ds}")
print("Now re-run: uv run python -u optuna_models.py --models tabpfn")
