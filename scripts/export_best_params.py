"""Export each outer fold's chosen hyperparameters to a long table.

One row per (model, dataset, fold, param). The checkpoints themselves are not
tracked (~350 MB of predictions), so this is the published record of what the
tuning picked. A backup checkpoint passed as ``label=path`` is exported under
that label, e.g. the CatBoost run that started from uniform probabilities:

Usage: uv run python -m scripts.export_best_params \
           catboost_uniform_start=results/ckpt/catboost.joblib.bak-uniform-start
"""
import sys

import joblib
import pandas as pd

from benchmark.checkpoints import available_models, ckpt_path

OUTPUT = "results/best_params.csv"

sources = [(m, ckpt_path(m)) for m in available_models()]
sources += [tuple(arg.split("=", 1)) for arg in sys.argv[1:]]
rows = []
for model, path in sources:
    for dataset, entry in joblib.load(path).items():
        for fold, params in enumerate(entry.get("best_params") or []):
            rows += [dict(model=model, dataset=dataset, fold=fold, param=k, value=v)
                     for k, v in params.items()]
pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}: {len(rows)} rows from {len(sources)} checkpoints")
