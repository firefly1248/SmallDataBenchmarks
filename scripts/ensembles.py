"""Average the stored out-of-fold predictions and ask whether any blend beats
the best single model.

Every combination of two or more models from the pool, three combiners:
probability mean, logit mean (softmax of the mean log-probability, which is the
mean logit on binary data) and rank mean (per-class ranks within the fold).
Scored like every model, as PR AUC averaged over the four outer folds, on the
datasets every pool model scored. A logit blend weighs each member by its
confidence, so a badly calibrated member skews it: CatBoost joined only after its
prior-start re-run replaced the uniform-start predictions.

Usage: uv run python -m scripts.ensembles [model ...]
"""
import sys
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from benchmark.checkpoints import ckpt_path
from benchmark.data import datasets_to_run
from benchmark.metrics import pr_auc_score
from benchmark.plots import MODEL_LABELS

POOL = sys.argv[1:] or ["tabfm", "tabicl", "tabpfn3", "tabpfn35", "tabpfn35fast",
                        "lgbm_linear", "xgboost", "lgbm", "catboost"]
OUTPUT = "results/ensembles.csv"


def blend(probs, combiner):
    if combiner == "probability":
        return np.mean(probs, axis=0)
    if combiner == "logit":
        log_p = np.mean([np.log(np.clip(p, 1e-7, 1)) for p in probs], axis=0)
        e = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return np.mean([rankdata(p, axis=0) for p in probs], axis=0)


ckpts = {m: joblib.load(ckpt_path(m)) for m in POOL}
datasets = [d for d in datasets_to_run() if all(ckpts[m].get(d, {}).get("preds") is not None for m in POOL)]
single = pd.DataFrame({m: [np.mean(ckpts[m][d]["scores"]) for d in datasets] for m in POOL}, index=datasets)
best = single.mean().idxmax()

rows = []
for k in range(2, len(POOL) + 1):
    for members in combinations(POOL, k):
        for combiner in ("probability", "logit", "rank"):
            scores = np.array([np.mean([pr_auc_score(y, blend([ckpts[m][d]["preds"][f] for m in members], combiner))
                                        for f, y in enumerate(ckpts[members[0]][d]["labels"])])
                               for d in datasets])
            diff = scores - single[best].to_numpy()
            rows.append(dict(members=" + ".join(MODEL_LABELS[m].split(" (")[0] for m in members),
                             size=k, combiner=combiner, mean=scores.mean(), vs_best_single=diff.mean(),
                             win_rate=(diff > 0).mean(), p=wilcoxon(diff).pvalue))
df = pd.DataFrame(rows).sort_values("vs_best_single", ascending=False)
df.round(5).to_csv(OUTPUT, index=False)

print(f"{len(datasets)} datasets; best single model {MODEL_LABELS[best]} at {single[best].mean():.4f}")
print(single.mean().sort_values(ascending=False).round(4).to_string())
print(f"\nTop 12 of {len(df)} blends:")
print(df.head(12).round(4).to_string(index=False))
print("\nBest combiner per member set:",
      df.loc[df.groupby("members").vs_best_single.idxmax()].combiner.value_counts().to_dict())
print(f"Saved {OUTPUT}")
