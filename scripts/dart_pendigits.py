"""Why `lgbm_dart` collapsed on the fourth outer fold of `pendigits`: the stored final model
predicts one class for most rows and scores at chance. With that fold's chosen parameters:
the final fit again, the four inner folds the search scored them on, other model seeds
(which also move LightGBM's `drop_seed`), and other `drop_seed` values alone.

Usage: uv run python -m scripts.dart_pendigits
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark.checkpoints import ckpt_path
from benchmark.data import load_data_df
from benchmark.metrics import pr_auc_score
from benchmark.models.registry import MODELS
from config import N_INNER_FOLDS, N_OUTER_FOLDS, RANDOM_STATE

OUTPUT = "results/dart_pendigits.csv"
DATASET, FOLD = "pendigits", 3

X, y, cat_cols = load_data_df(DATASET)
n_classes = len(np.unique(y))
train, test = list(StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))[FOLD]
X_train, y_train = X.iloc[train].reset_index(drop=True), y[train]
X_test, y_test = X.iloc[test].reset_index(drop=True), y[test]
entry = joblib.load(ckpt_path("lgbm_dart"))[DATASET]
best = entry["best_params"][FOLD]
print(f"{DATASET} fold {FOLD + 1}: {best}")


def fit(X_fit, y_fit, **lgbm):
    model = MODELS["lgbm_dart"].build(best, n_classes, cat_cols)
    model.named_steps["lgbm"].set_params(**lgbm)
    return model.fit(X_fit, y_fit)


def row(case, model, X_eval, y_eval, **extra):
    prob = model.predict_proba(X_eval)
    top = np.bincount(prob.argmax(axis=1), minlength=n_classes).max() / len(prob)
    return dict(case=case, **extra, prauc=pr_auc_score(y_eval, prob), top_class_share=top)


rows = []
final = fit(X_train, y_train)
rows.append(row("final, test", final, X_test, y_test))
rows.append(row("final, own training rows", final, X_train, y_train))
print("refit equals the stored predictions:", np.allclose(final.predict_proba(X_test), entry["preds"][FOLD], atol=1e-12))

inner = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for k, (fit_idx, val_idx) in enumerate(inner.split(X_train, y_train)):
    model = fit(X_train.iloc[fit_idx], y_train[fit_idx])
    rows.append(row("inner fold", model, X_train.iloc[val_idx], y_train[val_idx], inner_fold=k + 1))

for seed in (1, 2, 3):
    rows.append(row("final, model seed", fit(X_train, y_train, random_state=seed), X_test, y_test, seed=seed))
for drop_seed in (1, 2, 3):
    rows.append(row("final, drop_seed", fit(X_train, y_train, drop_seed=drop_seed), X_test, y_test,
                    drop_seed=drop_seed))

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)
print(out.round(4).to_string(index=False))
print(f"Saved {OUTPUT}")
