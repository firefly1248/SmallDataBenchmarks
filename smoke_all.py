"""
Quick smoke test for all 4 benchmark blocks.
Uses iris (binary subset) — 2 outer folds, 2 inner folds, 3 Optuna trials, 30s AutoML budget.
"""
import warnings, sys, importlib.util, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X_np, y_np = iris.data, iris.target

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── 1. compare_baseline_models ───────────────────────────────────────────────
print("\n=== 1. compare_baseline_models ===")
try:
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import average_precision_score, make_scorer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline

    PR_AUC_SCORER = make_scorer(average_precision_score, average="weighted", response_method="predict_proba")

    def _eval(pipeline, param_grid):
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
        clf = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=PR_AUC_SCORER, n_jobs=1)
        return cross_val_score(clf, X=X_np, y=y_np, cv=outer_cv, scoring=PR_AUC_SCORER, n_jobs=1)

    svc_pipe = Pipeline([("scaler", MinMaxScaler()), ("svc", SVC(kernel="linear", probability=True, random_state=0))])
    lr_pipe  = Pipeline([("scaler", MinMaxScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=0))])
    rf       = RandomForestClassifier(random_state=0)

    s1 = _eval(svc_pipe, {"svc__C": [0.1, 1.0]})
    s2 = _eval(lr_pipe,  {"lr__C": [0.1, 1.0]})
    s3 = _eval(rf,       {"max_depth": [4, None]})

    assert len(s1) == 2 and len(s2) == 2 and len(s3) == 2
    assert all(0 <= v <= 1 for v in [*s1, *s2, *s3])
    print(f"  SVC={s1.mean():.4f}  LR={s2.mean():.4f}  RF={s3.mean():.4f}  [{PASS}]")
except Exception as e:
    print(f"  [{FAIL}] {e}")


# ── 2. optuna_models ─────────────────────────────────────────────────────────
print("\n=== 2. optuna_models ===")
try:
    import pandas as pd
    m = load_mod("optuna_models.py", "optuna_models")
    # Patch to minimal budget
    m.N_TRIALS       = 10
    m.N_OUTER_FOLDS  = 2
    m.N_INNER_FOLDS  = 2
    m.N_JOBS         = 1

    X_df = pd.DataFrame(X_np)
    cat_cols = []

    for model_name in ["logreg", "sgd", "random_forest", "xgboost", "catboost", "svc", "lgbm", "lgbm_linear"]:
        t0 = time.time()
        scores = m.run_nested_cv(X_df, y_np, model_name, cat_cols)
        elapsed = time.time() - t0
        assert len(scores) == m.N_OUTER_FOLDS
        assert all(0 <= v <= 1 for v in scores)
        print(f"  {model_name:15s}: mean={np.mean(scores):.4f}  t={elapsed:.1f}s  [{PASS}]")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  [{FAIL}] {e}")


# ── 3. benchmark_autogluon ───────────────────────────────────────────────────
print("\n=== 3. benchmark_autogluon ===")
try:
    m = load_mod("benchmark_autogluon.py", "benchmark_autogluon")
    m.SEC           = 30
    m.N_OUTER_FOLDS = 2

    t0 = time.time()
    scores = m.evaluate_autogluon(X_np, y_np)
    elapsed = time.time() - t0
    assert len(scores) == m.N_OUTER_FOLDS
    assert all(0 <= v <= 1 for v in scores)
    print(f"  mean={np.mean(scores):.4f}  t={elapsed:.1f}s  [{PASS}]")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  [{FAIL}] {e}")


# ── 4. benchmark_mljar ───────────────────────────────────────────────────────
print("\n=== 4. benchmark_mljar ===")
try:
    m = load_mod("benchmark_mljar.py", "benchmark_mljar")
    m.SEC           = 30
    m.N_OUTER_FOLDS = 2

    t0 = time.time()
    scores = m.evaluate_mljar(X_np, y_np)
    elapsed = time.time() - t0
    assert len(scores) == m.N_OUTER_FOLDS
    assert all(0 <= v <= 1 for v in scores)
    print(f"  mean={np.mean(scores):.4f}  t={elapsed:.1f}s  [{PASS}]")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  [{FAIL}] {e}")

print("\n=== Done ===")
