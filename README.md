SmallDatasetBenchmarks
======================
Testing machine learning classifiers on small tabular datasets. The original blog post is here: https://www.data-cowboys.com/blog/which-models-are-best-for-small-datasets

## Setup

```bash
uv sync
```

## Experiments

Results are produced in `figures.ipynb` (all models) and `figures_no_automl.ipynb` (non-AutoML models only). Each benchmark uses nested cross-validation (4-fold outer × 4-fold inner) with stratified random splits and fixed seeds. The evaluation metric is **PR AUC** (weighted average precision, OvR), which is less sensitive to class imbalance than ROC AUC.

| Script | Description |
|---|---|
| `compare_baseline_models.py` | SVC, Logistic Regression, Random Forest — tuned with `GridSearchCV` |
| `optuna_models.py` | SVC, LogReg (GridSearch); RF, XGBoost, SGD, LightGBM, CatBoost (Optuna, 50 trials per fold) |
| `benchmark_autogluon.py` | AutoGluon with a 1000s wall-clock time budget per fold (`best_quality` preset, 8 CPUs) |
| `benchmark_mljar.py` | MLJAR Supervised with a 1000s wall-clock time budget per fold (`Compete` mode, `n_jobs=8`) |

To reproduce all results sequentially:

```bash
python compare_baseline_models.py
python optuna_models.py
python benchmark_mljar.py
python benchmark_autogluon.py   # must use .venv/bin/python, not uv run
```

### Categorical features

`compare_baseline_models.py` uses one-hot encoding. `optuna_models.py` handles categories properly:
- **CatBoost** — native `cat_features` support
- **RF, XGBoost, LightGBM** — ordinal encoding via `category_encoders`
- **SVC, LogReg, SGD** — encoding strategy is an Optuna hyperparameter (ordinal, target, James–Stein, m-estimate, CatBoost encoder)

AutoGluon and MLJAR handle categorical features internally.

## Data

A subset of UCI++: "a huge collection of preprocessed datasets for supervised classification problems in ARFF format"
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)

146 datasets, up to 10 000 rows each (larger datasets are subsampled). Note that UCI++ reuses the same datasets in different configurations and some categorical features are not clearly labeled.

## Observations

- Non-linear models outperform linear ones even on datasets with fewer than 100 samples.
- Optuna-tuned XGBoost and CatBoost are the strongest individual models, competitive with AutoML frameworks.
- AutoGluon and MLJAR show higher median PR AUC, but require a substantial wall-clock budget (1000s/fold used here).
- Proper categorical feature handling gives a meaningful boost on datasets with string features (~30% of the benchmark).
- LightGBM with linear trees (`boosting_type="rf"` variant) is a useful addition to the Optuna model set.
