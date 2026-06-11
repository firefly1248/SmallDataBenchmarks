SmallDatasetBenchmarks
======================
Testing machine learning classifiers on small tabular datasets. The original blog post is here: https://www.data-cowboys.com/blog/which-models-are-best-for-small-datasets

## Setup

```bash
uv sync
```

## Experiments

Results are produced in `figures.ipynb` (all models including AutoML) and `figures_no_automl.ipynb` (non-AutoML models only). A focused three-way comparison of the strongest models lives in `figures_tabicl_vs_tabpfn_vs_catboost.ipynb`. Each benchmark uses nested cross-validation (4-fold outer × 4-fold inner) with stratified random splits and fixed seeds. The evaluation metric is **PR AUC** (weighted average precision, OvR), which is less sensitive to class imbalance than ROC AUC.

| Script | Description |
|---|---|
| `compare_baseline_models.py` | SVC, Logistic Regression, Random Forest — tuned with `GridSearchCV` |
| `optuna_models.py` | SVC, LogReg, TabPFN, TabICL (GridSearch); RF, XGBoost, SGD, LightGBM, LightGBM-linear, CatBoost, HistGradientBoosting, TabNet, ResNet, FT-Transformer (Optuna TPE, 50 trials per outer fold). Notes: TabICL skips datasets with ≥500 features (4 datasets, recorded as NaN) — see [TabICL_notes.md](TabICL_notes.md). FT-Transformer was run on a 67/146 subset only — see [FT_transformer_notes.md](FT_transformer_notes.md). |
| `benchmark_autogluon.py` | AutoGluon with a 1000s wall-clock time budget per fold (`best_quality` preset, 8 CPUs) |
| `benchmark_mljar.py` | MLJAR Supervised with a 1000s wall-clock time budget per fold (`Compete` mode, `n_jobs=8`) |

To reproduce all results sequentially:

```bash
uv run python run_all.py
# AutoGluon must use the venv Python directly (Ray incompatibility with uv run),
# and stdout must be unbuffered to see progress in log files:
PYTHONUNBUFFERED=1 .venv/bin/python -u benchmark_autogluon.py
```

### Categorical features

`compare_baseline_models.py` uses one-hot encoding. `optuna_models.py` handles categories properly:
- **CatBoost** — native `cat_features` support
- **TabPFN** — native categorical indices
- **TabICL** — auto-detects categorical columns from pandas dtype (internal OrdinalEncoder + imputer)
- **RF, XGBoost, LightGBM, HistGradientBoosting** — ordinal encoding via `category_encoders` (NaN handled natively)
- **TabNet, FT-Transformer, ResNet** — ordinal-encode + impute + StandardScaler inside the wrapper
- **SVC, LogReg, SGD** — encoding strategy is an Optuna hyperparameter (ordinal, target, James–Stein, m-estimate, CatBoost encoder)

AutoGluon and MLJAR handle categorical features internally.

### Headline: TabICL is the new default

[TabICL](https://github.com/soda-inria/tabicl) (in-context tabular foundation model, INRIA) is the strongest model in this benchmark and **also fast enough for everyday use** — a rare combination among tabular foundation models. On 107 non-trivial datasets where all three competitive models have valid scores:

- TabICL beats tuned CatBoost on **76%** of datasets (mean ΔPR-AUC **+0.033**, median +0.007).
- TabICL beats TabPFN on **72%** of datasets (mean Δ +0.050) — and is **4× faster** than TabPFN at median train time.
- Mean rank 1.51 vs ~2.24 for both TabPFN and CatBoost; TabICL is rank-1 on 62/107 datasets, the others ~22 each.
- Median train time only **1.3× slower than CatBoost** (190s vs 143s). TabPFN's median is 724s.

TabICL has a feature-count limit: it stalls (multi-hour, ballooning RSS) on datasets with ≥500 features and is skipped for those (4 datasets total). See [TabICL_notes.md](TabICL_notes.md) for the full analysis including shared blind spots with TabPFN.

### Note on FT-Transformer

FT-Transformer was run on 67 of 146 datasets before being stopped. The data is sufficient to conclude that it is **both impractical on CPU and not competitive on accuracy** for this kind of small tabular data: ~150× more compute than tuned gradient boosters for a mean PR-AUC delta of **−0.026** vs. the best GB per dataset. See [FT_transformer_notes.md](FT_transformer_notes.md) for the full breakdown. Excluded from the headline results below.

## Results

### Model performance relative to Random Forest baseline

Mean PR AUC delta across 146 datasets (positive = better than RF).

![Model performance vs RF baseline](figures/model_performance_vs_rf.png)

### Time–accuracy tradeoff

Wall-clock training time per dataset vs mean PR AUC gain over RF.

![Time-accuracy tradeoff](figures/time_accuracy_tradeoff.png)

### Rank distribution across datasets

How often each model achieves each rank (1 = best on a given dataset).

![Rank distribution](figures/rank_distribution.png)

## Observations

- **TabICL is the strongest model overall** and the only foundation model with a favourable accuracy/cost trade-off on CPU. See the headline above and [TabICL_notes.md](TabICL_notes.md).
- For severely class-imbalanced small datasets (e.g. `blood-transfusion-service`, `appendicitis`, `pima-indians-diabetes`), tuned **CatBoost** still wins decisively — TabICL and TabPFN share the same blind spots there. Co-training both is cheap and robust.
- **TabPFN is strictly dominated** by TabICL on this benchmark: TabICL is more accurate (78/108 head-to-head wins) AND ~4× faster at median train time. The 10-class cap and high-feature stalls also limit TabPFN's coverage.
- TabNet, ResNet and FT-Transformer don't match tuned GB accuracy on average and cost 1–2 orders of magnitude more compute. See [FT_transformer_notes.md](FT_transformer_notes.md).
- Non-linear models outperform linear ones even on datasets with fewer than 100 samples.
- Optuna-tuned XGBoost, CatBoost, LightGBM, and the new HistGradientBoosting are strong individual models, competitive with AutoML frameworks.
- AutoGluon and MLJAR show higher median PR AUC than individual GBs, but require a substantial wall-clock budget (1000s/fold used here).
- Proper categorical feature handling gives a meaningful boost on datasets with string features (~30% of the benchmark).
- LightGBM with linear trees (`linear_tree=True`) is a useful addition to the Optuna model set.

### Note on AutoGluon operational complexity

Running AutoGluon reliably in a long CPU benchmark required several non-obvious workarounds — worth noting as a practical consideration when choosing an AutoML framework:

- **`dynamic_stacking=False` is required.** With the default `best_quality` preset, AutoGluon's stacking phase can consume more time during initialization than the `time_limit` budget allows, causing an `AssertionError` before any model is trained.
- **Neural network models (`NeuralNetFastAI`, `NeuralNetTorch`) must be excluded on CPU.** These models do not reliably respect `time_limit` on CPU hardware, causing the process to hang indefinitely — sometimes for 10+ hours — without producing any output or checkpoint updates. Add `excluded_model_types=["NeuralNetFastAI", "NeuralNetTorch"]` to `.fit()`.
- **Stdout must be unbuffered.** When running as a background process redirected to a log file, Python's default buffering suppresses all `print()` output, making it impossible to monitor progress. Launch with `python -u` or `PYTHONUNBUFFERED=1`.
- **Ray subprocess lifecycle.** AutoGluon spawns Ray worker processes that outlive crashes. After a hang or kill, Ray child processes must be cleaned up manually before restarting.

The upshot: AutoGluon is powerful but has meaningful operational overhead on CPU-only machines. For automated or unattended runs, MLJAR is significantly more robust out of the box.

## Data

A subset of UCI++: "a huge collection of preprocessed datasets for supervised classification problems in ARFF format"
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)

146 datasets, up to 10 000 rows each (larger datasets are subsampled). Note that UCI++ reuses the same datasets in different configurations and some categorical features are not clearly labeled.
