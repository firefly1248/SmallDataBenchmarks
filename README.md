SmallDatasetBenchmarks
======================
Testing machine learning classifiers on small tabular datasets. The original blog post is here: https://www.data-cowboys.com/blog/which-models-are-best-for-small-datasets

## Setup

```bash
uv sync
```

## Experiments

Results are produced in `figures.ipynb` (all models including AutoML) and `figures_no_automl.ipynb` (non-AutoML models only). Each benchmark uses nested cross-validation (4-fold outer × 4-fold inner) with stratified random splits and fixed seeds. The evaluation metric is **PR AUC** (weighted average precision, OvR), which is less sensitive to class imbalance than ROC AUC. Differences between models are tested with Friedman followed by pairwise Wilcoxon signed-rank with Holm correction, one dataset per observation. Calibration is reported separately as Brier score and top-label ECE, since PR AUC scores ranking only.

| Script | Description |
|---|---|
| `compare_baseline_models.py` | SVC, Logistic Regression, Random Forest — tuned with `GridSearchCV` |
| `optuna_models.py` | SVC, LogReg, TabPFN 2.6, TabPFN-3, TabICL (GridSearch); TabFM (zero-shot); RF, XGBoost, SGD, LightGBM, LightGBM-linear, CatBoost, HistGradientBoosting, ResNet, TabNet (Optuna TPE, 50 trials per outer fold) |
| `benchmark_autogluon.py` | AutoGluon with a 300s wall-clock budget per fold (`best_quality` preset, 8 CPUs) |
| `benchmark_mljar.py` | MLJAR Supervised with a 300s wall-clock budget per fold (`Compete` mode, `n_jobs=8`) |

The AutoML figures come from the **300s-per-fold** runs (`results/*_sec_300.joblib`, ~20 min per dataset), the budget both frameworks share. A 1000s MLJAR run also exists but the matching AutoGluon run was abandoned after 11 datasets, so plotting it would compare the two at different budgets.

FT-Transformer is **not in this iteration**. It and TabNet were previously reported on numbers produced by runs in which they were largely failing to train; TabNet is now measured properly. See [Findings_notes.md](Findings_notes.md#the-bug-that-produced-two-published-results) and [FT_transformer_notes.md](FT_transformer_notes.md).

To reproduce all results sequentially:

```bash
uv run python run_all.py
# AutoGluon must use the venv Python directly (Ray incompatibility with uv run),
# and stdout must be unbuffered to see progress in log files:
PYTHONUNBUFFERED=1 .venv/bin/python -u benchmark_autogluon.py
```

Runs skip the 15 UCI++ duplicates listed in `config.DUPLICATE_DATASETS`, which every
figure drops anyway. After a run, `uv run python -m scripts.check_prevalence_baseline`
lists any (model, dataset) pair scoring no better than predicting the class prior —
the failure that once produced two published results.

### Categorical features

`compare_baseline_models.py` uses one-hot encoding. `optuna_models.py` handles categories properly:
- **CatBoost** — native `cat_features` support
- **TabPFN, TabFM** — native categorical indices
- **TabICL** — auto-detects categorical columns from pandas dtype
- **RF, XGBoost, LightGBM, HistGradientBoosting** — ordinal encoding via `category_encoders` (NaN handled natively)
- **ResNet, TabNet** — ordinal-encode + impute inside the wrapper (ResNet also standardises)
- **SVC, LogReg, SGD** — encoding strategy is a search hyperparameter (ordinal, target, James–Stein, m-estimate, CatBoost encoder)

AutoGluon and MLJAR handle categorical features internally.

## Headline

There are three tiers, and they are separated by cost as much as by performance.

**AutoML and the current foundation models are one tier.** On the 106 datasets the figures use, MLJAR leads TabFM by 0.0295 mean PR AUC but wins only 61 of them. That lead is nominally significant (Wilcoxon signed-rank, p = 0.015) and does not survive correction for the 136 pairwise comparisons in the family (Holm p = 0.43) — the honest reading is that this benchmark does not separate them. Both separate cleanly from the best classical model: MLJAR over CatBoost p = 0.001, TabFM over CatBoost p = 4e-10 on 89 of 106 datasets. What sets AutoML apart here is the budget it is handed — twenty minutes a dataset — not a different class of result.

**Foundation models are the cheapest way into that tier, on the right hardware.** TabFM reaches 0.8653 over its 126 datasets in 4.3 hours, but it ran on MPS while everything else here ran on CPU. At the 17-36x CPU/MPS ratio measured on this machine that is 73-155 CPU-hours against XGBoost's 7.4, so its time column is not comparable to the rest of the table. TabICL is the CPU answer: 0.8574 over 142 datasets in 34.7 hours. The three current-generation models span 0.0043, and the test still separates TabFM from TabPFN-3 (p = 0.004, 73 wins of 106) — the gap is small, not absent. See [FoundationModels_notes.md](FoundationModels_notes.md).

**The classical models are nearly interchangeable, with one real ordering inside.** CatBoost 0.8386, LightGBM-linear 0.8374, Random Forest 0.8359, LightGBM 0.8345, XGBoost 0.8328, HistGradientBoosting 0.8303 — the whole block spans 0.009, which is less than the run-to-run seed variance measured on a single neural model. The test still puts CatBoost above HistGradientBoosting (p = 0.0003) and above Random Forest (p = 0.04, a 0.0021 gap won on 70 of 106 datasets), while leaving it tied with LightGBM-linear. Random Forest gets 0.8359 for 7.6 hours; CatBoost gets +0.003 more for 75.7.

Coverage differs by model and the means above are each over a model's own datasets. Only 124 of 146 datasets are scored by every model. Every figure below uses complete cases only and states its own `n`, which differs because the populations differ: the rank distribution is drawn without the two AutoML models and with both the tuned and the untuned entries of SVC, LogReg and Random Forest (108 datasets, 18 entries), while the critical-difference diagram includes AutoML and drops the untuned duplicates (106 datasets, 17). Reconciling the two is [open work](#known-gaps).

## Results

### Model performance relative to Random Forest baseline

![Model performance vs RF baseline](figures/model_performance_vs_rf.png)

### Time vs performance

Wall-clock training time per dataset vs mean PR AUC gain over RF.

![Time vs performance](figures/time_performance_tradeoff.png)

### Rank distribution across datasets

How often each model achieves each rank (1 = best on a given dataset).

![Rank distribution](figures/rank_distribution.png)

### Which differences the data supports

Average rank over the datasets every model scores, with a bar over each group the
paired test cannot separate. A bar requires every pair inside it to be inseparable,
so MLJAR and TabFM carry no common bar even though the test does not separate them:
TabPFN-3 sits between them in rank and does separate from TabFM.

![Critical difference diagram](figures/critical_difference.png)

### Calibration

PR AUC scores ranking only, so a model can order every case correctly and still be
systematically overconfident. Brier and ECE come from the same stored out-of-fold
predictions; AutoGluon and MLJAR keep none and are absent here.

![Calibration vs performance](figures/calibration.png)

## Observations

- **Cost does not track performance.** The three most expensive models — TabNet (485.3 h), TabPFN 2.6 (236.3 h) and ResNet (207.7 h) — rank last, eleventh and twelfth of fifteen, all below Random Forest at 7.6 h. Full ladder in [Findings_notes.md](Findings_notes.md#cost-does-not-track-performance).
- **Foundation models have no shared blind spot.** They match or beat the best of ten classical models on 116 of 146 datasets (79.5%), and only 2 datasets have any classical model ahead by more than 0.02. An earlier version of this README claimed a blind spot on small imbalanced medical data; that was a scoring bug, described in [Findings_notes.md](Findings_notes.md#label-ordering-silently-changed-the-metric).
- **Ensembling never helped.** Averaging stored predictions — probability, logit and rank — across every combination tried failed to beat the best single model. The strongest, a logit average of the three foundation models, ties it to within 0.0001 and wins on 39% of datasets. Adding CatBoost to that trio makes it worse.
- **Where foundation models win big is synthetic structured noise**, not small data generally: on `hill-valley-with-noise` CatBoost scores 0.5560 against TabICL's 0.9967.
- **TabPFN-3 is the coverage answer.** It is the only foundation model that scores all 146 datasets, and on the 20 that at least one other foundation model refuses it beats the best classical model on 16.
- **The best-ranking model is not the best-calibrated one.** The three current foundation models take the three lowest ECE values (0.0367-0.0443); CatBoost, the strongest classical model on PR AUC, is twelfth of fourteen at 0.0848. Anything that consumes the probability rather than the ordering — a threshold, a cost model, a downstream expected value — gets a different answer from these two rankings.
- **Trained-from-scratch neural networks lose.** ResNet spends 207.7 h to land below Random Forest, and TabNet 485.3 h to finish last. The line is not "neural loses" — TabICL is a neural model and is both cheap and strong — but between *trained from scratch on your 1500 rows* and *pretrained, used in context*.
- Non-linear models outperform linear ones even on datasets with fewer than 100 samples.
- Proper categorical feature handling gives a meaningful boost on datasets with string features (~30% of the benchmark).

Method defects found and fixed during this iteration, including two that had produced published numbers, are written up in [Findings_notes.md](Findings_notes.md).

### Known gaps

- The figures are rendered by two notebooks over different model sets, so `rank_distribution.png` (no AutoML, 18 entries) and `critical_difference.png` (AutoML, 17) disagree about the denominator of a rank. One loader owning the dataset filters and the model set would remove the mismatch.
- The 0.99 "everything already solves it" filter is still hardcoded in both notebooks. The duplicate list moved to `config.DUPLICATE_DATASETS` and the runners now skip it, but the ~10 % of compute already spent on those datasets is spent.
- AutoGluon and MLJAR store no predictions, so they are absent from the calibration figure and cannot be ensembled or re-scored without a re-run.

### Note on AutoGluon operational complexity

Running AutoGluon reliably in a long CPU benchmark required several non-obvious workarounds:

- **`dynamic_stacking=False` is required.** With the default `best_quality` preset, AutoGluon's stacking phase can consume more time during initialization than the `time_limit` budget allows, causing an `AssertionError` before any model is trained.
- **Neural network models (`NeuralNetFastAI`, `NeuralNetTorch`) must be excluded on CPU.** These do not reliably respect `time_limit` on CPU hardware and hang indefinitely — sometimes 10+ hours — without output or checkpoint updates.
- **Stdout must be unbuffered.** Launch with `python -u` or `PYTHONUNBUFFERED=1`, or background-process output is suppressed entirely.
- **Ray subprocess lifecycle.** AutoGluon spawns Ray workers that outlive crashes and must be cleaned up manually before restarting.

MLJAR is significantly more robust out of the box, and scores marginally higher here.

## Data

A subset of UCI++: "a huge collection of preprocessed datasets for supervised classification problems in ARFF format"
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)

146 datasets, up to 10 000 rows each (larger datasets are subsampled). UCI++ reuses the same data in different configurations; 15 such duplicates are excluded from the figures but still computed — see [Findings_notes.md](Findings_notes.md#fifteen-datasets-are-duplicates).
