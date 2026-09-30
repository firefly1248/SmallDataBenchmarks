SmallDatasetBenchmarks
======================
Testing machine learning classifiers on small tabular datasets. The original blog post is here: https://www.data-cowboys.com/blog/which-models-are-best-for-small-datasets

## Setup

```bash
uv sync
```

## Experiments

Results are produced in `figures.ipynb` (all models including AutoML) and `figures_no_automl.ipynb` (non-AutoML models only). Each benchmark uses nested cross-validation (4-fold outer × 4-fold inner) with stratified random splits and fixed seeds. The evaluation metric is **PR AUC** (weighted average precision, OvR), which, unlike ROC AUC, is not inflated by a large majority class. Differences between models are tested with Friedman followed by pairwise Wilcoxon signed-rank with Holm correction, one dataset per observation. Calibration is reported separately as Brier score and top-label ECE, since PR AUC scores ranking only.

| Script | Description |
|---|---|
| `compare_baseline_models.py` | SVC, Logistic Regression, Random Forest — tuned with `GridSearchCV` |
| `optuna_models.py` | SVC, LogReg, TabPFN-3, TabPFN-3.5, TabPFN-3.5-fast, TabICL (GridSearch); TabFM (zero-shot); RF, XGBoost, SGD, LightGBM, LightGBM-linear, CatBoost, HistGradientBoosting, ResNet, TabNet (Optuna TPE, 50 trials per outer fold) |
| `benchmark_autogluon.py` | AutoGluon 1.6.3 with a 300s wall-clock budget per fold (`best_quality` preset, 16 CPUs) |
| `benchmark_mljar.py` | MLJAR Supervised 1.3.2 with a 300s wall-clock budget per fold (`Compete` mode, `n_jobs=16`) |

Foundation-model weights carry their own licences, separate from the code:

| Model | Code | Weights |
|---|---|---|
| TabPFN-3, TabPFN-3.5, TabPFN-3.5-fast | Apache-2.0 | Prior Labs licence: non-commercial and non-production use only, outputs included |
| TabFM | Apache-2.0 | `tabfm-non-commercial-v1.0` |
| TabICL | BSD-3-Clause | BSD-3-Clause |

`results/datasets.csv` gives each dataset's size, features, classes and minority share; `results/best_params.csv` every outer fold's chosen hyperparameters.

The AutoML figures come from the **300s-per-fold** runs (`results/*_sec_300.joblib`, ~20 min per dataset), the budget both frameworks share. They were re-run in September 2026 with probabilities stored: the 300s numbers published before then were the upstream project's 2020 results, scored by ROC AUC rather than PR AUC, and ranked both frameworks first — see [Findings_notes.md](Findings_notes.md#the-published-automl-numbers-were-roc-auc). A 1000s MLJAR run also exists (PR AUC, correctly scored) but the matching AutoGluon run was abandoned after 11 datasets, so plotting it would compare the two at different budgets.

FT-Transformer is **not in this iteration**. It and TabNet were previously reported on numbers produced by runs in which they were largely failing to train; TabNet is now measured properly. See [Findings_notes.md](Findings_notes.md#the-bug-that-produced-two-published-results) and [FT_transformer_notes.md](FT_transformer_notes.md).

To reproduce all results sequentially:

```bash
uv run python run_all.py
# AutoGluon must use the venv Python directly (Ray incompatibility with uv run),
# and stdout must be unbuffered to see progress in log files:
PYTHONUNBUFFERED=1 .venv/bin/python -u benchmark_autogluon.py
# LightAutoML pins xgboost<3 and statsmodels<=0.14.0, so it has its own venv.
# torch 2.14 with lightgbm 4.7 deadlocks there on macOS; the file pins the pair
# the main env uses.
uv venv .venv-lama --python 3.11
uv pip install --python .venv-lama/bin/python -r requirements-lightautoml.txt
PYTHONUNBUFFERED=1 .venv-lama/bin/python -u benchmark_lightautoml.py
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

The ranking has a clear top, and cost does not follow it.

**The foundation models are the top tier on their own, and AutoML is not in it.** On the 108 datasets the figures use, the weakest foundation model here, TabPFN-3, beats AutoGluon on 81 (+0.0104, Holm p = 6e-8), and all ten foundation-versus-AutoML pairs separate. TabFM over CatBoost is +0.0254 on 91 of 108 (p = 2e-10). An earlier version of this README put AutoML first; that rested on AutoML scored by ROC AUC against everything else by PR AUC.

**Below them, AutoGluon heads the rest and MLJAR is one of them.** Over the same 131 datasets AutoGluon scores 0.8427 in 43.1 hours and MLJAR 0.8315 in 37.4, against CatBoost's 0.8348 in 73.7. AutoGluon separates from every classical model except CatBoost, whose gap (+0.0111, 71 wins of 108) clears the raw test but not the correction for 153 comparisons (Holm p = 0.08). MLJAR does not: on those 108 datasets it is level with CatBoost to four decimals and inside the gradient-booster group, and AutoGluon separates from it (+0.0111, 72 wins, Holm p = 0.016).

**A new generation of foundation model arrived mid-benchmark and it is both better and cheaper.** Over the same 131 datasets, TabPFN-3.5 scores 0.8627 against TabPFN-3's 0.8571 in **17.8 hours against 63.6** — a 3.6x cost cut that the test confirms as a real gain (p = 0.007, 72 wins of 108). The `v3.5-fast` variant halves the cost again to 9.3 hours for 0.8608, and the test cannot separate it from full 3.5 (p = 0.08). Within one model family the price moved by a factor of nearly seven while performance moved in the third decimal. See [FoundationModels_notes.md](FoundationModels_notes.md).

**Foundation models are the cheapest way into the top tier, on the right hardware.** TabFM reaches 0.8653 over its 126 datasets in 4.3 hours, but it ran on MPS while everything else here ran on CPU. At the 17-36x CPU/MPS ratio measured on this machine that is 73-155 CPU-hours against XGBoost's 7.4, so its time column is not comparable to the rest of the table. On CPU the answer is now TabPFN-3.5-fast at 9.3 hours, ahead of TabICL's 29.8 over the same 131 datasets — and ahead of both AutoML frameworks at a quarter of their compute. The four current-generation models span 0.0079, and the test still separates TabFM from TabPFN-3 (p = 0.006, 74 wins of 108) — the gaps are small, not absent.

**The classical models are nearly interchangeable.** CatBoost 0.8386, LightGBM-linear 0.8374, Random Forest 0.8359, LightGBM 0.8345, XGBoost 0.8328, HistGradientBoosting 0.8303 — the whole block spans 0.009, which is less than the run-to-run seed variance measured on a single neural model. The test puts CatBoost above HistGradientBoosting (p = 0.0001) and ties it with LightGBM-linear and Random Forest; the 0.0017 gap over Random Forest, won on 70 of 108, clears the raw test (p = 0.002) but not the correction (Holm p = 0.08). Random Forest gets 0.8359 for 7.6 hours; CatBoost gets +0.003 more for 75.7.

Coverage differs by model, and the means and hours above are each over a model's own datasets — 146 for the models that predate the duplicate skip, 131 for the two TabPFN-3.5 variants, fewer where a foundation model refuses a dataset. Where two models are compared on cost, both numbers are over the same 131. Of the 131 datasets a run now covers, 109 are scored by every model; the rest are refused by one foundation model or another on feature or class limits. Every figure below uses complete cases only and states its own `n`. The rank distribution is drawn without the two AutoML models and with both the tuned and the untuned entries of SVC, LogReg and Random Forest, while the critical-difference diagram includes AutoML and drops the untuned duplicates (108 datasets, 18 entries). Reconciling the two is [open work](#known-gaps).

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
paired test cannot separate. The top bar spans TabFM, TabPFN-3.5, TabICL and
TabPFN-3.5-fast. A bar requires every pair inside it to be inseparable, so TabPFN-3
falls outside it while sharing the second one: it separates from TabFM and full
TabPFN-3.5. No bar joins a foundation model to anything else. AutoGluon shares one
only with CatBoost, and MLJAR sits inside the gradient-booster group.

![Critical difference diagram](figures/critical_difference.png)

### Calibration

PR AUC scores ranking only, so a model can order every case correctly and still be
systematically overconfident. Brier and ECE come from the same stored out-of-fold
predictions, both AutoML frameworks included.

![Calibration vs performance](figures/calibration.png)

## Observations

- **Cost does not track performance.** The two most expensive models — TabNet (485.3 h) and ResNet (207.7 h) — finish last and fourth from last, both below Random Forest at 7.6 h. CatBoost spends 75.7 h to beat Random Forest by 0.003. Full ladder in [Findings_notes.md](Findings_notes.md#cost-does-not-track-performance).
- **Foundation models have no shared blind spot.** They match or beat the best of eleven classical models on 109 of 131 datasets (83.2%), and only 3 datasets have any classical model ahead by more than 0.02. An earlier version of this README claimed a blind spot on small imbalanced medical data; that was a scoring bug, described in [Findings_notes.md](Findings_notes.md#label-ordering-silently-changed-the-metric).
- **Ensembling never helped.** Averaging stored predictions — probability, logit and rank — over all 741 blends of the five foundation models and three boosters fails to beat the best single model. The strongest, a logit average of TabFM and TabPFN-3.5, gains 0.0015 and wins on half the datasets; the best of 741 tries at p = 0.04, it is nowhere near significant once corrected for them. No booster improves a foundation blend. CatBoost joins the analysis after its re-run.
- **Where foundation models win big is synthetic structured noise**, not small data generally: on `hill-valley-with-noise` CatBoost scores 0.5560 against TabICL's 0.9967.
- **The TabPFN line is the coverage answer.** All three of its versions score every dataset they are given — 131 of 131 — while TabICL refuses 4 and TabFM 20 on feature or class limits. On the 20 datasets TabICL or TabFM refuses, TabPFN beats the best of eleven classical models on 18.
- **PR AUC rank does not predict calibration.** TabFM has the lowest ECE (0.0367); AutoGluon (0.0391) and TabICL (0.0405) are next, in an order that depends on how ECE is binned; MLJAR (0.0456) is ahead of every classical model. CatBoost, among the strongest classical models on PR AUC, is near the bottom at 0.0848, level with TabNet and ahead only of SGD — mostly a tail of undertrained fits, made worse on imbalanced data by starting from uniform probabilities rather than the class prior; its median is the best of the boosters, and a re-run from the prior is in progress ([details](Findings_notes.md#catboost-started-from-uniform-probabilities)). Anything that consumes the probability rather than the ordering — a threshold, a cost model, a downstream expected value — gets a different answer from these two rankings.
- **Trained-from-scratch neural networks lose.** ResNet spends 207.7 h to land below Random Forest, and TabNet 485.3 h to finish last. The line is not "neural loses" — TabICL is a neural model and is both cheap and strong — but between *trained from scratch on your 1500 rows* and *pretrained, used in context*.
- Non-linear models outperform linear ones even on datasets with fewer than 100 samples.
- Proper categorical feature handling gives a meaningful boost on datasets with string features (~30% of the benchmark).

Method defects found and fixed during this iteration, including two that had produced published numbers, are written up in [Findings_notes.md](Findings_notes.md).

### Known gaps

- The figures are rendered by two notebooks over different model sets, so `rank_distribution.png` (no AutoML, 19 entries) and `critical_difference.png` (AutoML, 18 entries) rank the same 108 datasets out of different fields. The dataset filters, the threshold and the model labels are now shared constants, but the load-and-reshape block is still copied; one loader owning it would remove the mismatch.
- The ~10 % of compute already spent on the duplicate datasets is spent. The runners skip them now, which only helps a future run.

### Note on AutoGluon operational complexity

Running AutoGluon reliably in a long CPU benchmark required several non-obvious workarounds:

- **`dynamic_stacking=False` is required.** With the default `best_quality` preset, AutoGluon's stacking phase can consume more time during initialization than the `time_limit` budget allows, causing an `AssertionError` before any model is trained.
- **The neural-network exclusion does not take effect, and the runs were measured with it not taking effect.** `excluded_model_types=["NeuralNetFastAI", "NeuralNetTorch"]` passes class names; AutoGluon matches registry keys (`FASTAI`, `NN_TORCH`) and silently ignores anything else. `NeuralNetTorch` trains, and on `blood-transfusion-service` it takes the top six leaderboard places. `NeuralNetFastAI` is absent only because `fastai` is not installed. The published run was measured with the argument in place, and it is kept so that run stays reproducible.
- **Stdout must be unbuffered.** Launch with `python -u` or `PYTHONUNBUFFERED=1`, or background-process output is suppressed entirely.
- **Ray subprocess lifecycle.** AutoGluon spawns Ray workers that outlive crashes and must be cleaned up manually before restarting.

MLJAR is much more robust out of the box; AutoGluon scores higher (+0.0111, 72 wins of 108, Holm p = 0.016).

## Data

A subset of UCI++: "a huge collection of preprocessed datasets for supervised classification problems in ARFF format"
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)

146 datasets, up to 10 000 rows each (larger datasets are subsampled). UCI++ reuses the same data in different configurations; 15 such duplicates are excluded from the figures, and runs now skip them, so a fresh run covers 131 — see [Findings_notes.md](Findings_notes.md#fifteen-datasets-are-duplicates).
