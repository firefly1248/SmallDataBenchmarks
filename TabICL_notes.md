# TabICL notes

Findings from running [TabICL](https://github.com/soda-inria/tabicl) (INRIA's in-context tabular foundation model, v2 checkpoint `tabicl-classifier-v2-20260212`) in the nested-CV benchmark across 146 small tabular datasets.

## TL;DR

TabICL is **the strongest model on this benchmark** and **the only tabular foundation model that's also fast enough for everyday CPU use**. It beats tuned CatBoost on 76% of non-trivial datasets at a median compute cost of only 1.3× CatBoost. It strictly dominates TabPFN (more accurate AND 4× faster at median).

## Coverage

- 142 / 146 datasets with valid scores (97%).
- 4 datasets skipped (NaN) by a hard feature-count guard, threshold n_features ≥ 500: `arcene` (10000 features), `amazon-commerce-reviews` (10000), `dbworld-bodies` (4702), `dbworld-bodies-stemmed` (3721). See the operational section below.
- Total wall-clock on Apple M-series CPU: ~2 days for 142 datasets.

After filtering trivially-easy datasets (every model > 0.99 PR-AUC) and intersecting with TabPFN + CatBoost coverage, 107 datasets are usable for head-to-head analysis.

## Quantitative verdict (107 shared datasets)

| Metric                                       | TabICL          | TabPFN     | CatBoost (tuned) |
|----------------------------------------------|-----------------|------------|-------------------|
| Mean PR-AUC                                  | **0.839**       | 0.789      | 0.806             |
| Mean rank (1 = best)                         | **1.51**        | 2.23       | 2.25              |
| Rank-1 datasets                              | **62 / 107**    | 21 / 107   | 22 / 107          |
| Median train time (per dataset)              | 190 s           | 724 s      | 143 s             |
| Total compute on these datasets              | 20.7 h          | 157.5 h    | 18.5 h            |

Pairwise head-to-head wins (rows beat columns on mean PR-AUC):

| Row \ Column | TabICL  | TabPFN     | CatBoost   |
|--------------|---------|------------|------------|
| **TabICL**   | —       | 77 (72%)   | 81 (76%)   |
| **TabPFN**   | 30      | —          | 52 (49%)   |
| **CatBoost** | 24      | 55 (51%)   | —          |

Mean delta vs CatBoost: TabICL **+0.033**, TabPFN **−0.017**. Median delta: TabICL **+0.007**, TabPFN **0.000**.

Mean delta TabICL − TabPFN: **+0.050** (TabICL strictly better in aggregate).

## Where TabICL shines

Big wins concentrate where gradient boosters underfit due to noise or sparse signal:
- `seismic-bumps` +0.73 vs CatBoost (severe class imbalance — CatBoost lands near random PR-AUC, TabICL prior wins)
- `thoracic-surgery` +0.66, `hill-valley-with/without-noise` +0.31–0.33 (synthetic structured noise)
- `hepatitis` +0.28, `blogger` +0.22, `autoUniv-au1-1000` +0.29 (small-data noise pattern)

## Shared blind spots

TabICL and TabPFN fail on the **same** small medical / severely imbalanced datasets. The correlation between their deltas vs CatBoost is strong, both ways. These are the foundation-model failure mode for this kind of data:
- `blood-transfusion-service` (TabICL −0.36, TabPFN −0.56)
- `appendicitis`, `saheart`, `pima-indians-diabetes` — small medical sets with severe imbalance
- `thyroid-sick-euthyroid`, `wilt` — extreme imbalance, CatBoost ~0.998, TabICL ~0.91

Practical implication: foundation models are not a free win on every dataset. For severely imbalanced binary classification on small data, tuned GB with `class_weight="balanced"` is still the safe choice. **Co-train CatBoost alongside TabICL** — it's cheap and catches the blind spots.

## Operational

- **Default device is CPU.** MPS works on isolated single fits but **SIGSEGVs in detached background processes** during the repeated-fit GridSearch path (single fits = 0.5 s on MPS; running the same path under `nohup`/`run_in_background` crashes within seconds, exit 139). Reproducible — root cause not isolated, but the workaround (CPU-only) was straightforward.
- **OpenMP init race.** `import torch` must precede `xgboost`/`lightgbm`/`catboost` in `optuna_models.py`. Without it, libomp `__kmp_suspend_64` SIGSEGVs in a worker thread on the first dataset. Combined with `KMP_DUPLICATE_LIB_OK=TRUE` env var, eliminates the crash.
- **`batch_size=4`** in the wrapper (vs library default 8) caps peak ensemble-forward memory on this 24 GB machine. No accuracy effect.
- **Feature-count guard (≥ 500 features → NaN).** TabICL docs claim support up to 2000 features, but empirically `cnae-9` (856) stalled past 25 min with no completion; `arcene` (10000) ran 8h+ with RSS ballooning to 5–10 GB and never completed. Lower threshold catches all known offenders (`madelon` 500, `multiple-features` 649, `har` 561, `cnae-9` 856, `micro-mass` 1300, `arcene` 10000) without over-skipping.
- **HP grid is intentionally small.** TabICL authors state defaults are SOTA without tuning. Grid: `n_estimators ∈ {4, 8, 16} × softmax_temperature ∈ {0.7, 0.9, 1.1}`. That's 9 configs per outer fold via GridSearchCV.
- **`n_jobs=1` at GridSearchCV level** (like TabPFN) — TabICL uses PyTorch internally and parallel GridSearch workers trigger OMP mutex conflicts on macOS.

## Practical recommendation

For small tabular classification on this kind of data:

1. **Default to TabICL.** Zero HP tuning, ~3 min median train time, beats 50-trial-Optuna CatBoost on ~3/4 of datasets.
2. **Always co-train CatBoost.** It's the only model that meaningfully wins on the shared blind-spot datasets (severe class imbalance + small n), and it's cheap.
3. **Drop TabPFN.** TabICL strictly dominates it on accuracy and speed.

See [TabPFN_notes.md](TabPFN_notes.md) for the older foundation model's limits and [FT_transformer_notes.md](FT_transformer_notes.md) for the abandoned heavy-CPU foundation model attempt.
