# TabICL notes

[TabICL](https://github.com/soda-inria/tabicl) (INRIA's in-context tabular
foundation model, v2 checkpoint `tabicl-classifier-v2-20260212`) in the nested-CV
benchmark across 146 small tabular datasets.

For the cross-model comparison see [FoundationModels_notes.md](FoundationModels_notes.md);
this note covers what is specific to TabICL.

## TL;DR

TabICL is **the best foundation model to reach for on a CPU**: practically tied with
TabFM and TabPFN-3 on performance, 2.7x cheaper per fit than either, and it needs no
GPU. It beats tuned CatBoost on 85 % of non-trivial datasets at a median cost of
1.7x CatBoost.

It is not the strongest model in the benchmark. Both AutoML frameworks beat it, on
about 60 % of shared datasets.

## Coverage

- 142 / 146 datasets with valid scores (97 %).
- 4 skipped by a hard feature-count guard at 500 features: `arcene` (10 000),
  `micro-mass-mixed-spectra` (1300), `multiple-features` (649), `madelon` (500).
- A fifth over the threshold, `cnae-9` (856), has a valid score because it ran
  before the guard was added. A clean re-run of the current code would skip it, so
  coverage would be 141 / 146.
- Total wall clock: 34.7 h for 142 datasets, median 364 s per dataset.

## Where TabICL wins big

Not where the previous version of this note said. Two of its three headline
examples were a scoring bug, not a result:

| dataset | claimed vs CatBoost | actual |
|---|---|---|
| `seismic-bumps` | +0.73 | **+0.0026** |
| `thoracic-surgery` | +0.66 | **+0.0002** |

Both came from the positive-class defect described in
[Findings_notes.md](Findings_notes.md#label-ordering-silently-changed-the-metric):
on binary problems the metric reads `y_prob[:, 1]`, and the two paths were encoding
labels in opposite orders.

What survives is one clean pattern — synthetic structured noise, where gradient
boosting fails outright and the in-context prior does not:

| dataset | TabICL | CatBoost | delta |
|---|---|---|---|
| `hill-valley-without-noise` | 0.9999 | 0.6199 | **+0.3800** |
| `hill-valley-with-noise` | 0.9967 | 0.5560 | **+0.4407** |
| `blogger` | 0.8881 | 0.8515 | +0.0366 |
| `saheart` | 0.6509 | 0.6244 | +0.0266 |
| `wilt` | 0.9597 | 0.9358 | +0.0238 |

Outside the two `hill-valley` variants the margins are small. The aggregate lead
over CatBoost is +0.024 mean, +0.009 median: TabICL wins often and narrowly, plus
two datasets where it wins enormously.

## Blind spots: none that hold up

The previous version listed `blood-transfusion-service`, `appendicitis`, `saheart`,
`pima-indians-diabetes`, `thyroid-sick-euthyroid` and `wilt` as shared foundation
model failures on "severely imbalanced small medical data", and recommended
co-training CatBoost to cover them.

Every one of those was the same scoring bug. On `blood-transfusion-service` TabICL
scores 0.5284 against CatBoost's 0.5162 — it wins. Across all 146 datasets only two
have any classical model ahead of every foundation model by more than 0.02, and
neither is small, binary, or medical.

The co-training recommendation is withdrawn.

## Operational

- **Default device is CPU.** MPS works on isolated single fits but **SIGSEGVs in
  detached background processes** during the repeated-fit GridSearch path (single
  fits 0.5 s on MPS; the same path under `nohup` crashes within seconds, exit 139).
  Reproducible; root cause not isolated, workaround is CPU-only.
- **OpenMP init race.** `import torch` must precede `xgboost`/`lightgbm`/`catboost`
  in `optuna_models.py`, or libomp `__kmp_suspend_64` SIGSEGVs in a worker thread on
  the first dataset. With `KMP_DUPLICATE_LIB_OK=TRUE`, this eliminates the crash.
- **`batch_size=4`** in the wrapper (library default 8) caps peak ensemble-forward
  memory on this 24 GB machine. No effect on scores.
- **Feature-count guard at 500.** The docs claim support to 2000, but `arcene`
  (10 000 features) ran 8 h+ with RSS ballooning to 5-10 GB and never completed. The
  threshold is set on wall-clock risk, not hard failure: `cnae-9` (856) did finish,
  in 2.2 h.
- **HP grid is intentionally small.** The authors state defaults are SOTA untuned.
  Grid: `n_estimators ∈ {4, 8, 16} × softmax_temperature ∈ {0.7, 0.9, 1.1}`, 9
  configs per outer fold.
- **`n_jobs=1` at GridSearchCV level** — TabICL uses PyTorch internally and parallel
  GridSearch workers trigger OMP mutex conflicts on macOS.

## Practical recommendation

1. **Default to TabICL on CPU.** No tuning, ~6 min median, beats 50-trial-Optuna
   CatBoost on 85 % of non-trivial datasets.
2. **Switch to TabPFN-3 for wide or many-class data**, which TabICL skips.
3. **Run AutoML instead if 20 minutes a dataset is acceptable** — MLJAR beats TabICL
   on 61 % of shared datasets.
