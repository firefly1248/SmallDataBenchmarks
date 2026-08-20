# Tabular foundation models on small data

Four tabular foundation models through the nested-CV benchmark (4 outer x 4 inner
folds, weighted PR AUC) across 146 small tabular datasets, on an Apple M4 Pro with
24 GB RAM.

| model | version | coverage | tuning | device |
|---|---|---|---|---|
| TabFM | `tabfm` 1.0.1, Google Research | 126 / 146 | none (zero-shot defaults) | MPS |
| TabICL | `tabicl` 2.1.1, INRIA | 142 / 146 | GridSearch, 9 configs | CPU |
| TabPFN-3 | `tabpfn` 8.2.0, Prior Labs | **146 / 146** | GridSearch, 4 configs | CPU |
| TabPFN 2.6 | `tabpfn` 7.1.1, Prior Labs | 129 / 146 | GridSearch, 8 configs | CPU |

## TL;DR

1. The three current-generation models are **practically equivalent on performance**.
   Over the 105 non-trivial datasets all of them cover: TabFM 0.8386, TabICL 0.8359,
   TabPFN-3 0.8343 — a spread of 0.0043. No significance test was run.
2. They are **not** equivalent on cost: 24 s / 190 s / 933 s median per dataset.
3. What separates them is **coverage**. TabPFN-3 is the only model in the benchmark
   that scores all 146; TabFM drops 20 and TabICL 4 to hard limits.
4. **There is no shared blind spot.** The earlier version of this note claimed 13
   such datasets. After the positive-class fix only 2 survive, and the explanation
   built on them was an artefact. See [The blind spot that wasn't](#the-blind-spot-that-wasnt).
5. Both vendors' speed claims fail on this workload. See [Cost](#cost-measured-not-advertised).
6. All three lose to AutoML given a 300 s/fold budget. See [Against AutoML](#against-automl).

## Performance: a three-way tie

105 datasets where TabFM, TabICL, TabPFN-3, TabPFN 2.6 and CatBoost all have scores,
after dropping the 21 trivially-easy ones (every model above 0.99 PR AUC):

| model | mean PR AUC | mean rank | rank-1 | median time |
|---|---|---|---|---|
| TabFM | 0.8386 | 1.94 | 56 | 24 s |
| TabICL | 0.8359 | 2.53 | 20 | 190 s |
| TabPFN-3 | 0.8343 | 2.77 | 16 | 933 s |
| CatBoost (tuned) | 0.8116 | 3.98 | 10 | 214 s |
| TabPFN 2.6 | 0.7853 | 3.70 | 7 | 724 s |

Pairwise wins (row beats column):

| row \ col | TabFM | TabICL | TabPFN-3 | TabPFN 2.6 | CatBoost |
|---|---|---|---|---|---|
| **TabFM** | — | 68 | 77 | 86 | 86 |
| **TabICL** | 35 | — | 55 | 76 | 89 |
| **TabPFN-3** | 27 | 49 | — | 72 | 83 |
| **TabPFN 2.6** | 19 | 29 | 33 | — | 55 |
| **CatBoost** | 18 | 15 | 21 | 50 | — |

TabFM takes rank-1 on more than half the datasets while leading TabICL by 0.0027 on
average — it wins often, not by much. Read the rank counts as "who is on top most
often", not "by how much".

Against tuned CatBoost the gap is consistent: +0.027 / +0.024 / +0.023 mean PR AUC,
winning on 82 / 85 / 79 % of datasets.

## Coverage: the actual differentiator

Hard limits, all recorded as NaN rather than as a low score:

- **TabFM**: max 10 classes (architectural, raises in `fit`) and ~500 features.
  Loses 15 many-class datasets and 5 wide ones.
- **TabICL**: stalls on wide data; guarded at 500 features. Loses 4.
- **TabPFN 2.6**: max 10 classes plus multi-hour hangs on wide data. Loses 17.
- **TabPFN-3**: no losses. The class cap is lifted (`MAX_NUMBER_OF_CLASSES = 160` in
  the v3 checkpoint) and wide data completes instead of hanging.

On the 20 datasets where at least one of the other three fails, TabPFN-3 beats the
best classical model on **16 of 20**, mean delta +0.0109:

| dataset | TabPFN-3 | best classical | delta |
|---|---|---|---|
| `plant-species-leaves-shape` | 0.8987 | ResNet 0.8250 | +0.0737 |
| `movement-libras` | 0.9694 | SVC 0.9069 | +0.0625 |
| `madelon` | 0.9586 | CatBoost 0.9181 | +0.0405 |
| `walking-activity` | 0.6839 | LightGBM-linear 0.6505 | +0.0334 |
| `kr-vs-k` | 0.9031 | HistGradientBoosting 0.8702 | +0.0329 |
| `plant-species-leaves-margin` | 0.9501 | CatBoost 0.9243 | +0.0258 |

This is invisible in the head-to-head table above, which by construction only covers
datasets every model handles.

## Cost: measured, not advertised

Per-fit cost, normalising away the different search-grid sizes:

| model | fits per dataset | median per dataset | per fit | device |
|---|---|---|---|---|
| TabICL | 148 | 364 s | **2.5 s** | CPU |
| TabPFN 2.6 | 132 | 678 s | 5.1 s | CPU |
| TabFM | 4 | 27 s | 6.8 s | **GPU** |
| TabPFN-3 | 68 | 1199 s | **17.6 s** | CPU |

Two claims do not hold up here.

**"TabPFN-3 is up to 20x faster than 2.5."** Measured, it is **3.4x slower per fit**
than 2.6. Its lower total (73.6 h vs 236.3 h) comes from not hanging on wide data,
not from throughput. The vendor claim targets million-row data on an H100 and does
not transfer to hundreds-to-thousands of rows on a CPU.

**"TabFM is cheap."** Its 27 s median is an artefact of doing no hyperparameter
search at all — 4 fits against TabICL's 148. Per fit it is 2.7x *more* expensive
than TabICL, while running on the GPU against TabICL's CPU. Without a GPU it is not
usable: CPU inference measured 17-36x slower than MPS.

TabICL is the genuinely fast model here, on the cheapest hardware.

## The blind spot that wasn't

The previous version of this note reported 13 datasets where the best classical
model beat the best foundation model by more than 0.02 PR AUC, led by
`blood-transfusion-service` at +0.3661, and built an explanation on top of them:
first "severe class imbalance", then, in a correction, "small binary data".

Both explanations described a bug. `pr_auc_score` takes `y_prob[:, 1]` on binary
problems, so which class counts as positive follows label ordering; the classical
and foundation paths were encoding labels differently, and the affected datasets
were being scored against opposite classes — full write-up in
[Findings_notes.md](Findings_notes.md#label-ordering-silently-changed-the-metric).
After the fix:

| dataset | old gap | actual gap |
|---|---|---|
| `blood-transfusion-service` | +0.3661 | −0.0122 (foundation wins) |
| `appendicitis` | +0.2361 | −0.0314 (foundation wins) |
| `saheart` | +0.2132 | +0.0131 |
| `pima-indians-diabetes` | +0.1868 | +0.0079 |
| `thyroid-sick-euthyroid` | +0.0695 | −0.0077 (foundation wins) |

What survives across all 146 datasets:

| threshold | datasets where classical beats foundation |
|---|---|
| any margin | 30 |
| > 0.005 | 12 |
| > 0.01 | 7 |
| > 0.02 | **2** |
| > 0.05 | 1 |

The two are `planning-relax` (SVC 0.4044 vs TabPFN-3 0.3461) and
`localization-for-person-activity` (LightGBM-linear 0.7995 vs TabICL 0.7749). Two
datasets support no characterisation at all, and neither is small or binary —
`localization-for-person-activity` has 10 000 rows and 11 classes.

Foundation models match or beat the best of ten classical models on **116 of 146
datasets (79.5 %)**. The correct summary is that they rarely lose, not that they
lose in a describable place.

Where they win big is structured synthetic noise, where gradient boosting collapses
outright: `hill-valley-with-noise` CatBoost 0.5560 against TabICL 0.9967,
`hill-valley-without-noise` 0.6199 against 0.9999.

## Against AutoML

On the 124 datasets covered by every model including both AutoML frameworks:

| model | mean PR AUC | median time |
|---|---|---|
| MLJAR (300 s/fold) | 0.8965 | 21 min |
| AutoGluon (300 s/fold) | 0.8944 | 20 min |
| TabFM | 0.8721 | 25 s |
| TabICL | 0.8696 | 3.7 min |
| TabPFN-3 | 0.8682 | 17 min |
| CatBoost (tuned) | 0.8495 | 3.6 min |

MLJAR beats TabFM on 56 % of datasets, TabICL on 61 %, TabPFN-3 on 60 %. The
foundation models sit clearly between tuned single models and AutoML, not above
everything.

Two caveats keep this from being a clean verdict. The AutoML budget is fixed at
300 s per fold whatever the dataset, so it spends far more on easy data and is
capped on hard data; and AutoML ensembles many models, so it is a different kind of
object than a single estimator.

## TabPFN 2.6 to TabPFN-3

The largest generational jump in this benchmark: mean PR AUC 0.8246 to 0.8644
(**+0.0398**) on the 129 datasets both cover, and 129 to 146 datasets scored.

This invalidates the conclusion in [TabPFN_notes.md](TabPFN_notes.md) that TabPFN is
"strictly dominated" by TabICL — true for 2.6, false for v3, which ties TabICL on
performance and beats it on coverage.

The `arcene` case is the reverse of what the earlier note claimed: 2.6 completed it
in 0.47 h scoring 0.9656, and v3 took 4.71 h for 0.9690. Ten times the cost for
+0.0034.

## Practical recommendation

1. **Default to TabICL** on CPU. Practically tied with the others, cheapest per fit
   by 2.7x, and its coverage gap is 4 datasets.
2. **Reach for TabPFN-3 when the data is wide or many-class.** It is the only one
   that handles those, and it beats classical models there on 16 of 20.
3. **TabFM only with a GPU.** Its performance lead is within noise, so the case for
   it is convenience — no tuning — not quality.
4. **Drop TabPFN 2.6.** Superseded by v3 on every axis.
5. **If 20 minutes a dataset is acceptable, run AutoML instead.** Both frameworks
   beat all three foundation models on this benchmark.

The earlier recommendation to always co-train a classical baseline is withdrawn. It
rested entirely on the blind-spot table, and that table was a scoring bug.

## Method caveats

- **TabFM ran on MPS, everything else on CPU.** Its time column is not comparable.
  The measured CPU/MPS ratio on this machine is 17-36x.
- **TabFM's context was capped at 5000 rows** (`max_num_rows`), affecting 22 of its
  126 datasets. Uncapped, a 10 000-row dataset allocated 13.55 GB and drove the
  machine into swap. PR AUC across caps 2500 / 5000 / 7500 measured 0.476 / 0.501 /
  0.485 on `bank-marketing-full` — flat, so the cap does not appear to cost
  performance.
- **TabFM used `n_estimators=4`** against a library default of 32. Measured PR AUC
  spread across 1 / 4 / 8 / 32 members was under 0.005 while cost scales linearly.
- **TabPFN-3 ran without `OMP_NUM_THREADS=1`**, which [TabPFN_notes.md](TabPFN_notes.md)
  recommends, so its cost is an upper bound; the size of the effect was not measured.
- **Search grids differ by model** (148 / 68 / 4 fits per dataset), following each
  author's guidance on how much tuning their model needs. "Per dataset" costs
  therefore compare deployment recipes, not architectures.
- **Elapsed time covers the whole nested CV**, and the CV-level `n_jobs` differs by
  family: 16 for the classical models, 1 for the neural ones, serial for the
  foundation models.
- **Subsampling to 10 000 rows is unstratified** and runs after the per-class
  filter. Worst case measured: `kr-vs-k` rarest class 27 -> 11.
- **Model weights for TabPFN-3 and TabFM are non-commercial.** Both were used under
  the evaluation terms their licences permit. The code is separately licensed
  (Apache 2.0 with attribution for TabPFN, Apache 2.0 for TabFM).
