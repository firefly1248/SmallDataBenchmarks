# Tabular foundation models on small data

Tabular foundation models through the nested-CV benchmark (4 outer x 4 inner folds,
weighted PR AUC) on an Apple M4 Pro with 24 GB RAM.

| model | version | coverage | tuning | device |
|---|---|---|---|---|
| TabFM | `tabfm` 1.0.1, Google Research | 126 / 146 | none (zero-shot defaults) | MPS |
| TabICL | `tabicl` 2.1.1, INRIA | 142 / 146 | GridSearch, 9 configs | CPU |
| TabPFN-3 | `tabpfn` 9.0.0, Prior Labs | **146 / 146** | GridSearch, 4 configs | CPU |
| TabPFN-3.5 | `tabpfn` 9.0.0, Prior Labs | 131 / 131 | GridSearch, 4 configs | CPU |
| TabPFN-3.5-fast | `tabpfn` 9.0.0, Prior Labs | 131 / 131 | GridSearch, 4 configs | CPU |

The 3.5 rows cover 131 because runs now skip the 15 UCI++ duplicates, not because
anything was refused. TabPFN 2.6 was measured here earlier and has been removed: it
does not reproduce under `tabpfn` 9.0.0. Its numbers survive below where they are
part of a historical comparison, and the reason is in
[Findings_notes.md](Findings_notes.md#an-upgrade-that-moved-one-model-and-not-another).

## TL;DR

1. The current-generation models are **close on performance and far apart on cost**.
   TabFM 0.8653, TabPFN-3.5 0.8627, TabPFN-3.5-fast 0.8608, TabPFN-3 0.8591,
   TabICL 0.8574, each over its own coverage — a spread of 0.0079 against a cost
   range of 4.3 h to 73.6 h.
2. **TabPFN-3.5 supersedes TabPFN-3 outright**: higher score, 3.6x cheaper, and the
   paired test separates them (Holm p = 0.005, 71 wins of 106).
3. The **fast variant halves the cost again** for 0.0019 of mean PR AUC, which the
   test cannot separate from full 3.5 (Holm p = 0.06). It is the cheapest way into
   the top tier on a CPU.
4. What separates them otherwise is **coverage**. Every TabPFN version scores each
   dataset it is given; TabFM drops 20 and TabICL 4 to hard limits.
5. **There is no shared blind spot.** The earlier version of this note claimed 13
   such datasets. After the positive-class fix only 2 survive, and the explanation
   built on them was an artefact. See [The blind spot that wasn't](#the-blind-spot-that-wasnt).
6. Both vendors' speed claims fail on this workload. See [Cost](#cost-measured-not-advertised).
7. The gap to AutoML closed: the test no longer separates the top foundation models
   from either framework. See [Against AutoML](#against-automl).

## Performance: a three-way tie

Measured before TabPFN-3.5 existed and kept as the record of that round; the current
ordering is in the TL;DR above. 105 datasets where TabFM, TabICL, TabPFN-3,
TabPFN 2.6 and CatBoost all have scores,
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
| TabPFN-3.5-fast | 68 | 85 s | **1.3 s** | CPU |
| TabPFN-3.5 | 68 | 180 s | 2.6 s | CPU |
| TabICL | 148 | 364 s | 2.5 s | CPU |
| TabFM | 4 | 27 s | 6.8 s | **GPU** |
| TabPFN-3 | 68 | 1199 s | **17.6 s** | CPU |

Two claims do not hold up here.

**"TabPFN-3 is up to 20x faster than 2.5."** Measured on the version of the
benchmark that still carried TabPFN 2.6, it was **3.4x slower per fit**. Its lower
total came from not hanging on wide data, not from throughput. The vendor claim
targets million-row data on an H100 and does not transfer to hundreds-to-thousands
of rows on a CPU.

The 3.5 generation is where the speed actually arrived, and it arrived without a
claim attached: 13.5x faster per fit than v3 on the same hardware and the same
grid, with a higher mean.

**"TabFM is cheap."** Its 27 s median is an artefact of doing no hyperparameter
search at all — 4 fits against TabICL's 148. Per fit it is 2.7x *more* expensive
than TabICL, while running on the GPU against TabICL's CPU. Without a GPU it is not
usable: CPU inference measured 17-36x slower than MPS.

TabPFN-3.5-fast is now the cheapest per fit, and on the same CPU as TabICL. That
ordering is one release old: before 9.0.0, TabICL held this row by 7x.

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

MLJAR beats TabFM on 56 % of datasets, TabICL on 61 %, TabPFN-3 on 60 %, which put
the foundation models between tuned single models and AutoML.

That reading did not survive a test. On the 106 datasets the figures use, MLJAR's
0.0295 lead over TabFM is won on 61 of them, and Holm-corrected the pair is not
separable (p = 0.36). Six models now share the top bar of the critical-difference
diagram, both AutoML frameworks among them. AutoML is not a tier above; it is the
same tier bought with twenty minutes a dataset.

Two caveats keep this from being a clean verdict. The AutoML budget is fixed at
300 s per fold whatever the dataset, so it spends far more on easy data and is
capped on hard data; and AutoML ensembles many models, so it is a different kind of
object than a single estimator.

## TabPFN-3 to TabPFN-3.5

The second generational jump, and a different shape from the first. Over the 131
datasets a run now covers:

| | TabPFN-3 | TabPFN-3.5 | TabPFN-3.5-fast |
|---|---|---|---|
| mean PR AUC | 0.8571 | **0.8627** | 0.8608 |
| total hours | 63.6 | 17.8 | **9.3** |
| median per dataset | 1199 s | 180 s | 85 s |

Where 2.6 to v3 bought coverage at ten times the per-fit cost, v3 to v3.5 gives the
cost back: 3.6x cheaper overall, 13.5x cheaper per fit, and a higher mean. The
paired test separates 3.5 from v3 (Holm p = 0.005, 71 wins of 106).

The fast variant is a separate smaller model, not a mode of 3.5. It halves the cost
again and gives up 0.0019 of mean PR AUC, which the test cannot separate from full
3.5 (Holm p = 0.06) — though 3.5 does win 65 of the 106 head-to-head, so the
ordering is probably real and merely small.

Against TabFM the new model is an exact tie on means (-0.0002) while winning only
37 of 106. TabFM wins more often; TabPFN-3.5 wins by more when it wins.

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

1. **Default to TabPFN-3.5-fast** on CPU. Cheapest per fit of anything here, full
   coverage, and the test cannot separate it from full 3.5 or from AutoML.
2. **Use full TabPFN-3.5 when the extra 0.002 is worth double the compute.** It does
   score higher, and against TabPFN-3 the gain is real (Holm p = 0.005).
3. **Do not reach for TabPFN-3.** Its successor beats it on score, cost and
   reproducibility; the only reason to keep it is an existing pinned environment.
4. **TabFM only with a GPU.** Its performance lead is within noise, so the case for
   it is convenience — no tuning — not quality.
5. **AutoML is no longer the obvious upgrade.** At 20 minutes a dataset both
   frameworks land in the same statistical tier as the top foundation models, for
   more wall clock.

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
