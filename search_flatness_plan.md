# Plan: is the hyperparameter search bigger than it needs to be?

## Claims under test

The claims come from a newsletter that used HistGradientBoosting, one simulated 5,000-row dataset, 5-fold CV, ROC AUC and 60 random configurations.

- **C1a. Many configurations score alike.** 39 of 60 had a mean within one fold-SD of the best (0.930 ± 0.015), with learning rates from 0.01 to 0.47.
- **C1b. Hence the search need not be exhaustive.**
- **C2. Successive halving keeps the quality at lower cost.** `HalvingRandomSearchCV`, factor 3, starting on 500 of 5,000 rows. Its pick ranked 4th in the full random search over the same 60 candidates, within error, in 93 s against 154 s.
- **C3. The number of trees needs no search of its own: early stopping sets it.** 0.924 with early stopping against 0.929 for the same configuration with all 2,000 trees, at about a tenth of the training.

The article's manual narrowing step (re-searching a narrowed space after a coarse pass) is not tested.

The benchmark has 131 datasets, from 50 to 10,000 rows and 2 to 100 classes, and uses nested CV. The outer folds score every configuration on rows the selection never saw, so "within one SD inside CV" can be checked against held-out performance.

Every verdict holds for LightGBM, the benchmark's search space, weighted PR AUC, 4-fold inner CV and the 105 datasets where tuned LightGBM scores below 0.99. The article's conditions are approached by:
- the large stratum, which is primary for H3;
- the ROC AUC and learning-rate secondaries.

## Pre-registration

This file and `scripts/search_flatness.py` are committed before any run, and both hashes go into the notes. The analysis runs once, after both candidate seeds finish. The smoke test (see Compute) reads seconds only; its restated estimate goes into a separately committed note. Anything not specified here is labelled exploratory.

Evidence already on disk, cited rather than re-tested:
- **Trials.** TPE at 100 against 50 trials: +0.0000 and +0.0007 outer on two seeds.
- **Trajectories** (30 datasets). The pick from the first 10 random configurations scores 0.7880 to 0.8400 outer across three opening sets. Three tiny datasets drive the spread; without them the largest gap is 0.004.

The run scoring the best of 10 random opening configurations on all 131 datasets (`scripts/confopt_startup.py`) was in flight when this plan was fixed. Its results bear on H2. It finished on 2026-11-15 at 14:55 and its output was first read at 14:55, before any search_flatness code was written.

## Setup

**Model.** LightGBM over `lgbm_space()`: nine parameters, including `learning_rate` 1e-3–0.3 (log), `n_estimators` 50–500 and `class_weight`. One thread per fit. HGB is not run: re-timed one job at a time on 30 datasets, it cost 3.23 h against 0.64 h for LightGBM.

**Datasets.**
- The population is the 105 of the 131 non-duplicate datasets where the stored `lgbm_rerun` scores below 0.99 mean PR AUC.
- The other 26 are not run. At or above 0.99 the picks of a search almost always sit within 0.01 of the ceiling (one stored exception: a ConfOpt pick at 0.17 on one `pendigits` fold), so differences between picks are near zero; they would pull every mean towards 0 and every SD down.

**Large stratum.** The 25 datasets with 3,200 rows or more. There, halving's first round has at least 200 training rows, twice the largest `min_child_samples`; see the C2 secondaries for the effect of bagging. It is the closest to the article's 5,000 rows.

**Metric.** Weighted PR AUC (`pr_auc_score`) for selection and evaluation.
- Every training part contains every class. The smallest class anywhere has 11 rows, the subsets below keep at least one row per class, and the ES split is stratified.
- The runner asserts this instead of padding columns.

**Splits.** Outer 4-fold and inner 4-fold, both with seed `RANDOM_STATE`, identical to every stored run.

**Candidates.**
- 60 configurations per dataset, drawn by `optuna.samplers.RandomSampler(seed=zlib.crc32(dataset.encode()) + s)` and shared by that dataset's 4 outer folds.
- Two candidate seeds, s = 0 and 1.
- Drawing per dataset removes the common shift that one draw shared by all datasets would add to every difference.

**Parallelism.** As in `nested_cv.py`: outer folds and the 4 inner-split fits of a candidate run in threads, each fit on one thread.

**Timing.** Every fit and every `predict_proba` is timed with `time.thread_time()`, in every arm. The arms are interleaved per dataset, so performance and efficiency cores affect them alike. Search cost counts inner work only; outer refits are excluded.

**Failures.**
- A candidate with a NaN inner score ranks last everywhere.
- A fold whose pick has no outer score is dropped from that endpoint and counted.
- All ties are broken by candidate index.

## Arms

All arms run per dataset, outer fold and candidate seed, in one custom runner.

### R60

Every candidate gets an inner CV (per-fold PR AUC and ROC AUC, seconds) and a refit on the outer training part, scored on the outer test part (PR AUC and ROC AUC). The pick is the inner argmax on PR AUC.

### Rk

Random search with k of the 60 candidates. Its outer score is the exact expectation of the inner argmax over a uniformly random k-subset:
- with candidates ranked by inner score, ties by index, the j-th wins with probability C(60−j, k−1) / C(60, k);
- it is computed from the R60 table, with no fits;
- k = 20 is primary; k = 10 is descriptive.

### H60

Successive halving over the same 60 candidates, factor 3: 60 → 20 → 7, as sklearn's schedule for this case.

- **Subsets.** In each inner split, both the training and the test part are subsampled to 1/9, then 1/3, then all of it, as `HalvingRandomSearchCV` does (`subsample_test=True`).
- **Construction.** The subsets are nested and stratified, with max(1, floor(n_c · f)) rows of class c, seed `RANDOM_STATE`. One subset per inner split and round is shared by all candidates and both seeds.
- **Departures from sklearn.** `min_resources` is set to n/9, so that the third of sklearn's 3 rounds is all rows (the article: n/10, last round 4,500 of 5,000). The subsets are stratified, because sklearn's unstratified draw can drop a class. The last round uses all rows; sklearn uses 9 · (n // 9).
- **Scoring.** A round's score is the mean over the 4 inner splits.
- **Round 3** is all rows, so its 7 scores are R60's own: they are read from R60, and only rounds 1 and 2 are fitted.
- **Cost.** H60's cost is rounds 1 and 2 plus the R60 seconds of the 7 survivors.
- **Pick.** The outer score of the pick is read from R60.

### ES

The same 60 candidates without `n_estimators`.

- **Trees.** At most 500 trees, the comparator's own upper bound, with `lightgbm.early_stopping(20)` on log loss. Only how the tree count is chosen differs from R60. A larger cap would confound early stopping with more trees: 40 % of candidates have a learning rate under 0.01, and for those 500 trees binds.
- **Stopping set.** Carved inside each fit as a stratified 10 % of the fitting set (seed `RANDOM_STATE`, original row order kept). The encoder is fitted on the remaining 90 %.
- **Stopping weights.** The stopping set is weighted with the candidate's training weights: for "balanced", a dict computed on the 90 % part and passed as `eval_class_weight=[dict]`.
- **Outer refit.** Only the pick is refitted, with the same procedure, as HGB does.
- **Confounds.** ES trains on 90 % of the rows, R60 on all of them. This is part of what early stopping costs, and it is stated with the verdict.
- **Exclusions.** `plant-species-leaves-margin` and `-shape` are excluded by rule: their stopping set (about 90 rows) is smaller than their 100 classes. That leaves 103 datasets.

## Hypotheses and claim verdicts

The unit is the dataset. Its endpoint is the mean over its fold × seed cells.

| | Claim | Endpoint (outer PR AUC) | Datasets |
|---|---|---|---|
| H1 | C1a | mean over S1 without the argmax, minus the argmax. S1 = {c : mean_c ≥ mean_best − sd_best}, inner mean and SD per outer fold, SD with ddof 0 as in `cv_results_` | 105 |
| H2 | C1b | R20 − R60 pick | 105 |
| H3 | C2 | H60 pick − R60 pick | 25 (large) |
| H4 | C3 | ES pick − R60 pick | 103 |

Each claim's verdict is its hypothesis's verdict, except for C1a's premise rule below. There are no combined labels.

**H1 details.**
- A fold with |S1| = 1 has no H1 cell. A dataset with no cells is dropped and counted.
- If more than 10 % of the 105 datasets drop, H1 is inconclusive.
- **Premise rule.** If the median over the 105 datasets of each dataset's mean share |S1| / 60 is under 20 of 60 (the article had 39), C1a is reported as "premise fails" whatever H1 says. The share is reported with a bootstrap CI.
- **What H1 asks.** S1 admits candidates up to one inner SD below the best, a loose rule. H1 asks whether that looseness costs anything material on held-out rows, and δ defines material. If H1 is refuted, the size of the loss and the share of S1 are the information.
- Sensitivity: folds with |S1| = 1 scored as 0.

**C2's cost.** Below 1 is almost certain by construction (80 subset fits plus 7 full ones, against 60 full). So the H60 / R60 ratio of summed inner seconds is reported against the article's 0.60, as the median over H3's datasets, and C2's verdict rests on H3.

**C3's cost.** ES / R60 summed inner seconds and stopped trees / 500 per fit, both descriptive. C3's verdict is quality only.

**Decision rule (non-inferiority).**
- Margin δ = 0.0025 PR AUC.
- Interval: a percentile bootstrap of the mean of the per-dataset differences, 10,000 resamples, RNG seed 0, two-sided 97.5 %. That is one-sided 1.25 % per test: Bonferroni over four tests at a one-sided family-wise 5 %.

| verdict | condition |
|---|---|
| supported | lower bound above −δ; reported as "worse, within the margin" if the upper bound is also below 0 |
| refuted | upper bound below −δ |
| inconclusive | otherwise |

Each verdict comes with:
- the house line: mean, median, 10 % trimmed mean, ahead / behind, Wilcoxon p;
- the count of datasets with |difference| < 1e-4;
- a two-sided 97.5 % t-interval as a check: the percentile interval decides, and if the t-interval gives a different verdict, the verdict is reported as inconclusive.

Sensitivity analyses:
- δ = 0.005;
- a cluster bootstrap by family, where the family is the dataset name up to its first hyphen (`volcanoes-a1` → `volcanoes`);
- each candidate seed alone, shown next to the primary table.

**What δ is.** δ is a precision limit, not a judgement of what is immaterial: it equals the one-seed half-width of the proxy below on the 105. The notes report losses as small as 0.0020 (XGBoost DART) as findings. So "supported" means a loss under 0.0025, which can include a loss the notes would report.

**Power.** Two independent TPE searches differ per dataset with SD 0.0116 on the 105 and 0.0033 on the large 25 (the `lgbm_rerun` / `lgbm_seed1` pair; other stored seed pairs give 0.0094–0.0121 and 0.0027–0.0050, which would widen the large-25 half-widths up to 0.0022 for one seed and 0.0016 for two). This is a proxy: H2 is an expectation and will be tighter, while H4 changes every fit and may be wider; H1 has no proxy.

| | half-width | "supported" needs a mean above | "refuted" needs a mean below |
|---|---|---|---|
| one seed, 105 | 0.0025 | about 0 | about −0.0050 |
| one seed, large 25 | 0.0015 | about −0.0010 | about −0.0040 |
| two seeds (if seed noise dominates), 105 | 0.0018 | about −0.0007 | about −0.0043 |
| two seeds, large 25 | 0.0011 | about −0.0014 | about −0.0036 |

"Inconclusive" is a likely outcome for H2 and H4.

## Secondary analyses

All descriptive, raw p.

**C1a**
- Share |S1| / 60 by row tercile and class count.
- Range of learning rate and `num_leaves` within S1.
- Worst S1 member's outer score minus the argmax.
- Mean of all 60 minus the argmax (what search buys at all), and H1 as a fraction of it.
- H1 restricted to candidates with a learning rate of at least 0.01 (the article's lower bound), with the argmax and S1 recomputed within them.
- H1 with selection, S1 and evaluation on ROC AUC, defined as one-vs-rest, weighted, over the classes present in the test part.

**C1b**
- R10.

**C2**
- Rank of the H60 pick within R60, inner and outer (the article reports 4th).
- Share of round-1 candidates with constant predictions, and ties at each cut. With `subsample` bagging, a split needs subsample · n ≥ 2 · `min_child_samples`. About 30 % of candidates fail that at 200 rows, 10 % at 300, none from 500. Of the 25 large datasets, 12 start below 500 rows and 6 below 300.
- H3 on the 13 datasets with 8,000 rows or more, where the first round has at least 500 training rows.
- H3 on all 105, with first rounds below 200 rows, outside the method's range.

**C3**
- The article's own comparison: the ES pick against the same configuration refitted with all 500 trees, no early stopping, on all outer-training rows (4 fits per dataset and seed).
- H4 by row tercile of the 103 and on the large 25. A conclusion on C3 quotes both the 103-dataset and the large-25 result.
- H4 on the 23 datasets whose inner stopping set holds fewer than 2 expected rows of the rarest class.
- Share of fits that use all 500 trees.

**H1 and H2 on the large 25**, for comparison with H3.

## Checks

Before the run:
- R60's inner scores reproduce the benchmark objective (`FixedTrial`) on 2 datasets, |diff| < 1e-6.
- The runner's outer path reproduces the stored `lgbm_rerun` outer predictions when fed its stored `best_params`, on 2 datasets (atol 1e-12).
- H60's subset code at fraction 1 reproduces R60's inner scores on 2 datasets.
- Unit tests on the subsets: nested, stratified, at least one row per class, sizes as specified.

Before analysis:
- Row and candidate counts are complete.
- Every NaN and failed fit is listed.

## Compute

The run is a precondition-gated job: nothing else runs on the machine.

| arm, per seed | estimate | basis |
|---|---|---|
| R60 | about 4–5 h | 300 fits per fold, against 401 for the 5.3–5.4 h 100-trial TPE runs, on 105 of 131 datasets |
| H60 | about 1–2 h | rounds 1 and 2; small fits cost more than their share of rows |
| ES | about 4–7 h | 240 inner fits per fold plus the pick's refit; mean trees near R60's 275, up to 500 |
| ES pick at 500 trees | minutes | |

Total: 9–14 h per seed, 18–28 h for both. A smoke test on 3 datasets (one small, one mid-size, one with 10,000 rows) measures the seconds per arm.

## Implementation

- One script, `scripts/search_flatness.py`, with its own runner, because `optuna_models.py` stores neither per-fold inner scores nor the outer scores of every candidate.
- Checkpoints per dataset in `results/search_flatness/<arm>_seed<s>.joblib`, resumable.
- Outputs:
  - `results/search_flatness_candidates.csv`, one row per seed, dataset, fold and candidate;
  - `results/search_flatness_picks.csv`;
  - the printed output in `results/search_flatness.txt`.

## Out of scope

- HGB and other models.
- The manual narrowing step.
- Early stopping with a cap above the searched range.
- Halving on `n_estimators`.
- Optuna pruners.
- Calibration.
- A comparison with TPE.
- Any change to the benchmark's production search.
