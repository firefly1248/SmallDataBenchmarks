# Early stopping on time-ordered data: plan

To be committed before any runner code, as `search_flatness_plan.md` was. Dataset preparation, the pre-run checks and the counts below come from `scripts/es_time_prep.py` (committed with this plan). Anything this plan does not specify is exploratory.

## Question

On data ordered by time, what does early stopping cost compared with tuning the number of trees on time splits and refitting on the freshest rows?

Context: the first post (search_flatness) tested early stopping only with a random stopping set on unordered data. On time-ordered data a random stopping set shares neighbouring rows with the training rows. A stopping set cut by time takes the freshest slice, the one you would rather train on.

## Arms

Common to all arms: LightGBM with the benchmark's `lgbm_space` with one change: the learning rate runs from 0.01 (not 0.001) to 0.3 on a log scale, the newsletter's lower bound. The rest as in `lgbm_space`: `num_leaves` 8 to 256, `min_child_samples` 5 to 100, subsample, colsample, `reg_alpha`, `reg_lambda`, `class_weight` balanced or none. 60 random candidates per dataset and seed, drawn as in search_flatness (Optuna `RandomSampler`, seed `crc32(dataset) + seed`), the same list for every arm.

Categorical columns use LightGBM's native categorical handling (pandas `category` dtype, levels taken from the whole dataset). The benchmark's ordinal encoder codes levels in order of first appearance and unseen levels as -1; on time-sorted rows that turns every categorical into a time index and fills the freshest stopping set with -1 codes. The runner therefore differs from search_flatness here, and check 4 runs through the same encoding.

Trees: the cap is 500 for every arm, and T samples `n_estimators` from 50 to 500, as in `lgbm_space`. In search_flatness, with learning rates down to 0.001, 25.3 % of early-stopped inner fits reached the cap; the higher lower bound is there to cut that share (6 % of the sibling's inner fits at learning rates of 0.01 and above).

Early stopping as in search_flatness `fit_es`: log loss on the stopping set, patience 20, class weights in the evaluation set when `class_weight` is balanced. The stopping set is 10 % of the fitting rows. A fit is capped when its best iteration exceeds 480 (cap minus patience).

- **T.** The number of trees is a hyperparameter of the search. Every fit uses all of its fitting rows, no stopping set.
- **ER.** Stopping set: a stratified random 10 % of the fitting rows, drawn with the seed. The model trains on the other 90 %. This is the first post's ES arm.
- **EF.** Stopping set: the freshest 10 % of the fitting rows by time. The model trains on the older 90 %.
- **DR, D.** Take a configuration, run an ER fit (DR) or an EF fit (D) on it, and refit it on all fitting rows with the best iteration of that fit as `n_estimators`. The count is not rescaled from 90 % to 100 % of the rows. Applied to T's pick the same procedures are called DR_T and D_T, and they need their own ES fits of that pick, ER_T and EF_T.
- **D_own.** D with its own selection. Inside every inner fit, the EF fit on the older 90 % of the inner window gives a best iteration; the candidate is refitted on all rows of the inner window with that count and scored on the validation block. D_own picks the candidate with the best mean score over the three blocks, then runs EF and the refit on the final window. The inner EF fits are the same fits EF uses for its own selection, so D_own adds only the refits.

Selection: T, ER, EF and D_own each pick the candidate with the best mean PR AUC over the three inner validation blocks, with the arm's own fitting rule inside every inner fit (T: plain fit; ER and EF: the early-stopped model on 90 %; D_own: the refit model).

Metric: average precision of the minority class, which is class 1 in every dataset (the benchmark's `pr_auc_score` is the average precision of class 1 for binary targets; its `weighted` option has no effect there).

Final fits per cut, seed and scheme (all on the same final window, scored on the test block):

| Fit | Configuration | Procedure |
|---|---|---|
| T | T's pick | plain fit, all rows |
| ER | ER's pick | early stop on a random 10 %, model on 90 % |
| EF | EF's pick | early stop on the freshest 10 %, model on 90 % |
| D | EF's pick | EF's best iteration (the EF row), refit on all rows |
| DR | ER's pick | ER's best iteration (the ER row), refit on all rows |
| D_own | D_own's pick | an EF fit of that pick, its best iteration, refit on all rows (two fits) |
| ER_T | T's pick | early stop on a random 10 %, model on 90 % |
| EF_T | T's pick | early stop on the freshest 10 %, model on 90 % |
| DR_T | T's pick | ER_T's best iteration, refit on all rows |
| D_T | T's pick | EF_T's best iteration, refit on all rows |
| D_T9 | T's pick | EF_T's best iteration divided by 0.9 and rounded, refit on all rows; the cap does not apply (descriptive) |

## Splits

Rows are sorted by time. Ties are broken by a seeded random key (seed 0) before anything else, so file order inside one timestamp never carries the label. Every position below is a share of the rows (after the subsample) rounded to a row and moved to the nearest timestamp boundary, so that no timestamp falls on both sides of a boundary. The rule covers test and validation block starts, the start of every fitting window and the start of the EF stopping set. Datasets without a timestamp (row order is time) are cut by position.

- Test blocks: [60, 73.3), [73.3, 86.7), [86.7, 100].
- Inner validation blocks for a cut at c: three blocks of 8 % right before it, [c − 24, c − 16), [c − 16, c − 8), [c − 8, c).
- **E2, fixed length (primary).** Every fit, inner or final, trains on the L = 36 % of rows right before its validation or test block. The validation blocks are 0.6 of a test block; the inner training windows keep the length L.
- **E1, expanding (sensitivity).** The same validation and test blocks. Every fit trains on all rows from the first row up to its block.

| Cut | Test | Validation blocks | E2 inner training | E2 final | E1 inner training | E1 final |
|---|---|---|---|---|---|---|
| 1 | 60-73.3 | 36-44, 44-52, 52-60 | 0-36, 8-44, 16-52 | 24-60 | 0-36, 0-44, 0-52 | 0-60 |
| 2 | 73.3-86.7 | 49.3-57.3, 57.3-65.3, 65.3-73.3 | 13.3-49.3, 21.3-57.3, 29.3-65.3 | 37.3-73.3 | 0-49.3, 0-57.3, 0-65.3 | 0-73.3 |
| 3 | 86.7-100 | 62.7-70.7, 70.7-78.7, 78.7-86.7 | 26.7-62.7, 34.7-70.7, 42.7-78.7 | 50.7-86.7 | 0-62.7, 0-70.7, 0-78.7 | 0-86.7 |

The early-stopping fits (ER, EF) train on 90 % of the fitting rows; T, D, DR, D_own and the other refits train on all of them. Inner validation looks 0 to 8 % of the rows ahead, the test block 0 to 13.3 %. In E1 the tree count is chosen on windows of 60 to 91 % of the final size and used on the final window.

## Population

Every dataset enters in a thin variant; datasets with over 20,000 rows also enter in a dense one when they pass the checks on it:
- **Thin (primary).** 10,000 rows drawn at random (seed 0), indices sorted, so the time order and span are kept.
- **Dense (sensitivity, E2 only).** The last 10,000 rows after sorting, contiguous (the thin draw keeps under half of these datasets). It checks H1 in a contiguous, short, recent regime. It is not a test of neighbouring-row dependence: after the tie shuffle, thinning leaves label persistence unchanged on seven of nine datasets (elec 1.67 to 1.41). nycflights fails check 3 here (its last 10,000 rows span 12 days, so the one-day label lag covers a quarter of a window) and has no dense variant.

**Pre-run checks** (in `es_time_prep.py`, per dataset and variant; a dataset that fails any check is dropped and listed):
1. Rows are sorted by a time column, or the row order is documented as chronological.
2. Minority class, counted after boundary adjustment: at least 15 rows in every test block, 10 in every inner validation block, 30 in every fitting window (E2 and E1, inner and final), and 5 in every stopping set (EF: the freshest 10 % of every fitting window; ER: the expected stratified 10 %).
3. Label horizon, as a tolerance: the share of rows in every E2 fitting window whose label becomes known after the start of the block the window feeds is at most 5 % (E1 windows share their ends and are longer, so their share is smaller). The prep script computes it for every dataset with a label date; datasets with an imputed time carry a conservative label date. The share inside the EF stopping sets is reported with it and is not limited: where it exceeds 5 %, EF stops partly on labels that would not exist yet, and H1 is also shown without that dataset. A dataset above the tolerance in its fitting windows is dropped, not embargoed.
4. Leak screen: default LightGBM (200 trees, learning rate 0.05, native categoricals) on the E2 final window of each cut has ROC AUC below 0.99 on that cut's test block. The random-split value is the mean of five draws of the same sizes from the rows of the window and its test block. Check 4 only guards against gross leaks; the per-feature screens of check 5 guard against the rest. A dataset whose time-split ROC AUC is below 0.65 on all three cuts is marked low-signal.
5. Absolute time columns: calendar year and time index are dropped; so are month, week and day of month whenever the shortest E2 fitting window spans under a year. Cyclic ones (month in windows of over a year, weekday, hour, period of day) stay. For every retained numeric column the script prints its Spearman correlation with row position. For every retained feature, categoricals included, in both variants, it computes the one-feature ROC AUC of the label (window to test block, cut 1) and the one-feature ROC AUC that separates the window from the test block. Features above 0.8 on either are reviewed by hand and the decision is written in the preparation list below.

**Included, thin variant** (counts from `es_time_prep.py`, after boundary adjustment and tie shuffle; "minority per test block" is the class scored; the fitting-window and stopping-set minima are over E2 and E1, inner and final windows; persistence is the share of consecutive rows with the same label over its value under independence; ROC AUC from check 4: the time split on each cut, and the mean of five random splits of the same sizes on cut 1):

| Dataset | Source | Rows (used) | Span | Class scored, share | Minority per test block | Min. per inner block | Min. per fit window | Min. per stopping set, EF / ER | Persistence | ROC AUC time, cuts 1 / 2 / 3 | ROC AUC random, cut 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| hotels | TidyTuesday 2020-02-11 (Antonio et al. 2019) | 119,390 (10,000) | 2015-07 to 2017-08 | cancelled, 36.7 % | 467 / 539 / 515 | 277 | 1,171 | 114 / 117 | 1.05 | 0.804 / 0.821 / 0.773 | 0.833 |
| himalaya | TidyTuesday 2020-09-22 (Himalayan Database) | 10,268 (10,000) | 1905 to 2019 | no success, 44.2 % | 607 / 625 / 452 | 287 | 1,479 | 136 / 147 | 1.16 | 0.676 / 0.650 / 0.704 | 0.689 |
| wildlife | TidyTuesday 2019-07-23 (FAA strikes, 4 airlines) | 52,654 (10,000) | 1990 to 2018 | any damage, 7.5 % | 66 / 55 / 66 | 32 | 186 | 10 / 18 | 1.00 | 0.808 / 0.828 / 0.805 | 0.763 |
| f1-dnf | TidyTuesday 2021-09-07 (Ergast) | 23,525 (10,000) | 1950 to 2021 | not classified, 41.0 % | 509 / 301 / 239 | 165 | 1,270 | 66 / 127 | 1.07 | 0.542 / 0.546 / 0.571 | 0.588 |
| nba | FiveThirtyEight nba-elo | 63,157 (10,000) | 1946 to 2015 | home loss, 37.8 % | 552 / 495 / 527 | 273 | 1,274 | 110 / 127 | 0.99 | 0.685 / 0.662 / 0.675 | 0.661 |
| elec | streaming-datasets (Harries 1999) | 45,312 (10,000) | row order, NSW 1996-98 | price up, 42.4 % | 516 / 506 / 596 | 294 | 1,426 | 103 / 142 | 1.41 | 0.693 / 0.761 / 0.832 | 0.833 |
| nebraska-weather | streaming-datasets (Elwell and Polikar 2011) | 18,159 (10,000) | row order, 1949-99 | rain, 31.0 % | 400 / 462 / 478 | 196 | 1,034 | 83 / 103 | 1.13 | 0.843 / 0.860 / 0.823 | 0.862 |
| nycflights | Rdatasets, nycflights13 | 327,346 (10,000) | 2013 | arrival 15+ min late, 24.6 % | 247 / 235 / 384 | 114 | 760 | 40 / 76 | 1.12 | 0.587 / 0.599 / 0.572 | 0.618 |
| bank-full | UCI++ in the benchmark | 45,211 (10,000) | row order, May 2008 to Nov 2010 | subscribed, 11.4 % | 142 / 149 / 554 | 35 | 157 | 12 / 15 | 1.05 | 0.617 / 0.594 / 0.529 | 0.559 |

Check 3 on the thin variant: himalaya and nycflights each reach 1 % of a fitting window and 10 % of an EF stopping set. For himalaya, 22 % of the rows have an imputed time (base camp plus 25 days for 13 %, a season fallback for 9 %) and carry a label date 47 days later, the 90th percentile of the gap on dated rows. Check 5: no numeric feature has a Spearman correlation with row position above 0.5 in absolute value (the largest are bank-full `previous` at 0.50 and `pdays` at 0.49). Single-feature screens above 0.8, thin and dense, all reviewed and kept: wildlife `species_id` (label 0.84; the species is identified from the remains after the strike, like every other wildlife column, which are all post-event reports), f1-dnf `driverId` and `constructorId` (window against test 0.97 and 0.86: the drivers and teams of an era), elec `nswprice` (label 0.72, drift 0.82: the price level itself) and, in the dense variant, elec `period` (label 0.80, time of day). Where a column is an era identifier, H1 is read against the unseen-level diagnostic.

**Included, dense variant** (E2 only; the last 10,000 rows, contiguous):

| Dataset | Span | Class scored, share | Minority per test block | Min. per inner block | Min. per fit window | Min. per stopping set, EF / ER | Persistence | ROC AUC time, cuts 1 / 2 / 3 | ROC AUC random, cut 1 |
|---|---|---|---|---|---|---|---|---|
| hotels | 2017-07 to 2017-08 | cancelled, 37.1 % | 469 / 534 / 445 | 291 | 1,269 | 103 / 126 | 1.01 | 0.705 / 0.727 / 0.784 | 0.789 |
| wildlife | 2016-05 to 2018-12 | any damage, 5.2 % | 99 / 51 / 84 | 21 | 177 | 6 / 17 | 1.00 | 0.754 / 0.699 / 0.765 | 0.777 |
| f1-dnf | 1995-09 to 2021-09 | not classified, 27.4 % | 240 / 269 / 209 | 139 | 703 | 57 / 70 | 1.04 | 0.601 / 0.545 / 0.558 | 0.620 |
| nba | 2007-12 to 2015-06 | home loss, 40.0 % | 510 / 561 / 564 | 300 | 1,396 | 131 / 139 | 1.01 | 0.676 / 0.646 / 0.657 | 0.678 |
| elec | row order, rows 35,312 to 45,311 | price up, 43.6 % | 642 / 523 / 593 | 286 | 1,301 | 112 / 130 | 1.66 | 0.922 / 0.865 / 0.887 | 0.941 |
| bank-full | row order, rows 35,211 to 45,210 | subscribed, 29.1 % | 589 / 710 / 664 | 119 | 281 | 34 / 28 | 1.16 | 0.585 / 0.629 / 0.656 | 0.684 |

The dense variant is a different regime for several datasets: hotels spans two months, and the class share moves away from the full-data value in bank-full (29.1 % against 11.7 % in the full file), f1-dnf and wildlife. In dense hotels the realized EF stopping set is 8 to 12 % of its window, because about 165 rows share a date. Its results are read as a check on H1, not pooled with the thin ones.

**Preparation per dataset** (in `es_time_prep.py`; the class scored is the minority class everywhere):
- hotels: time = arrival date; `agent` and `company` are ids and enter as categoricals. Dropped: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, `country`, `total_of_special_requests` (changed or recorded after booking or at check-in), `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month` (absolute time inside a window under a year). Ordered by arrival, every training label is known before the cut.
- himalaya: one row per expedition, "Unknown" termination dropped; time = termination date (78 % of rows), else base camp date plus the median gap of 25 days (13 %), else year and season (9 %); the label date of an imputed row is its time plus 47 days. Dropped: `termination_reason`, `highpoint_date`, `termination_date`, `basecamp_date`, `highpoint_metres`, both death counts, `oxygen_used` (recorded during the climb), `expedition_id`, `peak_name`, `year`.
- wildlife: rows without a damage code dropped; "M?" damage codes count as damage; in `precip` "None" (no precipitation) is a level and only "NA" is missing. Dropped: `cost_repairs_infl_adj`, `airport` and `species` (names of coded columns), `incident_year`. Reports are filed after the incident; the lag is not modelled.
- f1-dnf: results joined with races and status; classified = "Finished" or "+N Laps"; drivers who never started (position codes F and W) dropped; the class is "did not finish by status" (641 rows with a numeric classification but a retirement status count as positive). Features: grid, driver, constructor, circuit, round, the three ids as categoricals.
- nba: the non-copy row of each game, which is the home team's row; label = home loss; pre-game columns only (`elo_i`, `opp_elo_i`, playoffs, game of the season, franchises, league).
- elec: row order; all columns but the label and `vicprice`, `vicdemand`, `transfer` (a constant fill over the first 37.5 % of rows).
- nebraska-weather: row order; all columns but the label.
- nycflights: flights without an arrival delay dropped (cancelled and diverted). Features: scheduled departure and arrival time, carrier, origin, destination, distance, hour. The label is known within a day of the scheduled departure.
- bank-full: the benchmark's ARFF, file order; call duration, month, day of month and `contact` (an era flag: "unknown" in about all of the first tenths of the file, about none at the end) dropped. `campaign` counts contacts including the last one and stays; `poutcome`, `pdays` and `previous` are empty for the first half of the file and stay as campaign history.

**Excluded**, with the reason:
- launches (TidyTuesday 2019-01-15, JSR launch log; 5,712 rows, failure 5.7 %): the smallest EF stopping set holds 2 failures against the minimum of 5, and the test blocks hold 31, 29 and 24.
- ozone-eighthr: 10 and 11 minority rows in two test blocks, 5 in an inner block. ozone-onehr: 2 to 7 per test block.
- eeg-eye-state: one recording; inner blocks with no minority rows.
- cholera-tz-result: 4 minority rows in the first test block, none in an inner block.
- dengue-mexico-deaths: 70 deaths in the 10,000-row sample, 2 to 9 per test block.
- biotech-y36: 36-month outcome; 99 to 100 % of each training window has a label that becomes known after the cut.
- water (TidyTuesday 2021-05-04): report dates mark upload campaigns (about 100,000 rows each in 2010 and 2015, dates back to 1899), not the time of the process.
- IEEE-CIS (Kaggle): fraud labels come from chargebacks reported up to about 120 days after the transaction, against fitting windows of about 65 days, so it fails check 3. Kaggle data would also stay local.
- TabReD binary sets: Kaggle only, hundreds of thousands of rows and up to about 700 features.

## Seeds

Five seeds, 0 to 4. The seed changes the candidate list, the ER stopping set and LightGBM's `random_state`; the EF stopping set does not depend on it. The 10,000-row subsample is the same for all seeds (seed 0).

## Hypotheses and decision rule

Each hypothesis is a signed difference of average precision on the test block, positive when the first fit is better. Per dataset: the mean over the three cuts and five seeds. Pooled: the mean of the per-dataset values over the included datasets.

| | Difference | What it isolates | Expected |
|---|---|---|---|
| H1 | D_T − DR_T | random vs freshest stopping set, through the tree count only: same configuration (T's pick), same refit rows; the two counts come from models trained on different 90 % of the rows | D_T ahead or equal |
| H2 | D − EF | the cost of holding out the freshest 10 %: same pick, same tree count, refit on all rows against model on 90 % | D ahead or equal |
| H3 | T − D_own | two full pipelines, each selected under the rule it is deployed with | none |
| H4 | T − ER | tuned `n_estimators` vs early stopping on a random 10 %, whole procedures (the first post's ES arm) | none |

H1 covers the tree-count channel on T's pick only. In search_flatness an early stop of one configuration, against the same configuration at 500 trees, moved PR AUC by −0.0025 on average (103 datasets), and the ER and EF counts on one configuration will differ by much less than the gap between an early stop and the cap, so H1 is expected to be small. The stopping set's effect through selection is the descriptive difference D − DR. H3 and H4 compare procedures: T draws `n_estimators` together with eight other parameters, so they also measure search efficiency, and in H4 the final models differ in rows (100 % against 90 %).

Descriptive differences with no label: D − DR (the stopping set through selection and tree count), T − EF (the headline question for early stopping without refit; it equals H2 + H3 + (D_own − D)), D − D_own (the selection rule), DR − ER, D_T9 − D_T (rescaling the tree count), the ratio of the ER_T and EF_T best iterations (the mechanism behind H1), and the picks themselves (share of datasets and cuts where two arms, in particular D_own and EF, pick the same candidate).

**Interval.** A two-sided t-interval on the per-dataset means, n − 1 degrees of freedom, at 98.75 % (5 % two-sided, Bonferroni over the four E2 hypotheses). The percentile bootstrap is not used: at about ten datasets it covers 85 to 94 % when it claims 97.5 %. The half-width is reported with every label.

**δ.** A precision limit estimated from the run's own T results, not a materiality judgement, and not an error-controlled claim. Per dataset, for each of the 15 splits of seeds 0 to 4 into two disjoint pairs A and B (E2, thin): the mean over cuts of T's test score averaged over A minus T's score averaged over B. For each split the half-width is that of the pooled interval above, with the standard deviation taken around the sample mean over the datasets. δ is the root mean square of the 15 half-widths. A two-seed difference has the variance of one seed, a five-seed difference of two independent arms has 0.4 of it, so δ overstates the five-seed noise by about a factor of 1.6 before pairing by candidate list, which the A/A contrast lacks. A separate step reads only T's rows and writes δ with the hashes of its inputs to the run log; the analysis step refuses to run without it and never recomputes it. δ stays at its full-data value in every subset analysis. A T-only pilot on seeds outside 0 to 4 (not part of the run) gave δ of the order of 0.005 to 0.012 AP, set largely by the noisiest datasets (elec, wildlife).

Labels per hypothesis, from the pooled interval:
- **first fit ahead**: the interval lies above 0; **second fit ahead**: it lies below 0; **no difference found**: it contains 0;
- **beyond δ**, added to a direction: the interval lies above +δ (below −δ);
- **equivalent within δ**: the interval lies inside (−δ, +δ). It can come with a direction ("first fit ahead, within δ").

δ is printed with every label. With δ of that size, "beyond δ" needs a pooled difference of roughly δ plus the half-width, and "equivalent" needs a half-width below δ; many labels will be out of reach and the post says so. The four E2 thin labels are the primary result and the only ones the post quotes as verdicts.

Next to every pooled label: the number of datasets where the first fit is ahead, behind and tied (|difference| below 1e-4), and a robustness flag. A dataset is flagged when removing it changes the sign of the pooled estimate or moves it across +δ or −δ, with δ held at its full-data value. A flagged label is quoted as not robust. Pooled estimates are also shown without the low-signal datasets (check 4) and, for H1, without the datasets above 5 % in the EF stopping-set share of check 3.

Cap rule: H1 is also reported without the datasets where more than 25 % of the ER_T or EF_T final fits are capped, because a capped stop says nothing about the stopping set; the share of cells where D_T and DR_T have equal tree counts is shown beside it. The capped share is computed for every ES fit (ER, EF, ER_T, EF_T, the EF fit of D_own), over final fits and, where inner fits exist (ER, EF), over inner fits as a second column. The smoke run reports it and changes nothing in the design.

E1 and the dense variant are sensitivity analyses: the same four estimates and t-intervals at the same level, no verdict labels, no further correction. E1 shares data, blocks and candidates with E2, so agreement in sign between them is not independent evidence and is not counted.

Per dataset (descriptive, no verdict): the four differences per cut and per seed, with the mean and the 95 % t-interval over the five seeds (4 degrees of freedom). The interval covers seed variation only; the test rows are one sample and neighbouring rows are dependent. Pooled and per-dataset tables show the three cuts separately before they are averaged, since cuts differ in regime (bank-full minority share 11, 11 and 42 % in its test blocks). Pooled differences mix datasets whose average precision baselines run from 0.07 to 0.44 and whose test blocks hold from 55 to over 500 minority rows, so one or two datasets can set the pooled sign; the robustness flag is the protection.

## Descriptive diagnostics

- Trees per arm and cut; capped share of fits (best iteration above 480) per arm, inner and final.
- Optimism of the stopping set: log loss and PR AUC on the stopping set against the test block, ER vs EF.
- H1 against label persistence (measured after the tie shuffle and the thinning), as a scatter plot only.
- H2 against the minority share of the EF stopping set minus that of the test block, per cut, and against its share minus that of the older 90 %, as scatter plots only.
- Share of stopping-set rows with a categorical level absent from the model's training part, EF vs ER, and the same share for the test block; H1 on f1-dnf and himalaya is read against it.
- The best iteration by average precision on the same stopping set, beside the log-loss one.
- Thread seconds per arm.

## Runner changes against `scripts/search_flatness.py`

`fit_es`, `carve`, `candidates` and `build_lgbm` cannot be reused as written:
- `carve` takes the seed instead of the constant `RANDOM_STATE`, and a time carve (the last 10 % by position, snapped) is added.
- `random_state` is set from the seed after the model is built.
- The search space starts at learning rate 0.01; a test checks every candidate's learning rate is at least 0.01.
- ES arms drop the drawn `n_estimators`; D and the other refits set it from the stored best iteration.
- The runner reads the prepared files of `es_time_prep.py` in their order (no re-sorting) and asserts their sha256 against the manifest and that `t` and `label_known` are not among the features.
- Categoricals go in as pandas `category` dtype, with the level set of the prepared file.
- `fit_es` returns more than `best_iteration_`: the best iteration by average precision, the stopping-set log loss and average precision at the stop, and the share of stopping-set and test rows with an unseen level. The EF best iteration is kept per candidate and block, since D_own's refits need it.
- The verdict function has the labels above, and δ comes from its own step.

## Failures and runner assertions

- A candidate with a NaN inner score ranks last; ties go by candidate index.
- A cut whose pick has no test score is dropped from that difference and counted.
- The analysis runs once, after all seeds finish.
- Asserts: no timestamp lies on both sides of any boundary; every fitting row precedes every stopping, validation and test row of its fit; E2 windows have equal length up to the boundary shift; both classes are present in every fitting part, stopping set and block, and the check 2 minima hold in every stopping set; all arms of a dataset and seed share one candidate list; every refit's tree count equals the best iteration of the ES fit it names (D: EF; DR: ER; D_own: its own EF fit; D_T: EF_T; DR_T: ER_T); T is fitted without an evaluation set; subsample indices equal the prep script's; the row count per arm, cut, scheme and seed is complete; library versions and the plan and runner hashes are recorded.

## Compute

Per dataset, seed, scheme and cut: 3 blocks × 60 candidates × 4 selection rules (T, ER, EF, D_own) = 720 inner fits, 12 final fits, so 732. Over three cuts and two schemes that is 4,392 per dataset and seed, about 22,000 per dataset over five seeds, about 198,000 over nine datasets. The dense variant adds E2 only (about 11,000 per dataset over five seeds, about 66,000 over the six datasets that qualify). The smoke run on three named datasets (nba, elec, hotels) converts the count into hours and prints timings and capped shares only, never test scores. The compute budget in thread-hours is set after the smoke run, from its estimate. If the estimate exceeds that budget, the plan is reduced in a fixed order: the dense variant first, then E1; the seeds stay at five.

## Out of scope

XGBoost and CatBoost; tuning patience; TPE; multiclass and regression targets; group-aware splits (race, expedition, client); embargo gaps; data above 10,000 rows other than the thin and dense variants.

## Limits

- Nine datasets: the pooled interval is coarse, and many labels may be inconclusive.
- Most datasets are open data from GitHub mirrors (TidyTuesday, FiveThirtyEight, streaming-datasets, Rdatasets). The prep script records the sha256 of every file it reads. Licences and provenance are checked before any file enters the repository.
- Thinning leaves label persistence unchanged on most datasets, so the thin sample keeps whatever neighbouring-row structure a random stopping set could exploit; the dense variant is a short contiguous regime, checked on E2 only.
- Three of the nine datasets (f1-dnf, nycflights, bank-full) carry little signal under a time split (ROC AUC 0.53 to 0.62 on every cut); arms can hardly differ there, which pulls pooled estimates towards zero. Pooled estimates are also shown without the low-signal datasets.
- Labels are treated as known at the event time within the tolerance of check 3. Wildlife reports are filed after the event. A quarter of himalaya's last test block is ordered by an imputed date.
- Early-stopping fits train on 90 % of the rows that T, D, DR and D_own use, and the tree count is carried from 90 % to 100 % unscaled.
- Log loss on a stopping set whose minority share differs from the training part (f1-dnf, nycflights, bank-full, himalaya) stops for calibration reasons as well as ranking ones, so H1 mixes fresh rows with log loss under shift.
- Patience 20 at learning rate 0.01 on stopping sets of a few hundred rows is noisy; for the smallest datasets the early-stopping tree counts are partly noise.
- The three cuts are one correlated series: consecutive E2 final windows share 22.7 of 36 points, and cut 2's validation blocks contain cut 1's test block. The mean over cuts is not three independent tests.
- Learning rates below 0.01 and ensembles above 500 trees are not covered.
- E1, E2 and the dense variant share data and candidates, so they are not independent.

## Run conditions

- Check mode first (the pre-run checks above, per dataset and variant), then a smoke run on three datasets to estimate hours, as in search_flatness. The smoke run prints timings and capped shares; it changes nothing in the design beyond the fixed reduction order in "Compute".
- This plan is committed before the runner, and the runner before any results.

## Post

Written only after the run, from E2 labels and the per-dataset tables. The author's position is written after the results.

## Before the commit

- Licences of the GitHub-hosted datasets.
- The compute budget in thread-hours: set after the smoke run, from its estimate, and written here before the full run.
