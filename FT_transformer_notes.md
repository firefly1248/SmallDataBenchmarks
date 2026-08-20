# FT-Transformer notes

FT-Transformer (rtdl-revisiting-models) via Optuna in the nested-CV benchmark.

## Status: deferred, not measured

**No usable result exists.** The run reached 72 of 146 datasets with only 67 scored,
then stopped. That 67 is a biased subset — the datasets where the model happened not
to fail — so a mean over it is not comparable with the 146-dataset means every other
model reports, and it is not published.

The stored run also predates the stratified validation-split fix, so it is not
reproducible with current code. Its checkpoint is parked as
`results/ckpt/ft_transformer.joblib.bak-partial-run`.

Finishing it at the current 50-trial budget projects to **~863 h (36 days)**, and
that projection is itself unreliable because it scales from a run in which most fits
were failing.

## Why it failed, and why that matters

79 of 146 datasets produced no score, from the same defect that made TabNet look
cheap: the validation split inside the wrapper was unstratified and seeded from a
constant, so on any dataset where a class missed the validation fold, the logloss
metric raised — identically on every trial, every fold, every time.

The failure was silent. A dataset whose every trial raises completes quickly and
records NaN, so the run looked like it was making progress. Both this model and
TabNet were characterised in earlier notes on numbers produced this way.

The split is stratified as of 2026-08-02. See
[Findings_notes.md](Findings_notes.md#the-bug-that-produced-two-published-results).

## What the partial data suggested

Treat as anecdote, not measurement. Against the best of CatBoost / LightGBM /
XGBoost per dataset, over the 62 non-trivial datasets it covered, FT-Transformer ran
roughly 150x the compute for a mean delta of about −0.026 PR AUC. Its apparent wins
concentrated on datasets of 100-200 rows, where single outer folds swing wildly.

ResNet, which is measured properly on all 146 datasets, reaches 0.8234 mean
PR AUC for 207.7 h and still lands below Random Forest's 0.8359 at 7.6 h. Nothing in
the partial FT-Transformer data suggests it would land above ResNet.

## If it is ever run

- On MPS already, via `_train_rtdl_on_device`, except above 500 features where it
  falls back to CPU. That fallback costs about 3.5x and is what pushed
  `multiple-features` past the 12 h cap in the ResNet run.
- The Optuna space allows large architectures (`d_token` to 256, `n_blocks` to 6) on
  data of a few hundred rows; tying the space to dataset size is the first thing to
  try.
