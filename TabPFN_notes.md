# TabPFN notes

Findings from running TabPFN 2.6 (`tabpfn>=7.1.1`) in the nested-CV benchmark across 146 small tabular datasets.

## Hard limits

- **Max 10 classes.** Hard validation error in `tabpfn.validation.validate_num_classes`. Affects ~8 datasets in this benchmark (e.g. `letter`, `vowel`, `kr-vs-k`). Recorded as NaN.
- **No native NaN handling.** Imputation required upstream (we use the wrapper's preprocessing path).
- **No native categorical handling for high-cardinality features.** TabPFN's internal handling works only for low-cardinality categoricals; high-card columns must be encoded externally.

## Performance limits observed

TabPFN runtime scales poorly with the number of features. For our 12h-per-dataset cap:

| Dataset             | Shape         | Outcome                |
|---------------------|---------------|------------------------|
| `pendigits`         | 7494 × 16     | ~11h, completed        |
| `spambase`          | 4601 × 57     | ~8h, completed         |
| `waveform-v2`       | 5000 × 40     | ~7.5h, completed       |
| `mfeat-fourier`     | 2000 × 76     | ~3h, completed         |
| `mushroom`          | 8124 × 21     | ~7h, in progress       |
| `madelon`           | 2600 × 500    | 20h+ hang, skipped     |
| `cnae-9`            | 1080 × 856    | 18h+ hang, skipped     |
| `multiple-features` | 2000 × 649    | 15h+ hang, skipped     |

Rule of thumb: datasets with **500+ features** are unsafe to run without a hard wall-clock cap. The slowdown is dominated by feature count, not row count.

## Timeout caveats

- `signal.SIGALRM` per-dataset (12h) is implemented in `optuna_models.py`, but it does **not** reliably interrupt TabPFN. Python signal handlers only fire between bytecode instructions; TabPFN's long torch matmuls hold the GIL inside C extensions, so the alarm fires only after the operation returns. In practice we had to kill and manually mark NaN for `madelon` and `multiple-features`.
- For a reliable kill switch, the per-dataset run must be in a subprocess with an external timeout (e.g. `multiprocessing.Process` + `terminate()` or `signal.SIGKILL` from the parent).

## Quality findings

- On datasets where TabPFN completes, it is consistently competitive with or above tuned GBM baselines (LightGBM, CatBoost, XGBoost) on PR AUC.
- The mean PR AUC delta vs. RF baseline (over completed datasets) is among the highest of all benchmarked models, including AutoGluon and MLJAR.
- Underperforms on high-dimensional datasets when it does complete — not just slow, also weaker accuracy compared to GBMs on those.

## TabPFN-3 upgrade path

TabPFN-3 (released early 2026) addresses the main blockers:

- Removes the 10-class cap.
- ~20x faster inference; scales reasonably to ~1M rows.
- Same sklearn-style API (`TabPFNClassifier`); drop-in replacement for the wrapper.

Upgrading would let us recover the ~8 NaN datasets dropped for the class limit, and likely shorten the high-dimensional runs enough that the 12h timeout is no longer needed.

## Operational notes

- Run TabPFN single-threaded (`OMP_NUM_THREADS=1`, etc.) — it spawns its own torch workers internally and oversubscription degrades throughput.
- Checkpoint at the (dataset, model) level (`results/optuna_models_ckpt.joblib`) so hung-then-killed datasets can be marked NaN and skipped on restart without re-running the rest.
