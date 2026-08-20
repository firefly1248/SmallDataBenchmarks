# TabPFN notes

TabPFN 2.6 (`tabpfn>=7.1.1`) in the nested-CV benchmark across 146 small tabular
datasets. TabPFN-3 results are in
[FoundationModels_notes.md](FoundationModels_notes.md); this note documents 2.6,
which the benchmark still carries as the previous generation.

## Verdict

**Superseded.** v3 scores +0.0398 mean PR AUC over 2.6 on the 129 datasets both
cover (0.8246 -> 0.8644), and covers all 146 against 2.6's 129. There is no axis on
which 2.6 is preferable.

2.6 also costs the most of any model in the benchmark: **236.3 h**, more than
CatBoost, LightGBM, XGBoost, Random Forest, SVC, LogReg and SGD combined.

## Hard limits

- **Max 10 classes.** Hard validation error in `tabpfn.validation.validate_num_classes`.
  Affects **15** of the 146 datasets (`abalone-11class`, `connectionist-vowel`,
  `connectionist-vowel-reduced`, `kr-vs-k`, `letter`,
  `localization-for-person-activity`, `meta-data`, `movement-libras`,
  `movement-libras-10`, `plant-species-leaves-margin`, `plant-species-leaves-shape`,
  `tamilnadu-electricity`, `texture`, `turkiye-student`, `walking-activity`).
- **Two feature-count hangs** (`madelon` 500 features, `multiple-features` 649),
  bringing the total to 17 empty rows.
- **No native NaN handling.** Imputation required upstream.
- **No native categorical handling for high-cardinality features.**

## Performance limits observed

Runtime scales with feature count more than row count:

| Dataset             | Shape         | Outcome                |
|---------------------|---------------|------------------------|
| `pendigits`         | 7494 × 16     | ~11 h, completed       |
| `spambase`          | 4601 × 57     | ~8 h, completed        |
| `waveform-v2`       | 5000 × 40     | ~7.5 h, completed      |
| `mfeat-fourier`     | 2000 × 76     | ~3 h, completed        |
| `mushroom`          | 8124 × 21     | ~9 h, completed        |
| `cnae-9`            | 1080 × 856    | ~9.7 h, completed      |
| `madelon`           | 2600 × 500    | 20 h+ hang, skipped    |
| `multiple-features` | 2000 × 649    | 15 h+ hang, skipped    |

Datasets with 500+ features are unsafe without a hard wall-clock cap, though the cap
is about risk rather than a hard failure: `cnae-9` (856 features) finished.

## Quality

On datasets where it completes, 2.6 is **below** the tuned gradient boosters, not
above them: mean PR AUC 0.8246 against CatBoost's 0.8386 over its own coverage. It
also lands last or near-last more often than any other foundation model — its rank
distribution is bimodal, taking rank 4 on 22 % of datasets and rank 17 (last of 17)
on 21 %.

An earlier version of this note recorded the opposite, claiming its mean delta vs
the RF baseline was among the highest in the benchmark. That predates both the
positive-class fix and the re-fit of every classical model — see
[Findings_notes.md](Findings_notes.md#label-ordering-silently-changed-the-metric).

## Timeout caveats

- `signal.SIGALRM` per-dataset does **not** reliably interrupt TabPFN. Python signal
  handlers fire only between bytecode instructions, and long torch matmuls hold the
  GIL inside C extensions, so the alarm lands only after the operation returns. In
  practice `madelon` and `multiple-features` had to be killed and marked NaN by hand.
- For a reliable kill switch the per-dataset run must be in a subprocess with an
  external timeout.

## Operational notes

- Run single-threaded (`OMP_NUM_THREADS=1`) — it spawns its own torch workers and
  oversubscription degrades throughput.
- Checkpoint at the (dataset, model) level (`results/ckpt/<model>.joblib`) so
  hung-then-killed datasets can be marked NaN and skipped on restart.
