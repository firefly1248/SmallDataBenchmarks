# FT-Transformer notes

Findings from running FT-Transformer (rtdl-revisiting-models) via Optuna in the nested-CV benchmark. The run was stopped after ~3 days on 67 of 146 datasets; the data is sufficient to draw a clear conclusion.

## TL;DR

FT-Transformer is **not worth running on CPU** for small tabular data at this benchmark's scale and search budget. It is dramatically slower than tuned gradient boosters and, on the datasets evaluated, does not match their accuracy on average.

## Coverage

- 67 / 146 datasets with valid scores (47%), 5 with NaN.
- ~3 days of wall-clock on Apple M-series CPU, `OMP_NUM_THREADS=4`, `caffeinate`-wrapped.
- Optuna budget: 50 trials × 4 outer folds per dataset (same as other Optuna models in this benchmark).

After filtering trivially-easy datasets (every model > 0.99 PR-AUC), 62 datasets are usable for head-to-head analysis.

## Quantitative verdict (62 datasets)

Comparison is against the **best of CatBoost / LightGBM / XGBoost** per dataset.

| Metric                         | Value                  |
|--------------------------------|------------------------|
| FT-T wins (any margin)         | 14 / 62  (23%)         |
| FT-T wins by ≥ 0.01 PR-AUC     |  9 / 62  (15%)         |
| FT-T loses by ≥ 0.01 PR-AUC    | 32 / 62  (52%)         |
| FT-T loses by ≥ 0.02 PR-AUC    | 23 / 62  (37%)         |
| Mean   ΔPR-AUC vs best GB      | **−0.0257**            |
| Median ΔPR-AUC vs best GB      | **−0.0118**            |
| Median train time ratio FT-T / fastest GB | **109×**    |
| Mean   train time ratio FT-T / fastest GB | 193×        |
| Total compute, FT-T            | 133.6 h                |
| Total compute, best GB (same datasets) | 0.9 h          |

For roughly **150× the compute**, FT-T delivers a **negative mean PR-AUC delta**.

## Patterns

**Wins concentrate on tiny datasets.** Top wins: `hepatitis` (155 rows, +0.27), `blogger` (100, +0.15), `autoUniv-au1-1000` (+0.22). These margins exceed what is plausible from a "better model" and look like outer-fold variance — single splits on a 100-row dataset swing wildly.

**Losses are not subtle and look like training failures.** `blood-transfusion-service` −0.38, `appendicitis` −0.23, `artificial-characters` −0.19, `hill-valley-without-noise` −0.19. PR-AUC drops of this size against tuned gradient boosters suggest the model is not learning a useful representation on those datasets — likely a combination of:

- Optuna's search space allowing too-large architectures (`d_token` up to 192, `n_blocks` up to 3) on small data,
- short training schedule for a transformer to converge,
- no problem-aware regularization beyond what the search exposes.

## Operational

- CPU-only run. PyTorch thread limits (`OMP_NUM_THREADS=4`, etc.) reduced thermals significantly with minor throughput cost on this hardware.
- Single-dataset cost ranged from ~300s (small, dense) to **27000s ≈ 7.6h** (e.g. `eeg-eye-state`, 10k × 14). Median per dataset roughly an order of magnitude above CatBoost / XGBoost on the same data.
- Optuna trial failures with `nan` value were occasional and handled by TPE pruning the point — not the cause of any reported result.
- Checkpoint at the `(dataset, model)` level (`results/optuna_models_ckpt.joblib`) made the early stop loss-free: all completed datasets retained, no need to re-run.

## What would change the verdict

- **GPU** — would bring the 100× time gap down enough to argue for inclusion as an ensemble member.
- **Smaller search space tied to dataset size** — capping `d_token` and `n_blocks` for n < 1000 might fix the catastrophic losses; the win pattern on tiny data suggests there is real signal when training does not collapse.
- **Longer training schedule** with proper early stopping inside the trial — current per-trial budget may be too short for transformer convergence on harder datasets.

None of these are pursued in this benchmark; the model is documented here and excluded from the final headline results.
