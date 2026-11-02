# Block E plan (revised 2026-10-31)

Question: are the foundation models' results on the benchmark driven by benchmark data in training
(direct leakage) or by synthetic data generated from it? Post-release data cannot be in a model's
training set, so a gap that holds on post-release data is not explained by direct leakage.

## Models and cutoff
- Run: lgbm, catboost (baselines), tabicl (cutoff 2026-02-12), tabpfn3 (2026-05-05).
- Out: TabPFN-3.5 and 3.5-fast (user decision 2026-10-31); TabFM (CPU-unusable; its 2026-06-29
  cutoff would also cut the pool to 12).
- Pool rule: first public strictly after 2026-05-05 (the latest cutoff in the run).

## Pool (14 candidates; one set for all models)
Built (7): dengue-mexico-deaths, p2p-npl, biotech-y36, cholera-tz-result, malpdf-fs43-binary,
meconium-aspiration, immobility-lbp.
Built 2026-10-31 (all 14 load through benchmark.data; drop lists in datasets/external.csv):
- DRRisk, Dryad jwstqjqrm (2026-09-21, 49,931 rows); use *_original; 2152 duplicates; paper predates data.
- steroid-induced, Dryad ghx3ffc3x (2026-10-05, 540 rows); target case; drop post-outcome columns.
- Aedes sex, Dryad z08kprrx0 (2026-10-14, 559 rows); target Sex.
- child-obesity-beijing, Dryad gf1vhhn44 (2026-07-21, 6493 rows); target obesity; drop birthheight.
- hpv-genotypes, Dryad 5qfttdzkt (2026-05-26, 712 rows); target abnormal Pap smear (69 positive;
  "major" has 17); drop cytology columns. Default chosen here, to confirm.
- equine-ct, Dryad np5hqc09w (2026-09-03, 1749 rows); dedupe the 142 (sheet, case) pairs; target
  PrimSin1DenSin2 (primary vs dental, ~1020 labelled). Default chosen here, to confirm.
- CAP France, Dataverse TJ64VI (2026-06-01, 1028 rows): download, then build. Cutoff check passes.
If TabFM were kept, HPV and CAP would drop (pool 12).

## Provenance strata (to verify per dataset before the analysis)
- Underlying data public before the cutoff: cholera (2015-2018 cases), biotech (ClinicalTrials.gov
  records), MalPDF (features from older corpora), DRRisk (paper 2021), dengue Mexico (surveillance open data).
- Otherwise new: meconium, immobility, Aedes, HPV, equine, steroid, child obesity, CAP (to confirm).
The strata are reported separately: a gap that holds on the "new" stratum is not explained by data
that was already public.

## Analysis (pre-registered for this block)
1. Primary: tabpfn3 vs lgbm and tabpfn3 vs catboost on the pool; PR AUC (weighted, as the benchmark);
   paired Wilcoxon over datasets; Holm over the two pairs; paired bootstrap CI over datasets.
2. Contrast: the same pairs on UCI++ (131 datasets, as already run). If the foundation-model gap is
   much smaller on the post-release pool than on UCI++, the benchmark gap leans on the benchmark data.
3. Strata: the same pairs on the "new" and "old-public" subsets, reported without a test (n too small).
4. Calibration (ECE, Brier) raw p, as in the benchmark.
5. Timing: only in a window with no other compute; otherwise timings are not reported.

## Limits
- n = 14 gives low power; a null result is not evidence of no leakage.
- Subsampling at 10,000 rows applies (dengue).
- Group structure (hospital, clinic) is not handled by the loader: plain stratified CV.

## Run conditions
- Only after project-risk's go that its DART queue has finished. Chain ~/.sdb-runs/block_e.sh.
2026-11-01 20:25:48 review 1 of block E: consolidated in reviews/review1_blocke_consolidated.md
