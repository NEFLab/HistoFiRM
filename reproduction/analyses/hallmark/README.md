# Hallmark predictability analyses

This directory implements the complete 50-programme patient-level analyses in Tables S34–S37. Numerical kernels are carried over from the executed analyses. Inputs must contain aligned, fixed predictions; these scripts do not train models or reconstruct predictions from summary tables.

## Environment and checks

```bash
python -m pip install -r reproduction/analyses/hallmark/requirements.txt
python reproduction/analyses/hallmark/test_hallmark_core.py
python reproduction/analyses/hallmark/test_profile_core.py
```

## S34–S36: single-model and composition analyses

Supply `TCGA.npz` and `CPTAC.npz`, each containing:

| Array | Shape / definition |
|---|---|
| `ids`, `cancer` | Patient identity and cancer label, length n; unique IDs within each cohort |
| `targets` | The ordered 50 `HALLMARK::...` programme IDs; identical across cohorts |
| `Y`, `P` | n × 50 finite RNA targets and fixed HistoFiRM predictions |
| `outer` (TCGA) | Original held-out fold, integers 0–4 |
| `X` (TCGA) | n × 2 DNA purity and methylation-derived leukocyte fraction |
| `eligible` (TCGA) | Boolean original composition-analysis support, 5,646 patients |
| `inner` (TCGA) | Original nuisance fold 0–2 within each eligible cancer/outer-fold stratum; −1 otherwise |

Preserve the original patient order, fold binding and eligibility decisions. Do not infer nuisance folds from the order of a newly sorted table. In the study, the original three-fold nuisance partitions used shuffled KFold with random state `20260926 + outer_fold` within the original ordered eligible stratum. Ineligible strata had fewer than 15 patients. The registration count of 5,660 is not the analysed support of 5,646.

Use the complete S34 and S35 CSV files from the supporting tables as reference estimates:

```bash
python reproduction/analyses/hallmark/prepare_analysis.py hallmark --tcga data/hallmark/TCGA.npz --cptac data/hallmark/CPTAC.npz --s34 data/tables/S34_RAW_HALLMARK.csv --s35 data/tables/S35_COMPOSITION_HALLMARK.csv --output outputs/hallmark
python reproduction/analyses/hallmark/run_hallmark.py --root outputs/hallmark
```

Preparation verifies every raw and matched composition estimate before writing the input gate. The run performs 2,000 patient resamples, retains intervals with at least 1,900 valid replicates, refits the nuisance regressions in each composition resample, and retains paired Δr. TCGA resampling preserves the original strata; CPTAC draws are independent. S36 uses `atanh(r_CPTAC) − atanh(r_TCGA)`. Only this S36 transform clips numerical endpoints by 1e−12, following its fixed protocol.

## S37: cross-method and cross-cohort profiles

Supply a second pair of cohort NPZ files with the same identity, cancer and target arrays. Here `P` is n × 7 × 50 and `models` follows `profile_config.json`. `targets` must be sorted by original ID. TCGA `outer` contains its original folds; CPTAC `outer` is an all-zero vector. All seven predictions share the same patient and target support. `Y` remains n × 50.

```bash
python reproduction/analyses/hallmark/prepare_analysis.py profiles --tcga data/profiles/TCGA.npz --cptac data/profiles/CPTAC.npz --target-reference data/tables/INPUT_HALLMARK_8400.csv --summary-reference data/tables/B1_INTERVALS_38.csv --output outputs/profiles
python reproduction/analyses/hallmark/run_profiles.py --root outputs/profiles
```

The target reference contains all 8,400 cancer–programme–model records; the summary reference supplies the 38 original estimates. Preparation validates both levels. Outputs include 12 within-cancer profile medians, 12 doubly centred cross-method associations, and 14 raw/centred cross-cohort associations. Centre the Fisher-transformed 8 × 50 matrix by both row and column means. For these profile summaries, endpoint Pearson values invalidate the affected statistic rather than being clipped.

Patient resampling is joint across all methods and programmes within a cancer and independent across cohorts. The percentile intervals are conditional on fixed predictions and do not include retraining uncertainty. `point_outside_percentile_interval` is retained as a diagnostic, not corrected by moving the estimate or interval. No new significance tests or programme selection are performed.

## Inputs and interpretation

Private input arrays, patient covariates, trained checkpoints and gene-set resources are not included in this code directory. Obtain each source under its access terms and supply the frozen model outputs. Supporting summary tables alone cannot reproduce patient-bootstrap intervals. The routines are study-specific (16 TCGA cancer types, eight CPTAC cancer types and 50 Hallmark programmes); changed cohorts require a separately specified analysis.
