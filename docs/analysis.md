# Analysis interfaces

Functions operate on aligned arrays and tables. Retain complete target axes, unavailable entries, original model identity and evaluation unit. They do not recreate identifiers or select examples.

## Prediction and local readouts

training.fit performs inner selection or fixed-epoch refitting. Ridge-only epoch zero is an eligible checkpoint. Validation combines cancer-balanced target Pearson with the fixed MSE bound.

prediction.local_readout returns first-order response and heterogeneity arrays in decoded target-score units. Their sum averaged over every patch reconstructs the patient prediction. A display crop is not a new bag.

spatial.first_order_prediction returns the frozen first-order field in training-standardized units. It omits the bag-conditional variance term. Do not call it attention or interpolate it into unmeasured locations.

## Composition and pathology

composition.evaluate(y, prediction, covariates, outer_fold, target_ids) is called separately per cancer. Covariates are DNA-derived purity and methylation-derived leukocyte fraction. Separate nuisance regressions for RNA and predictions are cross-fitted within each original held-out fold. Raw, composition-only and conditional correlations share target-specific support. Approximate intervals exclude upstream retraining uncertainty.

pathology.mask_metrics requires original region/nuclear codes and denominators. Nuclear pixel proportions are not cell counts. pathology.associations requires aligned ROI responses, patient IDs, fixed target–annotation pairs and the minimum ROI count. The adjusted immune analysis ranks the two responses and uses the original unranked tumor-region proportion as a nuisance covariate. Different eligible-patient sets must not be subtracted as paired effects.

## Spatial correspondence

spatial.score computes the assay-supported rank-score proxy from gene × spatial-unit counts. It does not replace weighted PROGENy with an unsigned score; those targets remain unavailable.

spatial.evaluate uses finite pairs and at least 10 units. hierarchical_summary preserves sections within donors, donors within studies, and studies within cancers. The adopted HCC tumor subset has eight regions under five donor IDs. In the OV scale comparison, RNA counts are aggregated before rescoring and predictions are averaged within the same regions; the protocols are not independent patients.

## Exploratory clinical extension

survival.Prep implements training-reference preprocessing/PCA; survival.Net implements the additive C/H/F risk model. clinical_training.fit_fixed uses predetermined hyperparameters and epochs. Current CHF settings are in configs/clinical.json. Risk probability is evaluated at 36 months; survival follow-up is administratively truncated at 60 months.

survival provides KM, log-rank and concordance routines. clinical_statistics contains post-fit Cox, PH score diagnostics, HC3 functional effects and paired bootstrap. Functional effects retain the original C6 clinical design, not the later numeric-stage/residual design.

clinical.matched_cox fits grouped and continuous CHF with and without residual disease on exactly the same residual-available patients. Age and risk use full-TCGA standard deviations. Numeric stage is I–IV; missing stage uses the original stage-III placeholder plus missingness indicator. MSKCC TRUE in the mapped source field means complete gross resection, hence no macroscopic residual disease.

correct_risk_family requires both cohorts and both CHF forms, preserving four tests per adjustment family. Moving a plot does not change its test family.

Endpoint origins follow each source. Administrative truncation does not make diagnosis, surgery and imaging dates equivalent.

## Complete Hallmark and cross-method profiles

The [Hallmark study entry points](../reproduction/analyses/hallmark/README.md) extend the original composition functions with patient-resampling intervals. Composition nuisance models are refitted in each replicate. Programme axes and matched support are fixed; patient resampling is shared across programmes and, for cross-method analyses, across models. The cross-cohort profile contrast uses independent cohort draws. These intervals exclude retraining uncertainty and are not simultaneous confidence bands. Original percentile-interval offset flags remain in the complete output.

## Reproduction boundary

The core package and [adapted comparator entry points](../reproduction/README.md) are included. The [extended comparators](../reproduction/benchmarks/extended/README.md) add five patient-level configurations. Historical job queues, complete publication-layout scripts, source patient arrays and original trained checkpoints are excluded. Study figures and complete numerical comparisons accompany the manuscript. Exact numerical reconstruction also requires the matching inputs, fixed partitions, resource versions and runtime. The release tests do not constitute a repeat of all study training runs.
