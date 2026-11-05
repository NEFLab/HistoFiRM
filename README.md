# HistoFiRM

HistoFiRM predicts patient-level RNA-derived functional scores from whole-slide histology feature bags. It combines a cancer-specific Ridge anchor with learned means and variances of patch embeddings, a low-rank cancer adapter, and training-only Hallmark supervision.

This repository contains the current numerical implementation, portable training and inference commands, and the principal downstream analysis functions. It accompanies *Whole-slide histology captures interpatient variation in transcriptional programmes across human cancers*.

<img width="2600" height="1951" alt="Fig1_20261104_ACCEPTED_NAMES_20261104" src="https://github.com/user-attachments/assets/7d0bf751-76e8-4c18-9eba-748d3601612b" />

## Installation

Python 3.10 or later is required. Install PyTorch for the intended CPU or CUDA environment, then run:

~~~bash
python -m pip install -e ".[scoring,test]"
python -m pytest -q
~~~

CPU checks and the tested package versions are documented in requirements-tested.txt. This describes the release test environment, not the environment of every original experiment.

## Quick check

The example generates synthetic bags and labels. Its outputs are not study results.

~~~bash
python examples/make_demo.py --output outputs/demo
histofirm prepare --manifest outputs/demo/fit.json --manifest outputs/demo/validation.json --output outputs/demo/raw_moments.npz
histofirm train --fit outputs/demo/fit.json --validation outputs/demo/validation.json --cache outputs/demo/raw_moments.npz --config outputs/demo/config.json --fold 0 --output outputs/demo/select
histofirm train --fit outputs/demo/outer.json --selected outputs/demo/select/fit.json --cache outputs/demo/raw_moments.npz --config outputs/demo/config.json --fold 0 --output outputs/demo/refit
histofirm predict --manifest outputs/demo/test.json --checkpoint outputs/demo/refit/model.pt --output outputs/demo/prediction.npz
~~~

Use a new output directory for each training run. Input paths in a patient manifest resolve relative to that manifest; command-line paths resolve relative to the working directory.

## Study workflow

1. Prepare the fixed RNA-derived targets with scoring.py, the prescribed gene-set resources and configs/scoring.json. The ordered 1,564-target list is in configs/target_ids.txt.
2. Supply complete bags of pretrained patch embeddings and patient/slide manifests. Spatial coordinates are not model inputs.
3. Provide patient-level outer folds and disjoint inner fit/validation manifests. configs/model.json contains the study settings. Population transforms and Ridge coefficients are fitted on the training population; inner validation selects penalties and the neural checkpoint.
4. Refit on the outer training population using the selected penalties, epoch count and learning-rate schedule. Neither an outer test nor an external validation patient belongs in either training stage.
5. Predict each held-out fold separately. The merge command combines five nonoverlapping held-out outputs. Its --external option averages the five decoded predictions for the same external patients, without silently dropping missing fold predictions.
6. Apply downstream functions to explicitly aligned IDs, target axes and covariates. See [analysis interfaces](docs/analysis.md).

The [study reproduction directory](reproduction/README.md) provides the fixed TCGA partitions, CPTAC evaluation register, numerical settings and adapted comparator entry points. A real study run also requires feature arrays, RNA/clinical inputs and the specified resource versions, obtained under their providers' terms. These inputs and original trained checkpoints are not bundled. The synthetic example does not reproduce the paper's numerical results. See [data requirements](docs/data.md).

## Code map

| Component | Modules |
|---|---|
| RNA scores and fixed eligibility criteria | scoring |
| Ridge anchor and moment regression | linear, core, model |
| Complete-bag two-pass gradients | stream |
| Data, selection, refitting and prediction | data, prepare, training, prediction |
| Fixed-support evaluation and cancer-balanced summaries | evaluation |
| Composition-conditioned association | composition |
| Within-patient pathology associations | pathology |
| Spatial proxies and hierarchical summaries | spatial |
| Exploratory survival model and post-fit analyses | survival, clinical_training, clinical_statistics, clinical |

## Interpretation

Local first-order responses and bag-conditional heterogeneity contributions reconstruct a patient prediction when averaged over the complete bag. They are not attention weights or direct measurements of local RNA. Spatial RNA proxies and patient-level targets use different scoring protocols. The clinical extension is exploratory; association and risk stratification do not establish incremental clinical utility.

## License

Project code uses the MIT License. Dataset access, gene-set resources and pretrained encoder weights remain governed by their respective providers' terms. See [NOTICE](NOTICE).
