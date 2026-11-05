# HistoFiRM study reproduction materials

This supplement to the HistoFiRM code repository contains the fixed study partitions, numerical settings and the adapted comparator implementations used in the manuscript. It does not contain trained checkpoints, pretrained encoder weights, raw slides, RNA matrices, or individual clinical records.

## Contents

- study/: 6,781 TCGA patient identifiers, five outer folds and exact inner-fit/validation membership; 1,128 CPTAC evaluation record identifiers; ordered cancer and 1,564-target axes. These are deidentified source identifiers. CPTAC records are not a claim of 1,128 independent patients.
- configs/: the HistoFiRM and functional-scoring settings.
- scripts/: bind a supplied feature/label manifest to the fixed partitions, and verify separately downloaded DeepPT author code.
- benchmarks/: the adapted Ridge/MLP, DeepPT, SEQUOIA and HE2RNA numerical implementations and training/inference entry points.
- weights/: reserved for locally trained checkpoints; no original weights are supplied.

This directory complements the main HistoFiRM package in the repository root at https://github.com/NEFLab/HistoFiRM. Install the main package before running the commands below. Fixed study materials and adapted comparator implementations are kept here so that their scope and input requirements remain explicit.

## HistoFiRM workflow

Install the main package and obtain authorized feature and RNA inputs. The complete patient manifest must follow the repository's docs/data.md schema. Its patient membership, slide membership and patch counts must match this study register. Label arrays require truth, patient_ids, cancers and target_ids. Do not use invented values to fill unavailable targets.

From this directory:

~~~bash
python scripts/build_fixed_manifests.py --manifest inputs/tcga_manifest.json --study study --output outputs/bound
histofirm prepare --manifest outputs/bound/morphology.json --output outputs/raw_moments.npz
histofirm train --fit outputs/bound/fold_0/fit.json --validation outputs/bound/fold_0/validation.json --cache outputs/raw_moments.npz --config configs/model.json --fold 0 --output outputs/fold_0/select
histofirm train --fit outputs/bound/fold_0/outer_train.json --selected outputs/fold_0/select/fit.json --cache outputs/raw_moments.npz --config configs/model.json --fold 0 --output outputs/fold_0/refit
histofirm predict --manifest outputs/bound/fold_0/test.json --checkpoint outputs/fold_0/refit/model.pt --output outputs/fold_0/test.npz
~~~

Repeat the selection/refit/prediction steps for folds 1–4. Use the repository merge command for five nonoverlapping held-out outputs; use its --external option only for the same externally evaluated records predicted by all five refitted models.

The files supplied here fix membership and settings; they are not precomputed embeddings or scored RNA inputs. Original feature weights and historical environments cannot be inferred from a path name alone. The manuscript describes the remaining provenance limitations.

## Comparators

The archived numerical implementations have distinct input schemas. Their JSON jobs must explicitly bind the appropriate features or tokens, labels, source hashes, patient order, outer fold and inner partitions. Do not pass the HistoFiRM bag manifest to a comparator without preparing that comparator's schema.

Run the original GPU training drivers on Linux with a compatible PyTorch/CUDA environment. Dependencies are listed in benchmarks/requirements.txt; this is a dependency list, not a newly validated historical environment lock.

| Comparison | Training entry | Inputs |
|---|---|---|
| Shared Ridge and MLP, direct or gene route | benchmarks/reference/run.py train | Patient mean features, ordered labels, metadata and fixed outer folds |
| Adapted DeepPT gene route | benchmarks/deeppt/code/train_deeppt.py | Complete patch bags, gene labels and a cancer-specific training job |
| Adapted SEQUOIA direct or gene route | benchmarks/sequoia/code/train_sequoia.py | Per-slide 100-token bags and original patch-count patient pooling |
| Adapted HE2RNA gene route | benchmarks/he2rna/code/train_he2rna.py | Spatial supertiles, explicit valid masks, original patch counts and gene labels |

Example command structure:

~~~bash
python benchmarks/reference/run.py train --dataset inputs/reference_dataset.json --method RIDGE_DIRECT_SHARED --release benchmarks/reference/config/model.json --fold 0 --out outputs/ridge/fold_0
python benchmarks/deeppt/code/train_deeppt.py --job inputs/deeppt_job.json --protocol benchmarks/deeppt/config/PROTOCOL.json --out outputs/deeppt --device cuda:0
python benchmarks/sequoia/code/train_sequoia.py --job inputs/sequoia_job.json --protocol benchmarks/sequoia/config/PROTOCOL.json --out outputs/sequoia --device cuda:0
python benchmarks/he2rna/code/train_he2rna.py --job inputs/he2rna_job.json --protocol benchmarks/he2rna/config/PROTOCOL.json --out outputs/he2rna --device cuda:0
~~~

DeepPT needs the two exact upstream author files obtained from https://zenodo.org/records/11125591. After downloading under the author's terms, run:

~~~bash
python scripts/install_deeppt_source.py --source downloaded/DeepPT
~~~

The package verifies the two hashes before installing them. SEQUOIA's pinned MIT core is included with its license. HE2RNA's adapted component retains the GPL-3.0-or-later license; see NOTICE.txt. No third-party task weights are bundled.

Reference gene-route scoring uses m1ref/functional_scores.py and config/FT01_SCORING_LOCK.json. The extracted function bodies are unchanged from the verified scoring source. Obtain original gene sets, signed PROGENy footprints, mapping versions and the ordered gene axis as documented in the manuscript. Five decoded gene predictions are averaged before nonnegative projection and functional scoring.

Comparator training has not been rerun for this distribution. Packaging checks do not reproduce paper metrics, and a successful --help invocation is not a completed experiment.

## Data access

Obtain TCGA and CPTAC data through their NCI repositories, PanopTILs through the cited dataset, GSE210616 through GEO, the ovarian FFPE dataset through 10x Genomics, and the HCC author-processed spatial data through the Wu et al. publication. The MSKCC clinical source is registered at Synapse project syn25946117 (table syn30236913); follow the actual access conditions there.

Acquire MSigDB gene sets and UNI2-h weights directly from their providers. UNI2-h terms restrict commercial use of models trained on its outputs; the MIT code license does not override these terms.

## Scope

The main package and this add-on supply numerical code, identifiers and settings. They are not a self-contained distribution of all study data. Original checkpoints are intentionally not released. Figures and aggregate supplementary results accompany the manuscript separately. Do not describe this bundle as an end-to-end numerical replication of the full study.
