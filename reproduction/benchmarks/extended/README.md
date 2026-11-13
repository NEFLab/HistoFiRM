# Extended patient-level comparators

These are independent adaptations on the common UNI2-h patient bags and functional target axis. They are not reproductions of each original paper's complete data-processing pipeline.

| Comparator | Output route | Entry point |
|---|---|---|
| MOSBY | Direct programme regression | `run_mosby_adapted.py` |
| Gated ABMIL | Direct programme regression control | `run_abmil_regression.py` |
| TIGER/HistoBistro | Direct programme regression | `run_tiger_adapted.py` |
| RNAPath | Gene expression followed by the fixed functional scorer | `run_gene_models.py --method rnapath` |
| Path2Omics FFPE branch | Gene expression followed by the fixed functional scorer | `run_gene_models.py --method path2omics` |

Source papers/repositories, architecture changes, budgets and numerical settings are recorded in the protocol files or the preparation functions. RNAPath uses the fixed 200-epoch budget in this comparison; it is not claimed to have exhausted the original method's optimisation. Historical job queues and training checkpoints are not distributed.

## Setup

Install the root package and the dependencies listed here. Training requires a compatible CUDA PyTorch installation. The executed patch-bank implementations keep the complete relevant feature bank on the GPU and may require substantial GPU memory. No training is performed by installation or by the CPU checks.

```bash
python -m pip install -e .
python -m pip install -r reproduction/benchmarks/extended/requirements.txt
python reproduction/benchmarks/extended/test_numerical_contracts.py
```

## Shared inputs

Use the root manifest schema and the existing `reproduction/study/` partitions. Each slide feature record requires `features.source_ref` containing `path`, `sha256`, `bytes` and the HDF5 dataset `key`, plus the registered `slide_id` and `n_patches`. Arrays have 1,536 columns. TCGA labels are an NPZ with `patient_ids`, `cancers`, `target_ids` and `truth` on the ordered 1,564-target axis. Source paths inside manifests are relative to the manifest. External labels are not used in cache preparation or training.

```bash
python reproduction/benchmarks/extended/prepare_inputs.py bind --tcga data/tcga_manifest.json --cptac data/cptac_manifest.json --study reproduction/study --output outputs/comparison_inputs
python reproduction/benchmarks/extended/run_mosby_adapted.py prepare --base outputs/comparison_inputs --out outputs/mosby
```

The cache preserves all retained patches subject to the fixed 8,000-per-slide cap. Cap sampling uses the original slide-identity rule. Study data and encoder weights must be acquired from their providers. Original trained weights are not bundled.

## Training and prediction

Repeat each training command for folds 0–4. Each command selects epochs on the inner validation split, refits on the outer training set, and exports the held-out and external predictions. It does not open external RNA labels.

```bash
python reproduction/benchmarks/extended/run_mosby_adapted.py train --base outputs/comparison_inputs --out outputs/mosby --fold 0
python reproduction/benchmarks/extended/prepare_inputs.py method --method abmil --cache outputs/mosby --output outputs/abmil
python reproduction/benchmarks/extended/run_abmil_regression.py train --base outputs/comparison_inputs --out outputs/abmil --fold 0
python reproduction/benchmarks/extended/run_tiger_adapted.py prepare --base outputs/comparison_inputs --cache outputs/mosby --out outputs/tiger
python reproduction/benchmarks/extended/run_tiger_adapted.py train --base outputs/comparison_inputs --cache outputs/mosby --out outputs/tiger --fold 0
```

Prepare the gene branch once. Add `input/genes.json` (`{"ids": [...]}` for the fixed 18,977 genes), `input/patient_ids.json` matching the cache order, and `input/TCGA_GENE_TRUTH.npy` containing decoded, finite log2(TPM+1) values. Run all five folds for both methods; each fold fits the registered cancer types separately.

```bash
python reproduction/benchmarks/extended/prepare_inputs.py method --method gene --cache outputs/mosby --output outputs/gene
python reproduction/benchmarks/extended/run_gene_models.py --root outputs/gene --cache outputs/mosby --method rnapath --fold 0
python reproduction/benchmarks/extended/run_gene_models.py --root outputs/gene --cache outputs/mosby --method path2omics --fold 0
```

## Evaluation and gene-mediated scoring

Create `outputs/comparison_inputs/inputs/evaluation_sources.json` with the schema below for each cohort, and `inputs/features/support_panel.json` mapping each cancer to its frozen evaluation target IDs. These files are evaluation inputs; do not create a panel by filtering the observed prediction results.

```json
{"sources":{"TCGA16":{"TRUTH":{"source":{"path":"tcga_truth.npz","sha256":"<actual SHA-256>"},"keys":{"patient_ids":"patient_ids","cancers":"cancers","target_ids":"target_ids","truth":"truth"}}},"CPTAC8":{"TRUTH":{"source":{"path":"cptac_truth.npz","sha256":"<actual SHA-256>"},"keys":{"patient_ids":"patient_ids","cancers":"cancers","target_ids":"target_ids","truth":"truth"}}}}}
```

```bash
python reproduction/benchmarks/extended/evaluate_extension.py --base outputs/comparison_inputs --out outputs/mosby
python reproduction/benchmarks/extended/evaluate_abmil.py --base outputs/comparison_inputs --out outputs/abmil
python reproduction/benchmarks/extended/evaluate_tiger.py --base outputs/comparison_inputs --out outputs/tiger
```

For gene-mediated scoring, place `FT01_BINDING.json` in `outputs/gene/`. It contains `input_space: "HGNC_ORDERED_LOG2_TPM_PLUS1"`; `genes` and `targets` references to JSON `{"ids": [...]}` files; a `gmt` list of file references with their original `source` prefixes; a `progeny` CSV reference; and `scoring` exactly equal to `scoring_lock.json`. Each reference has `path` and `sha256`. Paths resolve relative to the binding file. The fixed resource universe must yield 3,086 scoreable ordinary gene sets and the registered 1,564 outputs. Gene-set memberships and the weighted PROGENy resource are obtained separately under their providers' terms. The bundled scoring functions preserve raw ssGSEA ES and weighted PROGENy scoring; no second log transform is applied.

```bash
python reproduction/benchmarks/extended/finish_gene_models.py --root outputs/gene --cache outputs/mosby --base outputs/comparison_inputs --method rnapath --workers 12
python reproduction/benchmarks/extended/finish_gene_models.py --root outputs/gene --cache outputs/mosby --base outputs/comparison_inputs --method path2omics --workers 12
```

External gene predictions are averaged across folds before projection at zero and functional scoring. All evaluators retain unavailable targets and use the original fixed-support cancer-balanced metrics. The release replaces private directory bindings and job scheduling with explicit input files; numerical training and scoring definitions are retained. The code release has CPU contract tests, but it has not repeated the full GPU training runs. Exact study-result regeneration requires the matching source arrays, resource versions and runtime, in addition to these scripts.
