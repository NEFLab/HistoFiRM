# Data and input requirements

## Feature bags

The study predictor takes complete patient bags of 1,536-dimensional UNI2-h patch embeddings. This release starts at the feature-array interface. Whole-slide tiling, scanner metadata and pretrained encoder acquisition remain separate. Reproduction requires the paper's physical field of view, image transform and encoder weights.

Example manifest schema:

~~~json
{
  "cohort": "COHORT",
  "input_dim": 1536,
  "target_ids": ["TARGET_A", "TARGET_B"],
  "source_ids": ["HALLMARK", "REACTOME"],
  "cancer_names": ["BRCA", "OV"],
  "patients": [{
    "patient_id": "PATIENT_ID",
    "cancer": "BRCA",
    "outer_fold": 0,
    "slides": [{
      "slide_id": "SLIDE_ID",
      "n_patches": 100,
      "features": {"path": "arrays/slide.npy", "sha256": "SHA256_OF_ARRAY"}
    }]
  }],
  "labels": {"path": "labels.npz", "sha256": "SHA256_OF_LABEL_FILE"}
}
~~~

The abbreviated axes illustrate the schema. Actual manifests must contain all ordered axes matching the checkpoint. Arrays may be NPY, NPZ or HDF5; NPZ/HDF5 references also need a key. Optional row_start/row_stop define contiguous slices. Optional coords use the same reference schema with shape (patches, 2).

Label NPZ files contain truth (patients × targets), patient_ids, cancers and target_ids. Their IDs must exactly match the manifest. SHA-256 hashes are checked. Keep consistent patient identifiers across slides and splits; a different filename does not establish independence.

## Functional targets

The panel contains 50 Hallmark, 1,165 Reactome, 188 C6 oncogenic signature, 147 C4:3CA and 14 PROGENy targets. Resource IDs are retained for alignment.

Ordinary patient-level targets use GSEApy 1.3.0 raw ssGSEA enrichment scores with the supplied settings. Duplicate HGNC symbols are summed in linear TPM before log2(TPM+1). Ensembl inputs must first use the frozen mapping resource. PROGENy uses signed weights divided by the sum of absolute mapped weights.

The RNA-only eligibility rule requires sufficient coverage, finite and nondegenerate scores, gene-dropout stability, and eligibility in at least 12 of 16 cancers. target_eligibility applies this rule to computed QC statistics; it does not rerun gene-dropout experiments. Do not reselect the fixed panel using image-prediction performance.

Gene membership and pretrained weights are not redistributed. Obtain resources from their providers and retain version/hash records:

- TCGA/CPTAC: [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/).
- Gene sets: [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/).
- Encoder and weight access: [UNI](https://github.com/mahmoodlab/UNI).
- PROGENy: use the signed footprint table identified in the manuscript and source-data registry.
- Spatial and clinical cohorts: use the specific accessions and source publications in the manuscript and supplementary source table.

## Public-data use

Follow original access conditions and cite originating studies. Public availability does not transfer image, dataset or weight licenses to this code license. Raw slides, expression matrices, patient-level clinical records, access tokens and pretrained encoder weights are not included. This release asserts no new ethics approval or unrestricted redistribution permission.
