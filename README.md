# APA zonation-axis spatial transcriptomics

Reproducibility code for the study "Patient-paired spatial transcriptomics reveals a shared aldosterone-oriented shift along the adrenal zonation axis in aldosterone-producing adenomas".

## Scope

This repository contains analysis code and selected machine-readable result tables supporting the manuscript. It does not contain raw or processed expression data, manuscript files, peer-review material, or participant-level information. All analysed datasets were previously public and should be obtained from their original repositories.

The primary inference compares seven patient-matched aldosterone-producing adenoma (APA) and adjacent-cortex spatial sections in GSE274314. Each patient pair is one inferential unit; spatial spots construct section-level estimates and null distributions only. Gene modules are fixed before the APA-versus-adjacent comparison. No genes were selected or weighted using APA-adjacent labels from GSE274314 or GSE60042.

## Data sources

| Resource | Role | Access |
| --- | --- | --- |
| GSE274314 | Primary paired Visium analysis | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274314 |
| GSE60042 | External paired expression transfer | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60042 |
| GSE60043 | Supplementary promoter-methylation context | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60043 |
| GSE64957 | Normal ZG/ZF calibration | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64957 |
| Zenodo 18128100 | Normal-adrenal single-nucleus reference | https://zenodo.org/records/18128100 |
| Zenodo 10560526 | Normal-adrenal spatial reference | https://zenodo.org/records/10560526 |

Place source files in data/raw/ and derived inputs in data/processed/. These directories are intentionally ignored by Git.

## Environment

The main Python analyses were run with Python 3.12.10 and the packages pinned in requirements-python.txt. The R reference-transfer workflow used R 4.5.1 and the package versions listed in requirements-r.txt. scripts/00_install_spatial_deconv_packages.R installs R dependencies used for spatial deconvolution checks.

## Analysis order

1. Obtain the public datasets above and place them under data/raw/. Generate required processed expression and sample-metadata inputs under data/processed/.
2. Run scripts/20_primary_zonation_axis_analysis.py for the reference-anchored spatial axis and random-gene-set controls.
3. Run scripts/25_primary_inference_and_robustness.py for patient-level exact testing, percentile bootstrap (10,000 resamples; seed 20260602), spatial null models and sensitivity analyses.
4. Run scripts/26_external_expression_and_methylation_context.py to transfer the frozen score to GSE60042 and generate supplementary methylation context.
5. For reference-transfer sensitivity, run scripts/00_install_spatial_deconv_packages.R and then scripts/10_* through scripts/14_*.
6. Scripts 27_*, 28_*, 29_* and 31_* reproduce supplementary exploratory or sensitivity analyses only. They are not used to establish the primary patient-paired claim.

Scripts write generated files beneath results/. The results/release_tables/ directory contains selected final numeric tables for quick inspection; it is not a substitute for rerunning the analyses from public source data.

## Key outputs

- results/release_tables/primary_spatial_paired_statistics.csv
- results/release_tables/primary_spatial_null_summary.csv
- results/release_tables/normal_zg_zf_calibration_tests.csv
- results/release_tables/external_expression_paired_tests.csv

## Release and citation

This public release is available at https://github.com/z137534509-spec/apa-zonation-axis. The copyright holders may add an explicit software licence before a future release. A Zenodo archive DOI can be added after the repository is versioned and archived.
