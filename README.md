# Patient-paired APA composite-score reanalysis

Executable analysis code and compact machine-readable results for the BMC Genomics submission, *Patient-paired transcriptomic analyses identify higher ZG/intermediate-to-ZF/ZR composite scores in aldosterone-producing adenomas*.

## Scope and interpretation boundary

This repository contains analysis code, a pinned software environment, and selected derived result tables. It excludes raw and processed expression data, manuscript files, peer-review material, and identifiable participant information. Public source data must be downloaded from their original repositories.

The primary endpoint is a transparent curated four-module composite: mean(ZG/aldosterone, intermediate steroidogenic) minus mean(ZF/cortisol, ZR/androgen). The patient pair, never an individual spatial spot, is the inferential unit. Exact panel membership, overlaps, and effective composite weights are exported in the release tables.

The panel was manually curated after project initiation and frozen for this release; it was not preregistered. It does not use target-cohort APA-versus-control differential expression, label transfer, or a target-cohort gene-selection threshold. All score-based results should therefore be interpreted as exploratory tissue-level associations, not as evidence of tumor-cell intrinsic reprogramming, histological zonation, lineage, steroid output, or causality.

## Data sources

| Resource | Role | Access |
| --- | --- | --- |
| GSE274314 | Primary paired Visium analysis | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274314 |
| GSE60042 | Independent paired expression cohort; Murakami et al. 2015 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60042 |
| GSE156931 | Processed-matrix paired directional-support cohort; Azizan et al. 2012 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156931 |
| Zenodo 18128100 | Normal-adrenal single-nucleus reference | https://zenodo.org/records/18128100 |
| Zenodo 10560526 | Normal-adrenal spatial reference | https://zenodo.org/records/10560526 |

Place downloaded inputs under `data/raw/` and required derived inputs under `data/processed/`; both locations are ignored by Git.

## Environment

The revision was run on Windows 11 with Python 3.12.10, numpy 2.3.5, pandas 2.3.3, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2, and scikit-learn 1.8.0. Python dependencies are listed in [requirements-python.txt](requirements-python.txt). The repository is MIT licensed and has no access restriction beyond availability of the public source inputs.

## Analysis order

1. Run `scripts/20_stage2_zonation_axis_strengthening.py` to create the retained spatial-score input from GSE274314.
2. Run `scripts/26_external_transcriptome_methylome_validation.py` to prepare the GSE60042 score input. Its GSE60043 methylation routines are legacy and excluded from the BMC analysis.
3. Run `scripts/51_bmc_gse156931_and_spatial_dispersion.py` for the original four-module BMC score, GSE156931 support, exact sign-flip testing, and 3-by-3/4-by-4/5-by-5 spatial-distribution sensitivity.
4. Run `scripts/53_bmc_score_robustness_and_spatial_qc.py` for the executable module specification, effective weights, common-gene, CYP11B2-removal, leave-one-module, leave-one-gene, raw-matrix spatial-QC, and equal-section-standardization sensitivity analyses.

The exact sign-flip test enumerates every possible sign assignment of observed patient-level paired differences. Percentile bootstrap intervals use 10,000 patient-level resamples with fixed seed 20260803. The new script additionally reports exact binomial sign and Wilcoxon sensitivity P values.

## Release tables

`results/release_tables/` includes module specification and weights, platform coverage, cohort-level module and robustness tests, section-level spatial QC and ROI-availability status, spatial-distribution tests, grid occupancy, Moran's I summaries, and the external-cohort paired outputs. No pathology-defined tumor ROI was available in the public GSE274314 archive; transcript-defined ROIs were not used as a primary restriction.

## Versioning and citation

The tagged BMC submission snapshot is archived in Zenodo as [10.5281/zenodo.21787456](https://doi.org/10.5281/zenodo.21787456), corresponding to GitHub release `bmc-genomics-20260804.3`. This README and `.zenodo.json` identify the matching code, release tables, software version, license, and input requirements.
