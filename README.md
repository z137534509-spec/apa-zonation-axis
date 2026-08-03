# Patient-paired cortical-program shift in APA

Reproducibility code and compact result tables for the BMC Genomics submission, *Patient-paired transcriptomic analyses reveal a shift toward zona glomerulosa/intermediate programs in aldosterone-producing adenomas*.

## Scope

This repository contains executable analysis code, a pinned software environment, and selected machine-readable patient-level results. It deliberately excludes raw or processed expression data, manuscript files, peer-review material, and participant-level information. All analysed datasets are public and must be obtained from their original repositories.

The main outcome is a fixed, reference-defined patient-paired cortical-program contrast: the mean of aldosterone-oriented zona glomerulosa (ZG) and intermediate steroidogenic module scores minus the mean of cortisol-oriented zona fasciculata (ZF) and androgen-oriented zona reticularis (ZR) module scores. A patient pair, not an individual spatial spot, is the inferential unit. Module genes are not selected from APA-versus-control differential-expression results in any target cohort, and no transferred cell-type labels are used for the primary score.

The current BMC revision also evaluates spatial score distribution at 3-by-3, 4-by-4, and 5-by-5 within-section rank grids. These regional-dispersion results are explicitly secondary sensitivity analyses: their direction was concordant, but no spatial-heterogeneity endpoint remained significant after false-discovery-rate control.

## Data sources

| Resource | Role | Access |
| --- | --- | --- |
| GSE274314 | Primary paired Visium analysis | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274314 |
| GSE60042 | Independent paired expression cohort; Murakami et al. 2015 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60042 |
| GSE156931 | Processed-matrix paired external support; Azizan et al. 2012 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156931 |
| Zenodo 18128100 | Normal-adrenal single-nucleus reference | https://zenodo.org/records/18128100 |
| Zenodo 10560526 | Normal-adrenal spatial reference | https://zenodo.org/records/10560526 |

Place downloaded inputs under `data/raw/` and required derived inputs under `data/processed/`; both locations are ignored by Git.

Legacy exploratory scripts that mention GSE60043 or GSE64957 are retained for provenance only. Those datasets and their outputs are not part of the BMC Genomics analysis, release tables, or manuscript claims.

## Environment

The BMC revision was run with Python 3.12.10, numpy 2.3.5, pandas 2.3.3, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2, and scikit-learn 1.8.0. Exact Python dependencies are listed in [requirements-python.txt](requirements-python.txt). Legacy R reference-transfer code and its requirements are retained separately and are not used for the BMC primary analysis.

## Analysis order

1. Run `scripts/20_stage2_zonation_axis_strengthening.py` to create the retained spatial score input from GSE274314.
2. Run `scripts/26_external_transcriptome_methylome_validation.py` to prepare the GSE60042 score input. Its GSE60043 methylation routines are legacy and are excluded from the BMC analysis.
3. Run `scripts/51_bmc_gse156931_and_spatial_dispersion.py` for the four-module BMC score, GSE156931 external support, exact sign-flip testing, and 3-by-3/4-by-4/5-by-5 spatial-dispersion sensitivity.

The BMC script reads the outputs of steps 1 and 2. It reports descriptive paired bootstrap intervals and uses an exact two-sided sign-flip test that enumerates all possible sign assignments of the observed paired differences; it is not a binomial sign test.

## Release tables

`results/release_tables/` includes the patient-level paired tests, section summaries, grid occupancy counts, Moran's I summaries, GSE156931 gene coverage and paired deltas, the GSE60042 BMC-axis test, and a short validation report. These compact tables support inspection; they do not replace rerunning the analyses from public inputs.

## Versioning and citation

Use the GitHub Release matching the manuscript submission for a versioned code snapshot. A `.zenodo.json` metadata file is included to prepare archival; the repository still requires a Zenodo archive DOI before formal journal submission.
