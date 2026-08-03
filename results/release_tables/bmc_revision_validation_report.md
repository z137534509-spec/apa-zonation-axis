# BMC revision validation report — 2026-08-03

## Analysis roles

- GSE156931 supplies an independent paired expression validation. Its public matrix contains eight identifiable APA/AAG code pairs. The GEO record states that raw data could not be located, so this analysis uses the deposited processed matrix and makes no preprocessing claim.
- Unpaired GSE156931 samples are retained in the audit table but are not assigned CPA, pheochromocytoma, or hyperplasia diagnoses because the deposited sample metadata does not provide auditable disease labels.
- GSE274314 spatial analyses use patient-paired section summaries for inference. Spot-level scores quantify within-section distribution and spatial autocorrelation, not additional patient replicates.

## GSE156931 paired expression transfer

- Identifiable APA/AAG pairs: 8; paired samples: 16.
- The reference-defined axis is the mean of the aldosterone-oriented ZG and intermediate modules minus the mean of the cortisol-oriented ZF and androgen-oriented ZR modules.
- Bootstrap 95% intervals are descriptive summaries of the paired mean. Primary inference uses an exact two-sided sign-flip test across all possible sign assignments of the observed paired differences; this is not a binomial sign test.
- Primary axis: mean APA-AAG difference 0.6324; descriptive bootstrap 95% CI 0.1240 to 1.1211; 7/8 pairs positive; exact two-sided sign-flip P=0.03906.
- CYP11B2-free sensitivity: mean APA-AAG difference 0.5819; descriptive bootstrap 95% CI 0.1130 to 1.0645; 7/8 pairs positive; exact two-sided sign-flip P=0.07031.

## GSE274314 spatial location and dispersion

- Lesion-level location: mean APA-adjacent difference 0.6100; 7/7 pairs positive; exact two-sided sign-flip P=0.01562.
- CYP11B2-free lesion-level location sensitivity: mean APA-adjacent difference 0.5609; 7/7 pairs positive; exact two-sided sign-flip P=0.01562.
- Raw spot-score dispersion (IQR): mean APA-adjacent difference -0.1239; 2/7 pairs positive; exact two-sided sign-flip P=0.12500.
- Regional dispersion (3×3 block-median IQR): mean APA-adjacent difference -0.0663; 2/7 pairs positive; exact two-sided sign-flip P=0.32812; FDR across seven spatial-heterogeneity endpoints=0.32812.
- Regional dispersion (4×4 block-median IQR): mean APA-adjacent difference -0.1284; 1/7 pairs positive; exact two-sided sign-flip P=0.03125; FDR across seven spatial-heterogeneity endpoints=0.17500.
- Regional dispersion (5×5 block-median IQR): mean APA-adjacent difference -0.1565; 1/7 pairs positive; exact two-sided sign-flip P=0.07812; FDR across seven spatial-heterogeneity endpoints=0.17500.
- Spatial autocorrelation (Moran's I): mean APA-adjacent difference -0.1084; 1/7 pairs positive; exact two-sided sign-flip P=0.17188. 7/7 APA sections exceeded the within-section permutation threshold P<0.05.

## Writing rule

GSE156931 is directionally concordant in seven of eight pairs and reaches the exact paired threshold for the primary axis, but its processed-matrix-only availability and modest sample size mean that it should be interpreted as independent directional support rather than pooled with the other cohorts. The seven spatial-heterogeneity endpoints are the raw spot-score IQR, median absolute deviation, 90th-minus-10th percentile range, block-median IQR at 3×3, 4×4, and 5×5 grids, and Moran's I. Location endpoints are not part of this FDR family. Report regional dispersion only when its direction is stable across grid resolutions, and note that lower lesion-level dispersion can reflect replacement of normally layered ZG-ZF-ZR cortex rather than intrinsic APA homogenization.
