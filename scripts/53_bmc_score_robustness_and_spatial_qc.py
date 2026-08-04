"""Robustness analyses for the BMC Genomics submission revision.

This script is deliberately self-contained for the four-module score.  It
does not use any APA-versus-control differential-expression result to choose
genes.  The ordered panel below is a transparent literature- and
reference-informed *curated* panel, rather than a preregistered gene set.
Consequently, all inferential results are reported as exploratory,
patient-paired evidence.  The script writes the exact membership, overlaps,
effective weights, coverage, and all sensitivity outputs used in the revised
submission package.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import itertools
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.io import mmread


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results" / "bmc_revision_20260803"
FIGURES = RESULTS / "figures"
SPATIAL_SPOTS = PROJECT / "results" / "stage2_zonation_axis" / "GSE274314_stage2_spot_scores.csv"
SPATIAL_RAW = PROJECT / "data" / "raw" / "GSE274314" / "GSE274314_RAW.tar"
GSE600_EXPR = PROCESSED / "GSE60042_gene_expression_mean_collapsed.csv"
GSE600_META = PROCESSED / "GSE60042_sample_metadata.csv"
GSE156_EXPR = PROCESSED / "GSE156931_gene_expression_mean_collapsed.csv"
GSE156_META = PROCESSED / "GSE156931_sample_metadata_audited.csv"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20260803
N_BOOT = 10_000
MIN_MODULE_COVERAGE = 0.80
QC_MIN_UMI = 500
QC_MIN_DETECTED = 250
QC_MAX_MITO_FRACTION = 0.20

# The four primary modules are copied here (rather than imported from a prior
# analysis script) so this file is the executable specification for this
# revision.  Genes are unique within modules.  Membership may overlap across
# biologically related modules; overlap is retained and its summed weight is
# written to BMC_module_definitions_and_effective_weights.csv.
CORE_MODULES: dict[str, list[str]] = {
    "ZG_aldosterone_program": [
        "CYP11B2", "HSD3B2", "AGTR1", "VSNL1", "DACH1", "LGR5", "ANO4",
        "PDE2A", "NR4A1", "NR4A2", "CACNA1D", "CACNA1H", "ATP2B3",
    ],
    "intermediate_steroidogenic_program": [
        "CYP11A1", "STAR", "HSD3B2", "CYP21A2", "MC2R", "FDX1", "FDXR",
        "SCARB1", "TSPO", "NR5A1",
    ],
    "ZF_cortisol_program": [
        "CYP11B1", "CYP17A1", "MC2R", "SULT2A1", "CYB5A", "PAPSS2", "AKR1B1", "PDE8B",
    ],
    "ZR_androgen_program": [
        "CYP17A1", "CYB5A", "SULT2A1", "SULT2B1", "PAPSS2", "AKR1C3", "HSD17B2", "HSD17B6",
    ],
}
POSITIVE_MODULES = ("ZG_aldosterone_program", "intermediate_steroidogenic_program")
NEGATIVE_MODULES = ("ZF_cortisol_program", "ZR_androgen_program")
CORE_GENE_UNIVERSE = tuple(sorted(set().union(*map(set, CORE_MODULES.values()))))
# ANO4 is the only primary-panel gene absent from GPL6883.  Removing it from
# every cohort produces an identical cross-platform gene universe.
COMMON_PLATFORM_MODULES = {
    name: [gene for gene in genes if gene != "ANO4"]
    for name, genes in CORE_MODULES.items()
}


def analysis_rng(label: str) -> np.random.Generator:
    """Make each bootstrap result independent of call order and reproducible."""
    digest = hashlib.sha256(f"{SEED}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.astype(float).copy()
    for column in out.columns:
        values = out[column]
        sd = values.std(ddof=0)
        out[column] = (values - values.mean()) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return out


def equal_section_zscores(frame: pd.DataFrame, section: pd.Series) -> pd.DataFrame:
    """Gene-wise z scores whose location and scale weight every section equally."""
    out = pd.DataFrame(index=frame.index)
    groups = section.astype(str)
    for column in frame.columns:
        values = frame[column].astype(float)
        means = values.groupby(groups).mean()
        variances = values.groupby(groups).var(ddof=0)
        mean = float(means.mean())
        second_moment = float((variances + means.pow(2)).mean())
        sd = float(np.sqrt(max(second_moment - mean**2, 0.0)))
        out[column] = (values - mean) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return out


def score_modules(z: pd.DataFrame, modules: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate module means and record platform-specific requested/used genes."""
    scores = pd.DataFrame(index=z.index)
    rows = []
    for module, genes in modules.items():
        requested = [gene.upper() for gene in genes]
        available = [gene for gene in requested if gene in z.columns]
        missing = [gene for gene in requested if gene not in z.columns]
        scores[module] = z[available].mean(axis=1) if available else np.nan
        rows.append(
            {
                "module": module,
                "n_requested": len(requested),
                "n_available": len(available),
                "coverage_fraction": len(available) / len(requested),
                "minimum_coverage_requirement": MIN_MODULE_COVERAGE,
                "meets_minimum_coverage": len(available) / len(requested) >= MIN_MODULE_COVERAGE,
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
            }
        )
    return scores, pd.DataFrame(rows)


def composite_from_modules(module_scores: pd.DataFrame) -> pd.Series:
    positive = module_scores[list(POSITIVE_MODULES)].mean(axis=1)
    negative = module_scores[list(NEGATIVE_MODULES)].mean(axis=1)
    return positive - negative


def make_modules_without_gene(gene: str) -> dict[str, list[str]]:
    return {name: [member for member in members if member != gene] for name, members in CORE_MODULES.items()}


def primary_effective_weights(modules: dict[str, list[str]], dataset: str, variant: str, available_columns: set[str]) -> pd.DataFrame:
    rows = []
    for module, genes in modules.items():
        present = [gene for gene in genes if gene in available_columns]
        module_weight = 1 / len(present) if present else np.nan
        composite_sign = 1 / len(POSITIVE_MODULES) if module in POSITIVE_MODULES else -1 / len(NEGATIVE_MODULES)
        for gene in present:
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "module": module,
                    "gene": gene,
                    "module_mean_weight": module_weight,
                    "primary_composite_membership_weight": composite_sign * module_weight,
                }
            )
    membership = pd.DataFrame(rows)
    if membership.empty:
        return membership
    combined = (
        membership.groupby(["dataset", "variant", "gene"], as_index=False)["primary_composite_membership_weight"]
        .sum()
        .rename(columns={"primary_composite_membership_weight": "primary_composite_effective_weight"})
    )
    return membership.merge(combined, on=["dataset", "variant", "gene"], how="left")


def bootstrap_percentile_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan
    draws = analysis_rng(label).choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    observed = abs(float(values.mean()))
    null = np.fromiter(
        ((values * np.asarray(signs)).mean() for signs in itertools.product((-1, 1), repeat=values.size)),
        dtype=float,
    )
    return float(np.mean(np.abs(null) >= observed - 1e-12))


def paired_test_row(
    values: pd.Series,
    dataset: str,
    analysis: str,
    metric: str,
    omitted_feature: str = "",
    standardized_units: str = "cohort-specific standardized units",
) -> dict[str, object]:
    delta = values.dropna().astype(float)
    ci_low, ci_high = bootstrap_percentile_ci(delta.to_numpy(), f"{dataset}:{analysis}:{metric}:{omitted_feature}")
    nonzero = delta[delta != 0]
    try:
        wilcoxon_p = float(stats.wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue) if len(nonzero) else np.nan
    except ValueError:
        wilcoxon_p = np.nan
    sign_p = float(stats.binomtest(int((delta > 0).sum()), len(delta), 0.5).pvalue) if len(delta) else np.nan
    return {
        "dataset": dataset,
        "analysis": analysis,
        "metric": metric,
        "omitted_feature": omitted_feature or pd.NA,
        "n_pairs": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "bootstrap_method": "percentile bootstrap of the patient-paired mean",
        "bootstrap_resamples": N_BOOT,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "n_positive": int((delta > 0).sum()),
        "n_negative": int((delta < 0).sum()),
        "exact_sign_flip_two_sided_p": exact_sign_flip_p(delta.to_numpy()),
        "exact_binomial_sign_p": sign_p,
        "exact_wilcoxon_two_sided_p": wilcoxon_p,
        "effect_unit": standardized_units,
        "interpretation": "exploratory patient-paired evidence; no cross-cohort pooling",
    }


def score_and_test(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset: str,
    case_col: str,
    tissue_col: str,
    control_label: str,
    analysis: str,
    standardization: str = "global",
    section_col: str | None = None,
    modules: dict[str, list[str]] | None = None,
    gene_robustness: bool = False,
    module_robustness: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sample scores, paired tests, coverage, and effective gene weights."""
    modules = modules or CORE_MODULES
    expression = expression.copy()
    expression.columns = expression.columns.astype(str).str.upper()
    z = equal_section_zscores(expression, metadata[section_col]) if standardization == "equal_section" else zscore_columns(expression)
    module_scores, coverage = score_modules(z, modules)
    score_frame = metadata.reset_index(drop=True).copy()
    for column in module_scores:
        score_frame[column] = module_scores[column].to_numpy()
    score_frame["ZG_intermediate_vs_ZF_ZR_composite"] = composite_from_modules(module_scores).to_numpy()
    rows = []
    metric_columns = [*modules.keys(), "ZG_intermediate_vs_ZF_ZR_composite"]
    for metric in metric_columns:
        wide = score_frame.pivot_table(index=case_col, columns=tissue_col, values=metric, aggfunc="mean")
        if {"APA", control_label}.issubset(wide.columns):
            rows.append(paired_test_row(wide["APA"] - wide[control_label], dataset, analysis, metric))

    if gene_robustness:
        for gene in sorted(set().union(*map(set, modules.values()))):
            reduced_scores, _ = score_modules(z, make_modules_without_gene(gene))
            composite = composite_from_modules(reduced_scores)
            temp = score_frame[[case_col, tissue_col]].copy()
            temp["value"] = composite.to_numpy()
            wide = temp.pivot_table(index=case_col, columns=tissue_col, values="value", aggfunc="mean")
            rows.append(paired_test_row(wide["APA"] - wide[control_label], dataset, "leave_one_gene_out", "ZG_intermediate_vs_ZF_ZR_composite", gene))

    if module_robustness:
        for module in CORE_MODULES:
            retained = {name: values for name, values in modules.items() if name != module}
            reduced_scores, _ = score_modules(z, retained)
            positive = [name for name in POSITIVE_MODULES if name in retained]
            negative = [name for name in NEGATIVE_MODULES if name in retained]
            composite = reduced_scores[positive].mean(axis=1) - reduced_scores[negative].mean(axis=1)
            temp = score_frame[[case_col, tissue_col]].copy()
            temp["value"] = composite.to_numpy()
            wide = temp.pivot_table(index=case_col, columns=tissue_col, values="value", aggfunc="mean")
            rows.append(paired_test_row(wide["APA"] - wide[control_label], dataset, "leave_one_module_out", "ZG_intermediate_vs_ZF_ZR_composite", module))

    coverage.insert(0, "dataset", dataset)
    coverage.insert(1, "analysis", analysis)
    weights = primary_effective_weights(modules, dataset, analysis, set(z.columns))
    return score_frame, pd.DataFrame(rows), coverage, weights


def read_tar_bytes(tar: tarfile.TarFile, member: str) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member)
    return gzip.decompress(handle.read())


def feature_symbols(raw: bytes) -> np.ndarray:
    features = pd.read_csv(io.BytesIO(raw), sep="\t", header=None)
    if features.shape[1] < 2:
        raise ValueError("Expected a feature table with a gene-symbol column")
    return features.iloc[:, 1].astype(str).str.upper().to_numpy()


def spatial_mitochondrial_qc(spots: pd.DataFrame) -> pd.DataFrame:
    """Recover mitochondrial fraction from the raw Visium matrices for QC reporting."""
    rows = []
    with tarfile.open(SPATIAL_RAW, "r") as tar:
        names = tar.getnames()
        prefixes = sorted({name.removesuffix("_matrix.mtx.gz") for name in names if name.endswith("_matrix.mtx.gz")})
        for prefix in prefixes:
            gsm = prefix.split("_", 1)[0]
            matrix = mmread(io.BytesIO(read_tar_bytes(tar, f"{prefix}_matrix.mtx.gz"))).tocsc()
            genes = feature_symbols(read_tar_bytes(tar, f"{prefix}_features.tsv.gz"))
            barcodes = np.asarray(read_tar_bytes(tar, f"{prefix}_barcodes.tsv.gz").decode("utf-8", errors="replace").splitlines())
            mito_rows = np.flatnonzero(np.char.startswith(genes.astype(str), "MT-"))
            total = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
            mito = np.asarray(matrix[mito_rows].sum(axis=0)).ravel().astype(float) if mito_rows.size else np.zeros(matrix.shape[1])
            detected = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(float)
            for barcode, total_umi, mito_umi, n_detected in zip(barcodes, total, mito, detected, strict=True):
                rows.append(
                    {
                        "geo_accession": gsm,
                        "barcode": barcode,
                        "raw_total_umi": total_umi,
                        "raw_n_detected_genes": n_detected,
                        "mitochondrial_umi_fraction": mito_umi / total_umi if total_umi > 0 else np.nan,
                        "n_mitochondrial_features": int(mito_rows.size),
                    }
                )
    qc = pd.DataFrame(rows)
    in_tissue = spots[["geo_accession", "barcode", "patient", "tissue", "total_umi", "n_detected_genes"]].copy()
    qc = in_tissue.merge(qc, on=["geo_accession", "barcode"], how="left", validate="one_to_one")
    if qc["mitochondrial_umi_fraction"].isna().any():
        raise RuntimeError("Could not calculate mitochondrial QC for every retained in-tissue spot")
    qc["passes_qc_500umi_250genes_mt20pct"] = (
        (qc["total_umi"] >= QC_MIN_UMI)
        & (qc["n_detected_genes"] >= QC_MIN_DETECTED)
        & (qc["mitochondrial_umi_fraction"] <= QC_MAX_MITO_FRACTION)
    )
    return qc


def spatial_qc_by_section(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["total_umi", "n_detected_genes", "mitochondrial_umi_fraction"]
    for (gsm, patient, tissue), section in qc.groupby(["geo_accession", "patient", "tissue"], sort=True):
        row: dict[str, object] = {
            "geo_accession": gsm,
            "patient": patient,
            "tissue": tissue,
            "n_in_tissue_spots": int(len(section)),
            "n_qc_passing_spots": int(section["passes_qc_500umi_250genes_mt20pct"].sum()),
            "qc_retained_fraction": float(section["passes_qc_500umi_250genes_mt20pct"].mean()),
            "qc_rule": f"total UMI >= {QC_MIN_UMI}; detected genes >= {QC_MIN_DETECTED}; mitochondrial fraction <= {QC_MAX_MITO_FRACTION:.0%}",
        }
        for metric in metrics:
            row[f"{metric}_min"] = float(section[metric].min())
            row[f"{metric}_median"] = float(section[metric].median())
            row[f"{metric}_p95"] = float(section[metric].quantile(0.95))
            row[f"{metric}_max"] = float(section[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def make_module_figure(tests: pd.DataFrame) -> None:
    """Plot the module-level primary analyses; scales are intentionally not pooled."""
    module_order = [
        "ZG_aldosterone_program", "intermediate_steroidogenic_program", "ZF_cortisol_program",
        "ZR_androgen_program", "ZG_intermediate_vs_ZF_ZR_composite",
    ]
    labels = {
        "ZG_aldosterone_program": "ZG / aldosterone",
        "intermediate_steroidogenic_program": "Intermediate",
        "ZF_cortisol_program": "ZF / cortisol",
        "ZR_androgen_program": "ZR / androgen",
        "ZG_intermediate_vs_ZF_ZR_composite": "Composite",
    }
    plot = tests.loc[
        tests["analysis"].eq("primary") & tests["metric"].isin(module_order),
        ["dataset", "metric", "mean_delta", "bootstrap_ci95_low", "bootstrap_ci95_high"],
    ].copy()
    plot["metric"] = pd.Categorical(plot["metric"], categories=module_order, ordered=True)
    plot = plot.sort_values("metric")
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["GSE274314", "GSE60042", "GSE156931"], strict=True):
        frame = plot.loc[plot["dataset"].eq(dataset)].copy()
        y = np.arange(len(module_order))
        frame = frame.set_index("metric").reindex(module_order).reset_index()
        ax.errorbar(
            frame["mean_delta"], y,
            xerr=[frame["mean_delta"] - frame["bootstrap_ci95_low"], frame["bootstrap_ci95_high"] - frame["mean_delta"]],
            fmt="o", color="#2b597a", ecolor="#6c8394", capsize=3.5, lw=1.2,
        )
        ax.axvline(0, color="#333333", lw=0.85)
        ax.set_yticks(y, [labels[item] for item in module_order])
        ax.set_title(dataset, fontsize=10.5)
        ax.set_xlabel("APA minus matched control\n(cohort-specific standardized units)")
    axes[0].set_ylabel("Curated module")
    fig.suptitle("Module-level patient-paired effects", fontsize=12.3, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_4_module_level_paired_effects.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_4_module_level_paired_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_tests: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []

    # Discovery spatial cohort: the first call reproduces the all-in-tissue,
    # spot-weighted standardization used in the primary analysis.
    spatial = pd.read_csv(SPATIAL_SPOTS)
    primary_expression = spatial[list(CORE_GENE_UNIVERSE)]
    spatial_meta = spatial[["patient", "tissue", "geo_accession"]].copy()
    _, tests, coverage, weights = score_and_test(
        primary_expression, spatial_meta, "GSE274314", "patient", "tissue", "Adjacent", "primary",
        section_col="geo_accession", gene_robustness=True, module_robustness=True,
    )
    all_tests.append(tests)
    all_coverage.append(coverage)
    all_weights.append(weights)

    cyp_modules = make_modules_without_gene("CYP11B2")
    _, tests, coverage, weights = score_and_test(
        primary_expression, spatial_meta, "GSE274314", "patient", "tissue", "Adjacent", "CYP11B2_free",
        section_col="geo_accession", modules=cyp_modules,
    )
    all_tests.append(tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")])
    all_coverage.append(coverage)
    all_weights.append(weights)

    _, tests, coverage, weights = score_and_test(
        primary_expression, spatial_meta, "GSE274314", "patient", "tissue", "Adjacent", "common_genes_all_platforms",
        section_col="geo_accession", modules=COMMON_PLATFORM_MODULES,
    )
    all_tests.append(tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")])
    all_coverage.append(coverage)
    all_weights.append(weights)

    spatial_qc = spatial_mitochondrial_qc(spatial)
    qc_spot_ids = spatial_qc.loc[spatial_qc["passes_qc_500umi_250genes_mt20pct"], ["geo_accession", "barcode"]]
    qc_spatial = spatial.merge(qc_spot_ids, on=["geo_accession", "barcode"], how="inner", validate="one_to_one")
    qc_meta = qc_spatial[["patient", "tissue", "geo_accession"]].copy()
    _, tests, coverage, weights = score_and_test(
        qc_spatial[list(CORE_GENE_UNIVERSE)], qc_meta, "GSE274314", "patient", "tissue", "Adjacent",
        "spatial_qc_500umi_250genes_mt20pct", section_col="geo_accession",
    )
    all_tests.append(tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")])
    all_coverage.append(coverage)
    all_weights.append(weights)

    _, tests, coverage, weights = score_and_test(
        primary_expression, spatial_meta, "GSE274314", "patient", "tissue", "Adjacent",
        "equal_section_standardization", standardization="equal_section", section_col="geo_accession",
    )
    all_tests.append(tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")])
    all_coverage.append(coverage)
    all_weights.append(weights)

    # Independent bulk cohort, re-scored from the deposited matrix so that the
    # CYP11B2-free and module-level results use exactly the same implementation.
    gse600_expression = pd.read_csv(GSE600_EXPR).set_index("gene_symbol")
    gse600_expression.index = gse600_expression.index.astype(str).str.upper()
    gse600_meta = pd.read_csv(GSE600_META)
    gse600_meta = gse600_meta.loc[gse600_meta["geo_accession"].isin(gse600_expression.columns)].copy()
    gse600_expr_samples = gse600_expression[gse600_meta["geo_accession"].tolist()].T
    gse600_expr_samples.index = gse600_meta.index
    for analysis, modules in [
        ("primary", CORE_MODULES),
        ("CYP11B2_free", cyp_modules),
        ("common_genes_all_platforms", COMMON_PLATFORM_MODULES),
    ]:
        _, tests, coverage, weights = score_and_test(
            gse600_expr_samples, gse600_meta, "GSE60042", "case", "tissue", "AAG", analysis,
            modules=modules, gene_robustness=analysis == "primary", module_robustness=analysis == "primary",
        )
        if analysis != "primary":
            tests = tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")]
        all_tests.append(tests)
        all_coverage.append(coverage)
        all_weights.append(weights)

    # The processed matrix supports a distinct, code-identifiable eight-pair
    # analysis. It is deliberately not pooled with the other datasets.
    gse156_expression = pd.read_csv(GSE156_EXPR).set_index("gene_symbol")
    gse156_expression.index = gse156_expression.index.astype(str).str.upper()
    gse156_meta = pd.read_csv(GSE156_META)
    gse156_meta = gse156_meta.loc[gse156_meta["paired_apa_aag"] & gse156_meta["geo_accession"].isin(gse156_expression.columns)].copy()
    gse156_expr_samples = gse156_expression[gse156_meta["geo_accession"].tolist()].T
    gse156_expr_samples.index = gse156_meta.index
    for analysis, modules in [
        ("primary", CORE_MODULES),
        ("CYP11B2_free", cyp_modules),
        ("common_genes_all_platforms", COMMON_PLATFORM_MODULES),
    ]:
        _, tests, coverage, weights = score_and_test(
            gse156_expr_samples, gse156_meta, "GSE156931", "case", "tissue", "AAG", analysis,
            modules=modules, gene_robustness=analysis == "primary", module_robustness=analysis == "primary",
        )
        if analysis != "primary":
            tests = tests.loc[tests["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite")]
        all_tests.append(tests)
        all_coverage.append(coverage)
        all_weights.append(weights)

    all_tests_frame = pd.concat(all_tests, ignore_index=True)
    all_coverage_frame = pd.concat(all_coverage, ignore_index=True)
    all_weights_frame = pd.concat(all_weights, ignore_index=True)
    module_spec_rows = []
    for module, genes in CORE_MODULES.items():
        for rank, gene in enumerate(genes, start=1):
            module_spec_rows.append(
                {
                    "module": module,
                    "rank_within_module": rank,
                    "gene": gene,
                    "selection_rule": "exact ordered membership in the transparent curated panel in script 53; no target-cohort DE threshold or result-based selection",
                    "panel_status": "curated after project initiation; frozen for this revision, not preregistered",
                }
            )
    module_spec = pd.DataFrame(module_spec_rows)
    overlap = (
        module_spec.groupby("gene", as_index=False)["module"].agg(lambda x: ";".join(x))
        .rename(columns={"module": "module_memberships"})
    )
    overlap["n_module_memberships"] = overlap["module_memberships"].str.count(";") + 1
    module_spec = module_spec.merge(overlap, on="gene", how="left")

    qc_section = spatial_qc_by_section(spatial_qc)
    roi_status = pd.DataFrame(
        [
            {
                "dataset": "GSE274314",
                "tumour_roi_available": False,
                "pathology_annotation_available": False,
                "primary_spatial_estimand": "all in_tissue spots summarized to the GEO-labelled APA or adjacent-cortex section",
                "restriction_not_performed_reason": "The public archive contains section images and coordinates but no returned blinded pathology/tumour ROI mask. Transcript-defined ROIs were not used because they would be circular for this score.",
            }
        ]
    )

    module_spec.to_csv(RESULTS / "BMC_module_specification.csv", index=False, encoding="utf-8-sig")
    all_weights_frame.to_csv(RESULTS / "BMC_module_definitions_and_effective_weights.csv", index=False, encoding="utf-8-sig")
    all_coverage_frame.to_csv(RESULTS / "BMC_platform_coverage_and_common_gene_sensitivity.csv", index=False, encoding="utf-8-sig")
    all_tests_frame.to_csv(RESULTS / "BMC_cohort_paired_module_and_robustness_tests.csv", index=False, encoding="utf-8-sig")
    spatial_qc.to_csv(RESULTS / "GSE274314_spatial_spot_qc.csv", index=False, encoding="utf-8-sig")
    qc_section.to_csv(RESULTS / "GSE274314_spatial_qc_by_section.csv", index=False, encoding="utf-8-sig")
    roi_status.to_csv(RESULTS / "GSE274314_spatial_roi_availability.csv", index=False, encoding="utf-8-sig")
    make_module_figure(all_tests_frame)

    primary_summary = all_tests_frame.loc[
        all_tests_frame["analysis"].eq("primary")
        & all_tests_frame["metric"].eq("ZG_intermediate_vs_ZF_ZR_composite"),
        ["dataset", "mean_delta", "median_delta", "n_positive", "n_pairs", "exact_sign_flip_two_sided_p"],
    ]
    print("Primary composite results:")
    print(primary_summary.to_string(index=False))
    print("Wrote BMC robustness and spatial-QC outputs to", RESULTS)


if __name__ == "__main__":
    main()
