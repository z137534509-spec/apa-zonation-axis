from __future__ import annotations

import gzip
import importlib.util
import itertools
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "GSE156931"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results" / "bmc_revision_20260803"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SERIES_MATRIX = RAW / "GSE156931_series_matrix.txt.gz"
PLATFORM_SOFT = RAW / "GPL6883_family.soft.gz"
SPATIAL_SCORE_FILE = PROJECT / "results" / "stage2_zonation_axis" / "GSE274314_stage2_spot_scores.csv"
GSE60042_SCORES = PROJECT / "results" / "external_transcriptome_methylome_20260602" / "GSE60042_expression_axis_scores.csv"
GSE60042_TESTS = PROJECT / "results" / "external_transcriptome_methylome_20260602" / "GSE60042_expression_paired_tests.csv"

RNG = np.random.default_rng(20260803)
N_BOOT = 10_000
N_MORAN_PERM = 999
PRIMARY_AXIS = "ZG_intermediate_vs_ZF_ZR_axis_score"
CYP11B2_FREE_AXIS = "ZG_intermediate_vs_ZF_ZR_axis_score_CYP11B2_free"
GRID_SIZES = (3, 4, 5)


def load_existing_analysis_module():
    """Reuse the established score definitions and GEO parser from the released analysis."""
    source = PROJECT / "scripts" / "26_external_transcriptome_methylome_validation.py"
    spec = importlib.util.spec_from_file_location("apa_external_validation", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load existing analysis definitions from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_existing_analysis_module()


def parse_illumina_probe_map(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                header = next(handle).rstrip("\n").split("\t")
                break
        else:
            raise RuntimeError(f"Could not find platform annotation table in {path}")

        for line in handle:
            if line.startswith("!platform_table_end"):
                break
            values = line.rstrip("\n").split("\t")
            if len(values) != len(header):
                continue
            record = dict(zip(header, values))
            probe = str(record.get("ID", "")).strip()
            symbols = str(record.get("Symbol", ""))
            for symbol in re.split(r"[;|/,]+", symbols):
                symbol = symbol.strip().upper()
                if re.fullmatch(r"[A-Z0-9_.-]+", symbol or ""):
                    rows.append({"ID_REF": probe, "gene_symbol": symbol})
    out = pd.DataFrame(rows).drop_duplicates()
    if out.empty:
        raise RuntimeError("No gene symbols could be parsed from GPL6883")
    return out


def add_gse156931_pair_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Use only code pairs that have both C and N samples; do not infer control diagnoses."""
    out = meta.copy()
    parsed = out["title"].astype(str).str.extract(r"^(?P<prefix>[A-Z]+)(?P<case>\d+)(?P<specimen>[CN])$")
    out = pd.concat([out, parsed], axis=1)
    out["pair_id"] = out["prefix"].fillna("") + out["case"].fillna("")
    paired_ids = (
        out.dropna(subset=["pair_id", "specimen"])
        .groupby("pair_id")["specimen"]
        .agg(lambda x: set(x))
    )
    paired_ids = sorted(pair_id for pair_id, members in paired_ids.items() if members == {"C", "N"})
    out["paired_apa_aag"] = out["pair_id"].isin(paired_ids)
    out["tissue"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[out["paired_apa_aag"] & out["specimen"].eq("C"), "tissue"] = "APA"
    out.loc[out["paired_apa_aag"] & out["specimen"].eq("N"), "tissue"] = "AAG"
    out["case"] = out["pair_id"].where(out["paired_apa_aag"])
    out["dataset"] = "GSE156931"
    return out


def zscore_by_gene(gene_expression: pd.DataFrame) -> pd.DataFrame:
    samples_by_gene = gene_expression.T.copy()
    for gene in samples_by_gene.columns:
        values = samples_by_gene[gene].astype(float)
        sd = values.std(ddof=0)
        samples_by_gene[gene] = (values - values.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return samples_by_gene


def score_expression(gene_expression: pd.DataFrame, meta: pd.DataFrame, standardization_scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_by_gene(gene_expression)
    z.columns = z.columns.astype(str).str.upper()
    scores = pd.DataFrame(index=z.index)
    availability_rows = []
    variants = {
        "original": BASE.SIGNATURES,
        "CYP11B2_free": {
            name: [gene for gene in genes if not (name == "ZG_aldosterone_program" and gene == "CYP11B2")]
            for name, genes in BASE.SIGNATURES.items()
        },
    }
    for variant, signatures in variants.items():
        module_scores = {}
        for name, genes in signatures.items():
            requested = [gene.upper() for gene in genes]
            available = [gene for gene in requested if gene in z.columns]
            missing = [gene for gene in requested if gene not in z.columns]
            module_scores[name] = z[available].mean(axis=1) if available else pd.Series(np.nan, index=z.index)
            availability_rows.append(
                {
                    "dataset": "GSE156931",
                    "standardization_scope": standardization_scope,
                    "variant": variant,
                    "signature": name,
                    "n_requested": len(requested),
                    "n_available": len(available),
                    "available_genes": ";".join(available),
                    "missing_genes": ";".join(missing),
                }
            )
        if variant == "original":
            for name, values in module_scores.items():
                scores[name] = values
            scores[PRIMARY_AXIS] = primary_axis_from_modules(module_scores)
            scores["ZG_vs_ZF_axis_score"] = module_scores["ZG_aldosterone_program"] - module_scores["ZF_cortisol_program"]
            scores["aldosterone_specificity_score"] = module_scores["ZG_aldosterone_program"] - module_scores["generic_steroidogenesis_program"]
        else:
            scores[CYP11B2_FREE_AXIS] = primary_axis_from_modules(module_scores)
    scores = scores.reset_index(names="geo_accession").merge(meta, on="geo_accession", how="left")
    scores["standardization_scope"] = standardization_scope
    return scores, pd.DataFrame(availability_rows)


def primary_axis_from_modules(module_scores: dict[str, pd.Series] | pd.DataFrame) -> pd.Series:
    """Return the fixed four-module cortical-program contrast used in the BMC revision."""
    return (
        (module_scores["ZG_aldosterone_program"] + module_scores["intermediate_steroidogenic_program"]) / 2
        - (module_scores["ZF_cortisol_program"] + module_scores["ZR_androgen_program"]) / 2
    )


def add_primary_axis_from_modules(scores: pd.DataFrame) -> pd.DataFrame:
    required = ["ZG_aldosterone_program", "intermediate_steroidogenic_program", "ZF_cortisol_program", "ZR_androgen_program"]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise KeyError(f"Cannot derive {PRIMARY_AXIS}; missing module columns: {missing}")
    scores = scores.copy()
    scores[PRIMARY_AXIS] = primary_axis_from_modules(scores)
    return scores


def add_spatial_cyp11b2_free_axis(spots: pd.DataFrame) -> pd.DataFrame:
    """Derive the CYP11B2-removal sensitivity score from stored spatial module scores."""
    if "CYP11B2" not in spots.columns:
        raise KeyError("The spatial score table does not contain CYP11B2 expression required for the sensitivity analysis.")
    n_zg_genes = len(BASE.SIGNATURES["ZG_aldosterone_program"])
    values = spots["CYP11B2"].astype(float)
    sd = values.std(ddof=0)
    if not sd or not np.isfinite(sd):
        raise ValueError("CYP11B2 has zero or invalid variation in the spatial score table.")
    standardized_cyp11b2 = (values - values.mean()) / sd
    scores = spots.copy()
    zg_without_cyp11b2 = (n_zg_genes * scores["ZG_aldosterone_program"] - standardized_cyp11b2) / (n_zg_genes - 1)
    scores[CYP11B2_FREE_AXIS] = (
        (zg_without_cyp11b2 + scores["intermediate_steroidogenic_program"]) / 2
        - (scores["ZF_cortisol_program"] + scores["ZR_androgen_program"]) / 2
    )
    return scores


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    boot = RNG.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    observed = float(values.mean())
    null = np.fromiter(
        ((values * np.asarray(bits)).mean() for bits in itertools.product([-1, 1], repeat=values.size)), dtype=float
    )
    return float(np.mean(np.abs(null) >= abs(observed))), float(np.mean(null >= observed))


def paired_tests(scores: pd.DataFrame, metrics: list[str], dataset: str, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = []
    delta_rows = []
    for metric in metrics:
        wide = scores.pivot_table(index="case", columns="tissue", values=metric, aggfunc="mean")
        if not {"APA", "AAG"}.issubset(wide.columns):
            continue
        delta = (wide["APA"] - wide["AAG"]).dropna().astype(float)
        ci_low, ci_high = bootstrap_ci(delta.to_numpy())
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        test_rows.append(
            {
                "dataset": dataset,
                "standardization_scope": scope,
                "metric": metric,
                "comparison": "APA vs AAG",
                "n_pairs": int(delta.size),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
            }
        )
        delta_rows.extend(
            {
                "dataset": dataset,
                "standardization_scope": scope,
                "metric": metric,
                "case": case,
                "delta_APA_minus_AAG": float(value),
            }
            for case, value in delta.items()
        )
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["exact_sign_flip_two_sided_fdr"] = BASE.bh_fdr(tests["exact_sign_flip_two_sided_p"])
    return tests, pd.DataFrame(delta_rows)


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if values.size < k + 2 or np.nanstd(values) == 0:
        return np.nan, np.empty((0, 0), dtype=int)
    neighbors = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]
    centered = values - values.mean()
    numerator = np.sum(centered[:, None] * centered[neighbors])
    denominator = np.sum(centered**2)
    return float((values.size / neighbors.size) * numerator / denominator), neighbors


def block_iqr(values: np.ndarray, x: np.ndarray, y: np.ndarray, grid_size: int) -> tuple[float, int]:
    frame = pd.DataFrame({"value": values, "x": x, "y": y}).dropna()
    if frame.empty:
        return np.nan, 0
    frame["x_block"] = np.minimum((frame["x"].rank(pct=True, method="first") * grid_size).astype(int), grid_size - 1)
    frame["y_block"] = np.minimum((frame["y"].rank(pct=True, method="first") * grid_size).astype(int), grid_size - 1)
    medians = frame.groupby(["x_block", "y_block"])["value"].median()
    return float(medians.quantile(0.75) - medians.quantile(0.25)), int(medians.size)


def section_spatial_summary(spots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    permutation_rows = []
    for (patient, tissue, accession), section in spots.groupby(["patient", "tissue", "geo_accession"], dropna=False):
        section = section.dropna(subset=[PRIMARY_AXIS, "lowres_x", "lowres_y"])
        values = section[PRIMARY_AXIS].to_numpy(float)
        coords = section[["lowres_x", "lowres_y"]].to_numpy(float)
        observed_moran, neighbors = morans_i(values, coords)
        null = np.array([])
        if neighbors.size:
            null = np.fromiter(
                (morans_i(RNG.permutation(values), coords)[0] for _ in range(N_MORAN_PERM)), dtype=float
            )
        moran_greater_p = float((np.sum(null >= observed_moran) + 1) / (null.size + 1)) if null.size else np.nan
        value_median = float(np.median(values))
        mad = float(np.median(np.abs(values - value_median)))
        row = {
            "patient": patient,
            "tissue": tissue,
            "geo_accession": accession,
            "n_spots": int(values.size),
            "axis_mean": float(np.mean(values)),
            "axis_median": value_median,
            "axis_mean_CYP11B2_free": float(section[CYP11B2_FREE_AXIS].mean()) if CYP11B2_FREE_AXIS in section.columns else np.nan,
            "axis_sd": float(np.std(values, ddof=1)),
            "axis_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "axis_mad": mad,
            "axis_p90_minus_p10": float(np.quantile(values, 0.90) - np.quantile(values, 0.10)),
            "moran_i_k6": observed_moran,
            "moran_i_greater_permutation_p": moran_greater_p,
        }
        for grid_size in GRID_SIZES:
            regional_iqr, n_blocks = block_iqr(values, coords[:, 0], coords[:, 1], grid_size)
            row[f"spatial_block_iqr_g{grid_size}"] = regional_iqr
            row[f"n_spatial_blocks_g{grid_size}"] = n_blocks
        rows.append(row)
        permutation_rows.append(
            {
                "patient": patient,
                "tissue": tissue,
                "geo_accession": accession,
                "moran_i_k6": observed_moran,
                "moran_i_greater_permutation_p": moran_greater_p,
                "n_permutations": int(null.size),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(permutation_rows)


def paired_section_tests(section_summary: pd.DataFrame) -> pd.DataFrame:
    location_metrics = ["axis_mean", "axis_median", "axis_mean_CYP11B2_free"]
    spatial_heterogeneity_metrics = [
        "axis_iqr", "axis_mad", "axis_p90_minus_p10",
        *[f"spatial_block_iqr_g{grid_size}" for grid_size in GRID_SIZES],
        "moran_i_k6",
    ]
    metrics = location_metrics + spatial_heterogeneity_metrics
    rows = []
    for metric in metrics:
        wide = section_summary.pivot_table(index="patient", columns="tissue", values=metric, aggfunc="mean")
        if not {"APA", "Adjacent"}.issubset(wide.columns):
            continue
        delta = (wide["APA"] - wide["Adjacent"]).dropna().astype(float)
        ci_low, ci_high = bootstrap_ci(delta.to_numpy())
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        rows.append(
            {
                "metric": metric,
                "endpoint_family": "location" if metric in location_metrics else "spatial_heterogeneity",
                "comparison": "APA vs adjacent cortex",
                "n_pairs": int(delta.size),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["fdr_family"] = np.where(
            output["endpoint_family"].eq("spatial_heterogeneity"),
            "seven spatial-heterogeneity endpoints",
            "not adjusted (location endpoints)",
        )
        output["exact_sign_flip_two_sided_fdr"] = np.nan
        mask = output["endpoint_family"].eq("spatial_heterogeneity")
        output.loc[mask, "exact_sign_flip_two_sided_fdr"] = BASE.bh_fdr(output.loc[mask, "exact_sign_flip_two_sided_p"])
    return output


def paired_line_plot(ax: plt.Axes, data: pd.DataFrame, metric: str, label: str, tissue_a: str, tissue_b: str) -> None:
    subset = data[["case", "tissue", metric]].dropna()
    order = [tissue_b, tissue_a]
    sns.stripplot(data=subset, x="tissue", y=metric, order=order, ax=ax, color="#202020", size=4.5, jitter=0.03)
    for _, group in subset.groupby("case"):
        pair = group.set_index("tissue")[metric]
        if set(order).issubset(pair.index):
            ax.plot([0, 1], [pair[tissue_b], pair[tissue_a]], color="#74818f", lw=0.9, alpha=0.75)
    means = subset.groupby("tissue")[metric].mean().reindex(order)
    ax.plot([0, 1], means.to_numpy(), color="#c24e42", lw=2.0, marker="o")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("")


def make_primary_spatial_figure(section_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    primary = section_summary.rename(columns={"patient": "case", "axis_mean": "value"})[["case", "tissue", "value"]]
    paired_line_plot(axes[0], primary, "value", "A. Curated primary composite", "APA", "Adjacent")
    axes[0].set_ylabel("Section mean composite score")
    cyp_free = section_summary.rename(columns={"patient": "case", "axis_mean_CYP11B2_free": "value"})[["case", "tissue", "value"]]
    paired_line_plot(axes[1], cyp_free, "value", "B. CYP11B2-free sensitivity", "APA", "Adjacent")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_3_primary_zonation_axis_signal.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_3_primary_zonation_axis_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def make_external_figure(
    gse60042_scores: pd.DataFrame,
    gse60042_tests: pd.DataFrame,
    gse156_scores: pd.DataFrame,
    gse156_tests: pd.DataFrame,
) -> None:
    gse60042_scores["case"] = gse60042_scores["case"].astype(str)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    paired_line_plot(axes[0], gse60042_scores, PRIMARY_AXIS, "A. GSE60042 (n=7 pairs)", "APA", "AAG")
    axes[0].set_ylabel("Curated composite score")
    paired_line_plot(axes[1], gse156_scores[gse156_scores["paired_apa_aag"]], PRIMARY_AXIS, "B. GSE156931 (n=8 pairs)", "APA", "AAG")
    axes[1].set_ylabel("")

    rows = []
    for dataset, tests in [("GSE60042", gse60042_tests), ("GSE156931", gse156_tests)]:
        if dataset == "GSE60042":
            row = tests[tests["metric"] == PRIMARY_AXIS]
        else:
            row = tests[(tests["metric"] == PRIMARY_AXIS) & (tests["standardization_scope"] == "paired_samples_only")]
        if row.empty:
            continue
        record = row.iloc[0]
        rows.append({"dataset": dataset, "mean_delta": record["mean_delta"], "ci_low": record["ci95_low"], "ci_high": record["ci95_high"]})
    summary = pd.DataFrame(rows)
    positions = np.arange(summary.shape[0])
    axes[2].errorbar(
        summary["mean_delta"], positions,
        xerr=[summary["mean_delta"] - summary["ci_low"], summary["ci_high"] - summary["mean_delta"]],
        fmt="o", color="#c24e42", ecolor="#465766", capsize=4, lw=1.3,
    )
    axes[2].axvline(0, color="#333333", lw=0.9)
    axes[2].set_yticks(positions, summary["dataset"])
    axes[2].set_xlabel("APA minus AAG mean paired difference\n(descriptive bootstrap 95% CI)")
    axes[2].set_title("C. Independent paired cohorts", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_6_external_paired_cohort_concordance.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_6_external_paired_cohort_concordance.pdf", bbox_inches="tight")
    plt.close(fig)


def make_spatial_figure(section_summary: pd.DataFrame, section_tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    location = section_summary.rename(columns={"patient": "case", "axis_mean": "value"})[["case", "tissue", "value"]]
    paired_line_plot(axes[0], location, "value", "A. Lesion-level location", "APA", "Adjacent")
    axes[0].set_ylabel("Section mean composite score")
    dispersion = section_summary.rename(columns={"patient": "case", "axis_iqr": "value"})[["case", "tissue", "value"]]
    paired_line_plot(axes[1], dispersion, "value", "B. Spot-score dispersion", "APA", "Adjacent")
    axes[1].set_ylabel("Section axis-score IQR")
    regional_rows = []
    for grid_size in GRID_SIZES:
        row = section_tests.loc[section_tests["metric"].eq(f"spatial_block_iqr_g{grid_size}")].iloc[0]
        regional_rows.append({"grid": f"{grid_size}×{grid_size}", "mean_delta": row["mean_delta"], "ci_low": row["ci95_low"], "ci_high": row["ci95_high"]})
    regional = pd.DataFrame(regional_rows)
    # Assign the display labels separately to avoid inheriting a malformed
    # multiplication glyph from an earlier text conversion.
    regional["grid"] = [f"{grid_size}×{grid_size}" for grid_size in GRID_SIZES]
    positions = np.arange(regional.shape[0])
    axes[2].errorbar(
        regional["mean_delta"], positions,
        xerr=[regional["mean_delta"] - regional["ci_low"], regional["ci_high"] - regional["mean_delta"]],
        fmt="o", color="#c24e42", ecolor="#465766", capsize=4, lw=1.3,
    )
    axes[2].axvline(0, color="#333333", lw=0.9)
    axes[2].set_yticks(positions, regional["grid"])
    axes[2].set_xlabel("APA minus adjacent difference\n(descriptive bootstrap 95% CI)")
    axes[2].set_ylabel("Grid resolution")
    axes[2].set_title("C. Regional-dispersion sensitivity", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_5_spatial_location_and_dispersion.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_5_spatial_location_and_dispersion.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(gse156_meta: pd.DataFrame, gse156_tests: pd.DataFrame, section_tests: pd.DataFrame, section_summary: pd.DataFrame) -> None:
    primary = gse156_tests[(gse156_tests["metric"] == PRIMARY_AXIS) & (gse156_tests["standardization_scope"] == "paired_samples_only")].iloc[0]
    cyp_free = gse156_tests[(gse156_tests["metric"] == CYP11B2_FREE_AXIS) & (gse156_tests["standardization_scope"] == "paired_samples_only")].iloc[0]
    location = section_tests[section_tests["metric"] == "axis_mean"].iloc[0]
    location_cyp_free = section_tests[section_tests["metric"] == "axis_mean_CYP11B2_free"].iloc[0]
    iqr = section_tests[section_tests["metric"] == "axis_iqr"].iloc[0]
    moran = section_tests[section_tests["metric"] == "moran_i_k6"].iloc[0]
    regional = {
        grid_size: section_tests[section_tests["metric"].eq(f"spatial_block_iqr_g{grid_size}")].iloc[0]
        for grid_size in GRID_SIZES
    }
    apa_spatially_structured = section_summary[(section_summary["tissue"] == "APA") & (section_summary["moran_i_greater_permutation_p"] < 0.05)].shape[0]
    lines = [
        "# BMC revision validation report — 2026-08-03",
        "",
        "## Analysis roles",
        "",
        "- GSE156931 supplies an independent paired expression validation. Its public matrix contains eight identifiable APA/AAG code pairs. The GEO record states that raw data could not be located, so this analysis uses the deposited processed matrix and makes no preprocessing claim.",
        "- Unpaired GSE156931 samples are retained in the audit table but are not assigned CPA, pheochromocytoma, or hyperplasia diagnoses because the deposited sample metadata does not provide auditable disease labels.",
        "- GSE274314 spatial analyses use patient-paired section summaries for inference. Spot-level scores quantify within-section distribution and spatial autocorrelation, not additional patient replicates.",
        "",
        "## GSE156931 paired expression transfer",
        "",
        f"- Identifiable APA/AAG pairs: {int(gse156_meta['paired_apa_aag'].sum() / 2)}; paired samples: {int(gse156_meta['paired_apa_aag'].sum())}.",
        "- The reference-defined axis is the mean of the aldosterone-oriented ZG and intermediate modules minus the mean of the cortisol-oriented ZF and androgen-oriented ZR modules.",
        "- Bootstrap 95% intervals are descriptive summaries of the paired mean. Primary inference uses an exact two-sided sign-flip test across all possible sign assignments of the observed paired differences; this is not a binomial sign test.",
        f"- Primary axis: mean APA-AAG difference {primary['mean_delta']:.4f}; descriptive bootstrap 95% CI {primary['ci95_low']:.4f} to {primary['ci95_high']:.4f}; {int(primary['n_positive'])}/{int(primary['n_pairs'])} pairs positive; exact two-sided sign-flip P={primary['exact_sign_flip_two_sided_p']:.5f}.",
        f"- CYP11B2-free sensitivity: mean APA-AAG difference {cyp_free['mean_delta']:.4f}; descriptive bootstrap 95% CI {cyp_free['ci95_low']:.4f} to {cyp_free['ci95_high']:.4f}; {int(cyp_free['n_positive'])}/{int(cyp_free['n_pairs'])} pairs positive; exact two-sided sign-flip P={cyp_free['exact_sign_flip_two_sided_p']:.5f}.",
        "",
        "## GSE274314 spatial location and dispersion",
        "",
        f"- Lesion-level location: mean APA-adjacent difference {location['mean_delta']:.4f}; {int(location['n_positive'])}/{int(location['n_pairs'])} pairs positive; exact two-sided sign-flip P={location['exact_sign_flip_two_sided_p']:.5f}.",
        f"- CYP11B2-free lesion-level location sensitivity: mean APA-adjacent difference {location_cyp_free['mean_delta']:.4f}; {int(location_cyp_free['n_positive'])}/{int(location_cyp_free['n_pairs'])} pairs positive; exact two-sided sign-flip P={location_cyp_free['exact_sign_flip_two_sided_p']:.5f}.",
        f"- Raw spot-score dispersion (IQR): mean APA-adjacent difference {iqr['mean_delta']:.4f}; {int(iqr['n_positive'])}/{int(iqr['n_pairs'])} pairs positive; exact two-sided sign-flip P={iqr['exact_sign_flip_two_sided_p']:.5f}.",
        *[
            f"- Regional dispersion ({grid_size}×{grid_size} block-median IQR): mean APA-adjacent difference {regional[grid_size]['mean_delta']:.4f}; {int(regional[grid_size]['n_positive'])}/{int(regional[grid_size]['n_pairs'])} pairs positive; exact two-sided sign-flip P={regional[grid_size]['exact_sign_flip_two_sided_p']:.5f}; FDR across seven spatial-heterogeneity endpoints={regional[grid_size]['exact_sign_flip_two_sided_fdr']:.5f}."
            for grid_size in GRID_SIZES
        ],
        f"- Spatial autocorrelation (Moran's I): mean APA-adjacent difference {moran['mean_delta']:.4f}; {int(moran['n_positive'])}/{int(moran['n_pairs'])} pairs positive; exact two-sided sign-flip P={moran['exact_sign_flip_two_sided_p']:.5f}. {apa_spatially_structured}/7 APA sections exceeded the within-section permutation threshold P<0.05.",
        "",
        "## Writing rule",
        "",
        "GSE156931 is directionally concordant in seven of eight pairs and reaches the exact paired threshold for the primary axis, but its processed-matrix-only availability and modest sample size mean that it should be interpreted as independent directional support rather than pooled with the other cohorts. The seven spatial-heterogeneity endpoints are the raw spot-score IQR, median absolute deviation, 90th-minus-10th percentile range, block-median IQR at 3×3, 4×4, and 5×5 grids, and Moran's I. Location endpoints are not part of this FDR family. Report regional dispersion only when its direction is stable across grid resolutions, and note that lower lesion-level dispersion can reflect replacement of normally layered ZG-ZF-ZR cortex rather than intrinsic APA homogenization.",
    ]
    (RESULTS / "bmc_revision_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SERIES_MATRIX.exists() or not PLATFORM_SOFT.exists():
        raise FileNotFoundError("Download GSE156931 series matrix and GPL6883 platform SOFT before running this script.")
    meta, probe_matrix = BASE.parse_series_matrix(SERIES_MATRIX)
    meta = add_gse156931_pair_metadata(meta)
    probe_map = parse_illumina_probe_map(PLATFORM_SOFT)
    gene_expression = BASE.collapse_probe_to_gene(probe_matrix, probe_map)

    paired_meta = meta[meta["paired_apa_aag"]].copy()
    paired_accessions = paired_meta["geo_accession"].tolist()
    paired_scores, availability = score_expression(gene_expression[paired_accessions], paired_meta, "paired_samples_only")
    all_scores, all_availability = score_expression(gene_expression, meta, "all_24_samples_sensitivity")
    all_paired_scores = all_scores[all_scores["paired_apa_aag"]].copy()
    metrics = [
        "ZG_aldosterone_program", "ZF_cortisol_program", "ZR_androgen_program", "intermediate_steroidogenic_program",
        "generic_steroidogenesis_program", PRIMARY_AXIS, CYP11B2_FREE_AXIS, "ZG_vs_ZF_axis_score", "aldosterone_specificity_score", "housekeeping_control",
    ]
    paired_tests_primary, paired_deltas_primary = paired_tests(paired_scores, metrics, "GSE156931", "paired_samples_only")
    paired_tests_all, paired_deltas_all = paired_tests(all_paired_scores, metrics, "GSE156931", "all_24_samples_sensitivity")
    gse156_tests = pd.concat([paired_tests_primary, paired_tests_all], ignore_index=True)
    gse156_deltas = pd.concat([paired_deltas_primary, paired_deltas_all], ignore_index=True)

    gse60042_scores = add_primary_axis_from_modules(pd.read_csv(GSE60042_SCORES))
    gse60042_scores["case"] = gse60042_scores["case"].astype(str)
    gse60042_tests, _ = paired_tests(gse60042_scores, [PRIMARY_AXIS], "GSE60042", "source-defined_standardization")

    spots = add_spatial_cyp11b2_free_axis(add_primary_axis_from_modules(pd.read_csv(SPATIAL_SCORE_FILE)))
    section_summary, moran_sections = section_spatial_summary(spots)
    section_tests = paired_section_tests(section_summary)
    occupancy = section_summary.melt(
        id_vars=["patient", "tissue", "geo_accession", "n_spots"],
        value_vars=[f"n_spatial_blocks_g{grid_size}" for grid_size in GRID_SIZES],
        var_name="grid_metric",
        value_name="n_occupied_blocks",
    )
    occupancy["grid_size"] = occupancy["grid_metric"].str.extract(r"g(\d+)").astype(int)
    occupancy = occupancy.drop(columns="grid_metric").sort_values(["grid_size", "patient", "tissue"])

    meta.to_csv(PROCESSED / "GSE156931_sample_metadata_audited.csv", index=False, encoding="utf-8-sig")
    probe_map.to_csv(PROCESSED / "GPL6883_probe_gene_map.csv", index=False, encoding="utf-8-sig")
    gene_expression.reset_index().to_csv(PROCESSED / "GSE156931_gene_expression_mean_collapsed.csv", index=False, encoding="utf-8-sig")
    paired_scores.to_csv(RESULTS / "GSE156931_expression_axis_scores_paired_samples_only.csv", index=False, encoding="utf-8-sig")
    all_scores.to_csv(RESULTS / "GSE156931_expression_axis_scores_all_24_samples_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat([availability, all_availability], ignore_index=True).to_csv(RESULTS / "GSE156931_signature_gene_availability.csv", index=False, encoding="utf-8-sig")
    gse156_tests.to_csv(RESULTS / "GSE156931_expression_paired_tests.csv", index=False, encoding="utf-8-sig")
    gse156_deltas.to_csv(RESULTS / "GSE156931_expression_paired_deltas.csv", index=False, encoding="utf-8-sig")
    gse60042_scores.to_csv(RESULTS / "GSE60042_expression_axis_scores_bmc_four_module_axis.csv", index=False, encoding="utf-8-sig")
    gse60042_tests.to_csv(RESULTS / "GSE60042_expression_paired_tests_bmc_four_module_axis.csv", index=False, encoding="utf-8-sig")
    section_summary.to_csv(RESULTS / "GSE274314_section_location_and_dispersion.csv", index=False, encoding="utf-8-sig")
    moran_sections.to_csv(RESULTS / "GSE274314_section_moran_permutation_tests.csv", index=False, encoding="utf-8-sig")
    section_tests.to_csv(RESULTS / "GSE274314_paired_location_and_dispersion_tests.csv", index=False, encoding="utf-8-sig")
    occupancy.to_csv(RESULTS / "GSE274314_spatial_grid_occupancy.csv", index=False, encoding="utf-8-sig")
    make_primary_spatial_figure(section_summary)
    make_external_figure(gse60042_scores, gse60042_tests, paired_scores, gse156_tests)
    make_spatial_figure(section_summary, section_tests)
    write_report(meta, gse156_tests, section_tests, section_summary)
    print(gse156_tests[(gse156_tests["metric"].isin([PRIMARY_AXIS, CYP11B2_FREE_AXIS])) & (gse156_tests["standardization_scope"] == "paired_samples_only")].to_string(index=False))
    print(section_tests.to_string(index=False))
    print(f"Wrote BMC revision validation outputs to {RESULTS}")


if __name__ == "__main__":
    main()
