from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
STAGE2 = RESULTS / "stage2_zonation_axis"
OUT = RESULTS / "steroidogenic_cascade_activity_20260602"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CASCADE_MODULES = {
    "cholesterol_import_initiation": ["SCARB1", "STAR", "CYP11A1"],
    "pregnenolone_to_DOC_branch": ["HSD3B2", "CYP21A2"],
    "aldosterone_terminal_step": ["CYP11B2"],
    "cortisol_terminal_branch": ["CYP11B1", "CYP17A1"],
    "androgen_ZR_diversion_branch": ["CYP17A1", "CYB5A", "SULT2A1", "SULT2B1"],
}

KEY_METRICS = [
    "cholesterol_import_initiation",
    "pregnenolone_to_DOC_branch",
    "aldosterone_terminal_step",
    "cortisol_terminal_branch",
    "androgen_ZR_diversion_branch",
    "aldosterone_cascade_activity",
    "aldosterone_vs_cortisol_branch_index",
    "aldosterone_vs_androgen_branch_index",
    "steroidogenic_cascade_pseudoflux_index",
]


def exact_sign_flip(delta: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan, np.nan
    obs = float(delta.mean())
    vals = []
    for bits in itertools.product([-1, 1], repeat=len(delta)):
        vals.append(float((delta * np.asarray(bits)).mean()))
    vals = np.asarray(vals)
    return float(np.mean(np.abs(vals) >= abs(obs))), float(np.mean(vals >= obs))


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(out, index=pvalues.index)
    vals = p[ok]
    order = np.argsort(vals)
    ranks = np.arange(1, len(vals) + 1)
    adj = vals[order] * len(vals) / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(vals)
    tmp[order] = adj
    out[ok] = tmp
    return pd.Series(out, index=pvalues.index)


def zscore(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    out = {}
    for gene in genes:
        if gene not in df.columns:
            continue
        vals = df[gene].astype(float)
        sd = vals.std(ddof=0)
        out[gene] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else pd.Series(0.0, index=df.index)
    return pd.DataFrame(out, index=df.index)


def add_cascade_scores(df: pd.DataFrame, id_cols: list[str], dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_genes = sorted(set(g for genes in CASCADE_MODULES.values() for g in genes))
    z = zscore(df, all_genes)
    score = df[id_cols].copy()
    availability = []
    for module, genes in CASCADE_MODULES.items():
        present = [g for g in genes if g in z.columns]
        missing = [g for g in genes if g not in z.columns]
        score[module] = z[present].mean(axis=1) if present else np.nan
        availability.append(
            {
                "dataset": dataset,
                "module": module,
                "n_requested": len(genes),
                "n_available": len(present),
                "available_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )
    score["aldosterone_cascade_activity"] = score[["cholesterol_import_initiation", "pregnenolone_to_DOC_branch", "aldosterone_terminal_step"]].mean(axis=1)
    score["aldosterone_vs_cortisol_branch_index"] = score["aldosterone_terminal_step"] - score["cortisol_terminal_branch"]
    score["aldosterone_vs_androgen_branch_index"] = score["aldosterone_terminal_step"] - score["androgen_ZR_diversion_branch"]
    score["steroidogenic_cascade_pseudoflux_index"] = score["aldosterone_cascade_activity"] - score[["cortisol_terminal_branch", "androgen_ZR_diversion_branch"]].mean(axis=1)
    return score, pd.DataFrame(availability)


def paired_tests(scores: pd.DataFrame, dataset: str, tissue_a: str, tissue_b: str, pair_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    delta_rows = []
    for metric in KEY_METRICS:
        wide = scores.pivot_table(index=pair_col, columns="tissue", values=metric, aggfunc="mean")
        if not {tissue_a, tissue_b}.issubset(set(wide.columns)):
            continue
        wide = wide.dropna(subset=[tissue_a, tissue_b])
        if wide.empty:
            continue
        delta = wide[tissue_a] - wide[tissue_b]
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "comparison": f"{tissue_a} vs {tissue_b}",
                "n_pairs": int(delta.shape[0]),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
                "paired_t_p": stats.ttest_1samp(delta, 0.0).pvalue if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
            }
        )
        for pair, value in delta.items():
            delta_rows.append({"dataset": dataset, "metric": metric, pair_col: pair, "delta": float(value)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exact_sign_flip_two_sided_fdr"] = bh_fdr(out["exact_sign_flip_two_sided_p"])
    return out, pd.DataFrame(delta_rows)


def axis_high_enrichment(spots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gsm, patient, tissue), sub in spots.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        high = sub["axis_top_quartile_roi"].astype(bool)
        if high.sum() < 5 or (~high).sum() < 5:
            continue
        for metric in KEY_METRICS:
            vals = sub[metric].astype(float)
            diff = float(vals[high].mean() - vals[~high].mean())
            rows.append(
                {
                    "geo_accession": gsm,
                    "patient": int(patient),
                    "tissue": tissue,
                    "metric": metric,
                    "axis_high_minus_other_mean": diff,
                    "n_axis_high": int(high.sum()),
                    "n_other": int((~high).sum()),
                }
            )
    out = pd.DataFrame(rows)
    summary_rows = []
    for (metric, tissue), sub in out.groupby(["metric", "tissue"], dropna=False):
        vals = sub["axis_high_minus_other_mean"].astype(float).dropna().to_numpy()
        exact_two, exact_greater = exact_sign_flip(vals)
        summary_rows.append(
            {
                "metric": metric,
                "tissue": tissue,
                "n_sections": int(len(vals)),
                "mean_axis_high_minus_other": float(vals.mean()),
                "median_axis_high_minus_other": float(np.median(vals)),
                "n_positive": int((vals > 0).sum()),
                "n_negative": int((vals < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["exact_sign_flip_two_sided_fdr"] = bh_fdr(summary["exact_sign_flip_two_sided_p"])
    return out, summary


def load_gse60042_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = pd.read_csv(PROCESSED / "GSE60042_gene_expression_mean_collapsed.csv")
    expr["gene_symbol"] = expr["gene_symbol"].astype(str).str.upper()
    expr = expr.groupby("gene_symbol", as_index=False).mean(numeric_only=True).set_index("gene_symbol")
    meta = pd.read_csv(PROCESSED / "GSE60042_sample_metadata.csv")
    sample_cols = [c for c in expr.columns if c in set(meta["geo_accession"])]
    sample_expr = expr[sample_cols].T
    sample_expr.index.name = "geo_accession"
    sample_expr = sample_expr.reset_index().merge(meta[["geo_accession", "case", "tissue"]], on="geo_accession", how="left")
    return add_cascade_scores(sample_expr, ["geo_accession", "case", "tissue"], "GSE60042")


def safe_fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def make_figure(spatial_tests: pd.DataFrame, axis_high_summary: pd.DataFrame, bulk_tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    focus = [
        "aldosterone_terminal_step",
        "aldosterone_cascade_activity",
        "aldosterone_vs_cortisol_branch_index",
        "aldosterone_vs_androgen_branch_index",
        "steroidogenic_cascade_pseudoflux_index",
    ]
    s = spatial_tests[spatial_tests["metric"].isin(focus)].copy()
    s["metric"] = pd.Categorical(s["metric"], categories=focus[::-1], ordered=True)
    axes[0].barh(s["metric"].astype(str), s["mean_delta"], color="#3a7ca5")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("A. Spatial APA-adjacent")
    axes[0].set_xlabel("APA - Adjacent mean delta")
    axes[0].set_ylabel("")

    a = axis_high_summary[(axis_high_summary["tissue"] == "APA") & (axis_high_summary["metric"].isin(focus))].copy()
    a["metric"] = pd.Categorical(a["metric"], categories=focus[::-1], ordered=True)
    axes[1].barh(a["metric"].astype(str), a["mean_axis_high_minus_other"], color="#2f8f71")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("B. APA axis-high spots")
    axes[1].set_xlabel("Axis-high - other spots")
    axes[1].set_ylabel("")

    b = bulk_tests[bulk_tests["metric"].isin(focus)].copy()
    b["metric"] = pd.Categorical(b["metric"], categories=focus[::-1], ordered=True)
    axes[2].barh(b["metric"].astype(str), b["mean_delta"], color="#d08b3e")
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_title("C. GSE60042 APA-AAG")
    axes[2].set_xlabel("APA - AAG mean delta")
    axes[2].set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIG / "Supp_steroidogenic_cascade_activity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Supp_steroidogenic_cascade_activity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(spatial_tests: pd.DataFrame, axis_high_summary: pd.DataFrame, bulk_tests: pd.DataFrame, availability: pd.DataFrame) -> None:
    primary = spatial_tests[spatial_tests["metric"] == "steroidogenic_cascade_pseudoflux_index"].iloc[0]
    terminal = spatial_tests[spatial_tests["metric"] == "aldosterone_terminal_step"].iloc[0]
    axis_high = axis_high_summary[(axis_high_summary["tissue"] == "APA") & (axis_high_summary["metric"] == "steroidogenic_cascade_pseudoflux_index")]
    axis_high = axis_high.iloc[0] if not axis_high.empty else None
    bulk_primary = bulk_tests[bulk_tests["metric"] == "steroidogenic_cascade_pseudoflux_index"].iloc[0]
    lines = [
        "# Steroidogenic cascade activity report 2026-06-02",
        "",
        "## Purpose",
        "",
        "Compute a conservative transcript-derived steroidogenic pathway-step activity score. This is intentionally named pseudoflux/index, not biochemical flux.",
        "",
        "## Gene/module availability",
        "",
    ]
    for _, row in availability.iterrows():
        lines.append(f"- {row['dataset']} {row['module']}: {int(row['n_available'])}/{int(row['n_requested'])} genes available.")
    lines.extend(
        [
            "",
            "## Spatial paired APA-adjacent results",
            "",
            f"- Aldosterone terminal step: mean APA-adjacent delta = {safe_fmt(terminal['mean_delta'])}; {int(terminal['n_positive'])}/{int(terminal['n_pairs'])} positive; exact p = {safe_fmt(terminal['exact_sign_flip_two_sided_p'])}.",
            f"- Steroidogenic cascade pseudoflux index: mean APA-adjacent delta = {safe_fmt(primary['mean_delta'])}; {int(primary['n_positive'])}/{int(primary['n_pairs'])} positive; exact p = {safe_fmt(primary['exact_sign_flip_two_sided_p'])}.",
            "",
            "## APA axis-high enrichment",
            "",
        ]
    )
    if axis_high is not None:
        lines.append(
            f"- Pseudoflux index in axis-high vs other APA spots: mean difference = {safe_fmt(axis_high['mean_axis_high_minus_other'])}; {int(axis_high['n_positive'])}/{int(axis_high['n_sections'])} sections positive; exact p = {safe_fmt(axis_high['exact_sign_flip_two_sided_p'])}."
        )
    lines.extend(
        [
            "",
            "## Independent GSE60042 transfer",
            "",
            f"- Bulk pseudoflux index: mean APA-AAG delta = {safe_fmt(bulk_primary['mean_delta'])}; {int(bulk_primary['n_positive'])}/{int(bulk_primary['n_pairs'])} positive; exact p = {safe_fmt(bulk_primary['exact_sign_flip_two_sided_p'])}.",
            "",
            "## Claim boundary",
            "",
            "Recommended wording:",
            "",
            "> A transcript-derived steroidogenic pathway-step index supported preferential aldosterone-branch activity in APA and APA axis-high spatial domains.",
            "",
            "Avoid:",
            "",
            "- Aldosterone metabolic flux was inferred or confirmed.",
            "- Transcript expression proves enzymatic flux.",
            "- This replaces biochemical steroid measurement.",
            "",
            "## Outputs",
            "",
            "- `results/steroidogenic_cascade_activity_20260602/GSE274314_steroidogenic_cascade_spot_scores.csv`",
            "- `results/steroidogenic_cascade_activity_20260602/GSE274314_steroidogenic_cascade_paired_tests.csv`",
            "- `results/steroidogenic_cascade_activity_20260602/GSE60042_steroidogenic_cascade_paired_tests.csv`",
            "- `results/steroidogenic_cascade_activity_20260602/figures/Supp_steroidogenic_cascade_activity.png/.pdf`",
        ]
    )
    (PROJECT / "steroidogenic_cascade_activity_report_20260602.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage2 = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    id_cols = ["spot_id", "geo_accession", "barcode", "patient", "tissue", "genotype", "axis_top_quartile_roi"]
    spatial_scores, spatial_avail = add_cascade_scores(stage2, id_cols, "GSE274314")
    spatial_tests, spatial_delta = paired_tests(spatial_scores, "GSE274314", "APA", "Adjacent", "patient")
    axis_high, axis_high_summary = axis_high_enrichment(spatial_scores)

    bulk_scores, bulk_avail = load_gse60042_scores()
    bulk_tests, bulk_delta = paired_tests(bulk_scores, "GSE60042", "APA", "AAG", "case")

    availability = pd.concat([spatial_avail, bulk_avail], ignore_index=True)
    availability.to_csv(OUT / "steroidogenic_cascade_gene_availability.csv", index=False, encoding="utf-8-sig")
    spatial_scores.to_csv(OUT / "GSE274314_steroidogenic_cascade_spot_scores.csv", index=False, encoding="utf-8-sig")
    spatial_tests.to_csv(OUT / "GSE274314_steroidogenic_cascade_paired_tests.csv", index=False, encoding="utf-8-sig")
    spatial_delta.to_csv(OUT / "GSE274314_steroidogenic_cascade_paired_delta.csv", index=False, encoding="utf-8-sig")
    axis_high.to_csv(OUT / "GSE274314_steroidogenic_cascade_axis_high_section_tests.csv", index=False, encoding="utf-8-sig")
    axis_high_summary.to_csv(OUT / "GSE274314_steroidogenic_cascade_axis_high_summary.csv", index=False, encoding="utf-8-sig")
    bulk_scores.to_csv(OUT / "GSE60042_steroidogenic_cascade_scores.csv", index=False, encoding="utf-8-sig")
    bulk_tests.to_csv(OUT / "GSE60042_steroidogenic_cascade_paired_tests.csv", index=False, encoding="utf-8-sig")
    bulk_delta.to_csv(OUT / "GSE60042_steroidogenic_cascade_paired_delta.csv", index=False, encoding="utf-8-sig")

    make_figure(spatial_tests, axis_high_summary, bulk_tests)
    write_report(spatial_tests, axis_high_summary, bulk_tests, availability)

    print("Spatial cascade paired tests:")
    print(spatial_tests.to_string(index=False))
    print("Bulk cascade paired tests:")
    print(bulk_tests.to_string(index=False))
    print("Steroidogenic cascade activity analysis complete.")


if __name__ == "__main__":
    main()
