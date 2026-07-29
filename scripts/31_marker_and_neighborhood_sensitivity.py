from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
STAGE2 = RESULTS / "stage2_zonation_axis"
NICHE = RESULTS / "spatial_niche_neighborhood_20260602"
OUT = RESULTS / "submission_hardening_20260603"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ink": "#17212B",
    "muted": "#68717A",
    "grid": "#D7DDE2",
    "apa": "#D55E00",
    "adjacent": "#0072B2",
    "zg": "#009E73",
    "zf": "#785EF0",
    "zr": "#CC79A7",
    "gold": "#E6A000",
    "teal": "#006D77",
    "gray": "#A7B0B8",
}


SIGNATURES = {
    "ZG_aldosterone_program": [
        "CYP11B2",
        "HSD3B2",
        "AGTR1",
        "VSNL1",
        "DACH1",
        "LGR5",
        "ANO4",
        "PDE2A",
        "NR4A1",
        "NR4A2",
        "CACNA1D",
        "CACNA1H",
        "ATP2B3",
    ],
    "ZF_cortisol_program": [
        "CYP11B1",
        "CYP17A1",
        "MC2R",
        "SULT2A1",
        "CYB5A",
        "PAPSS2",
        "AKR1B1",
        "PDE8B",
    ],
    "ZR_androgen_program": [
        "CYP17A1",
        "CYB5A",
        "SULT2A1",
        "SULT2B1",
        "PAPSS2",
        "AKR1C3",
        "HSD17B2",
        "HSD17B6",
    ],
    "intermediate_steroidogenic_program": [
        "CYP11A1",
        "STAR",
        "HSD3B2",
        "CYP21A2",
        "MC2R",
        "FDX1",
        "FDXR",
        "SCARB1",
        "TSPO",
        "NR5A1",
    ],
    "generic_steroidogenesis_program": [
        "STAR",
        "CYP11A1",
        "HSD3B2",
        "CYP21A2",
        "FDX1",
        "FDXR",
        "SCARB1",
        "TSPO",
        "NR5A1",
    ],
}

CASCADE_MODULES = {
    "cholesterol_import_initiation": ["SCARB1", "STAR", "CYP11A1"],
    "pregnenolone_to_DOC_branch": ["HSD3B2", "CYP21A2"],
    "aldosterone_terminal_step": ["CYP11B2"],
    "cortisol_terminal_branch": ["CYP11B1", "CYP17A1"],
    "androgen_ZR_diversion_branch": ["CYP17A1", "CYB5A", "SULT2A1", "SULT2B1"],
}

MARKERS_TO_DROP = ["CYP11B2", "HSD3B2", "STAR", "CYP11A1"]
PRIMARY_AXIS = "ZG_intermediate_vs_ZF_axis_score"
PATHWAY_INDEX = "steroidogenic_cascade_pseudoflux_index"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.12,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.55, alpha=0.7)


def exact_sign_flip(delta: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan, np.nan
    observed = float(delta.mean())
    vals = []
    for bits in itertools.product([-1, 1], repeat=len(delta)):
        vals.append(float((delta * np.asarray(bits)).mean()))
    vals = np.asarray(vals)
    return float(np.mean(np.abs(vals) >= abs(observed))), float(np.mean(vals >= observed))


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
    tmp = np.empty_like(vals)
    tmp[order] = np.clip(adj, 0, 1)
    out[ok] = tmp
    return pd.Series(out, index=pvalues.index)


def zscore_columns(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for gene in sorted(set(genes)):
        if gene not in df.columns:
            continue
        vals = df[gene].astype(float)
        sd = vals.std(ddof=0)
        out[gene] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return out


def module_scores(expr: pd.DataFrame, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = sorted(set(g for gene_list in signatures.values() for g in gene_list))
    z = zscore_columns(expr, genes)
    scores = pd.DataFrame(index=expr.index)
    availability = []
    for name, geneset in signatures.items():
        present = [g for g in geneset if g in z.columns]
        missing = [g for g in geneset if g not in z.columns]
        scores[name] = z[present].mean(axis=1) if present else np.nan
        availability.append(
            {
                "module": name,
                "n_requested": len(geneset),
                "n_available": len(present),
                "available_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )
    scores["ZG_intermediate_vs_ZF_axis_score"] = (
        scores[["ZG_aldosterone_program", "intermediate_steroidogenic_program"]].mean(axis=1)
        - scores["ZF_cortisol_program"]
    )
    scores["aldosterone_specificity_score"] = scores["ZG_aldosterone_program"] - scores["generic_steroidogenesis_program"]
    return scores, pd.DataFrame(availability)


def cascade_scores(expr: pd.DataFrame, modules: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = sorted(set(g for gene_list in modules.values() for g in gene_list))
    z = zscore_columns(expr, genes)
    scores = pd.DataFrame(index=expr.index)
    availability = []
    for name, geneset in modules.items():
        present = [g for g in geneset if g in z.columns]
        missing = [g for g in geneset if g not in z.columns]
        scores[name] = z[present].mean(axis=1) if present else np.nan
        availability.append(
            {
                "module": name,
                "n_requested": len(geneset),
                "n_available": len(present),
                "available_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )
    aldosterone_parts = [
        c
        for c in ["cholesterol_import_initiation", "pregnenolone_to_DOC_branch", "aldosterone_terminal_step"]
        if c in scores.columns and scores[c].notna().any()
    ]
    scores["aldosterone_cascade_activity"] = scores[aldosterone_parts].mean(axis=1) if aldosterone_parts else np.nan
    scores["aldosterone_vs_cortisol_branch_index"] = scores["aldosterone_terminal_step"] - scores["cortisol_terminal_branch"] if scores["aldosterone_terminal_step"].notna().any() else np.nan
    scores["aldosterone_vs_androgen_branch_index"] = scores["aldosterone_terminal_step"] - scores["androgen_ZR_diversion_branch"] if scores["aldosterone_terminal_step"].notna().any() else np.nan
    scores["steroidogenic_cascade_pseudoflux_index"] = scores["aldosterone_cascade_activity"] - scores[
        ["cortisol_terminal_branch", "androgen_ZR_diversion_branch"]
    ].mean(axis=1)
    return scores, pd.DataFrame(availability)


def paired_summary(section_scores: pd.DataFrame, metric: str, value_col: str, variant: str) -> dict:
    wide = section_scores.pivot_table(index="patient", columns="tissue", values=value_col, aggfunc="mean")
    wide = wide.dropna(subset=["APA", "Adjacent"], how="any")
    delta = wide["APA"] - wide["Adjacent"]
    exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
    ci_low, ci_high = (np.nan, np.nan)
    if len(delta) > 1:
        ci_low, ci_high = np.percentile(delta, [2.5, 97.5])
    return {
        "variant": variant,
        "metric": metric,
        "n_pairs": int(delta.shape[0]),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "ci95_low_empirical": float(ci_low),
        "ci95_high_empirical": float(ci_high),
        "n_positive": int((delta > 0).sum()),
        "n_negative": int((delta < 0).sum()),
        "exact_sign_flip_two_sided_p": exact_two,
        "exact_sign_flip_greater_p": exact_greater,
        "wilcoxon_p": stats.wilcoxon(delta).pvalue if len(delta) > 1 and np.all(delta != 0) else np.nan,
    }


def compute_marker_sensitivity(spots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_cols = ["geo_accession", "patient", "tissue", "genotype"]
    variants = [("Original signature", None), ("CYP11B2-free", "CYP11B2")] + [
        (f"Drop {gene}", gene) for gene in MARKERS_TO_DROP if gene != "CYP11B2"
    ]
    axis_rows = []
    pathway_rows = []
    avail_rows = []
    for variant_label, drop_gene in variants:
        sig = {
            name: [g for g in genes if g != drop_gene]
            for name, genes in SIGNATURES.items()
        }
        axis_scores, axis_avail = module_scores(spots, sig)
        axis_df = pd.concat([spots[id_cols].reset_index(drop=True), axis_scores.reset_index(drop=True)], axis=1)
        section_axis = axis_df.groupby(id_cols, dropna=False)[PRIMARY_AXIS].mean().reset_index()
        axis_rows.append(paired_summary(section_axis, PRIMARY_AXIS, PRIMARY_AXIS, variant_label))
        axis_avail["variant"] = variant_label
        axis_avail["score_family"] = "zonation_axis"
        avail_rows.append(axis_avail)

        cascade = {
            name: [g for g in genes if g != drop_gene]
            for name, genes in CASCADE_MODULES.items()
        }
        pathway_scores, path_avail = cascade_scores(spots, cascade)
        path_df = pd.concat([spots[id_cols].reset_index(drop=True), pathway_scores.reset_index(drop=True)], axis=1)
        section_path = path_df.groupby(id_cols, dropna=False)[PATHWAY_INDEX].mean().reset_index()
        pathway_rows.append(paired_summary(section_path, PATHWAY_INDEX, PATHWAY_INDEX, variant_label))
        path_avail["variant"] = variant_label
        path_avail["score_family"] = "pathway_step_index"
        avail_rows.append(path_avail)

    axis_out = pd.DataFrame(axis_rows)
    path_out = pd.DataFrame(pathway_rows)
    for df in [axis_out, path_out]:
        df["exact_sign_flip_two_sided_fdr"] = bh_fdr(df["exact_sign_flip_two_sided_p"])
    availability = pd.concat(avail_rows, ignore_index=True)
    return axis_out, path_out, availability


def compute_gene_overlap() -> pd.DataFrame:
    axis_positive = set(SIGNATURES["ZG_aldosterone_program"]) | set(SIGNATURES["intermediate_steroidogenic_program"])
    axis_negative = set(SIGNATURES["ZF_cortisol_program"])
    axis_all = axis_positive | axis_negative
    pathway_all = set(g for genes in CASCADE_MODULES.values() for g in genes)
    rows = []
    comparisons = [
        ("primary_axis_all_genes", axis_all, "pathway_step_index_all_genes", pathway_all),
        ("primary_axis_positive_arm", axis_positive, "pathway_step_index_all_genes", pathway_all),
        ("primary_axis_negative_ZF_arm", axis_negative, "pathway_step_index_all_genes", pathway_all),
        ("CYP11B2_free_primary_axis_all_genes", axis_all - {"CYP11B2"}, "CYP11B2_free_pathway_index_genes", pathway_all - {"CYP11B2"}),
    ]
    for a_name, a, b_name, b in comparisons:
        inter = sorted(a & b)
        union = sorted(a | b)
        rows.append(
            {
                "set_a": a_name,
                "set_b": b_name,
                "n_set_a": len(a),
                "n_set_b": len(b),
                "n_overlap": len(inter),
                "jaccard": len(inter) / len(union) if union else np.nan,
                "overlap_genes": ";".join(inter),
                "set_a_genes": ";".join(sorted(a)),
                "set_b_genes": ";".join(sorted(b)),
            }
        )
    return pd.DataFrame(rows)


def add_distance_to_edge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_edge"] = np.nan
    for _, idx in df.groupby("geo_accession").groups.items():
        idx = list(idx)
        sub = df.loc[idx]
        x = sub["lowres_x"].astype(float)
        y = sub["lowres_y"].astype(float)
        d = np.minimum.reduce([x - x.min(), x.max() - x, y - y.min(), y.max() - y])
        scale = np.nanmedian(sub["mean_neighbor_distance"].astype(float))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        df.loc[idx, "distance_to_edge"] = d / scale
    return df


def ols_beta_pvalue(y: pd.Series, x: pd.DataFrame, target: str) -> dict:
    data = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < x.shape[1] + 15 or data["y"].std(ddof=0) == 0:
        return {"n_spots": int(data.shape[0]), "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "partial_r": np.nan}
    y_arr = data["y"].to_numpy(float)
    x_arr = data[x.columns].to_numpy(float)
    x_mean = x_arr.mean(axis=0)
    x_sd = x_arr.std(axis=0)
    x_sd[x_sd == 0] = 1.0
    x_arr = (x_arr - x_mean) / x_sd
    y_arr = (y_arr - y_arr.mean()) / (y_arr.std(ddof=0) or 1.0)
    design = np.column_stack([np.ones(len(y_arr)), x_arr])
    try:
        inv = np.linalg.pinv(design.T @ design)
        beta = inv @ design.T @ y_arr
        pred = design @ beta
        resid = y_arr - pred
        df_resid = len(y_arr) - design.shape[1]
        sigma2 = float((resid @ resid) / df_resid)
        se = np.sqrt(np.diag(inv) * sigma2)
        target_pos = list(x.columns).index(target) + 1
        t_val = beta[target_pos] / se[target_pos] if se[target_pos] > 0 else np.nan
        p_val = 2 * stats.t.sf(abs(t_val), df_resid) if np.isfinite(t_val) else np.nan
        partial_r = t_val / np.sqrt(t_val**2 + df_resid) if np.isfinite(t_val) else np.nan
    except np.linalg.LinAlgError:
        return {"n_spots": int(data.shape[0]), "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "partial_r": np.nan}
    return {
        "n_spots": int(data.shape[0]),
        "beta": float(beta[target_pos]),
        "se": float(se[target_pos]),
        "t": float(t_val),
        "p": float(p_val),
        "partial_r": float(partial_r),
    }


def adjusted_neighborhood_models(niche: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = add_distance_to_edge(niche)
    predictors = [
        "neighbor_wnt_rspo_module",
        "neighbor_fibroblast_stromal",
        "WNT4_to_FZD_neighbor_product",
    ]
    covars = ["log10_total_umi", "n_detected_genes", "distance_to_edge", "fibroblast_stromal"]
    rows = []
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        for predictor in predictors:
            if predictor not in sub.columns:
                continue
            cols = [predictor] + [c for c in covars if c in sub.columns]
            res = ols_beta_pvalue(sub[PRIMARY_AXIS].astype(float), sub[cols].astype(float), predictor)
            rows.append(
                {
                    "geo_accession": gsm,
                    "patient": int(patient),
                    "tissue": tissue,
                    "predictor": predictor,
                    "adjusted_for": ";".join([c for c in covars if c in sub.columns]),
                    **res,
                }
            )
    section = pd.DataFrame(rows)
    if not section.empty:
        section["p_fdr_within_analysis"] = bh_fdr(section["p"])
    summary_rows = []
    for (predictor, tissue), sub in section.groupby(["predictor", "tissue"], dropna=False):
        vals = sub["beta"].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        exact_two, exact_greater = exact_sign_flip(vals)
        summary_rows.append(
            {
                "predictor": predictor,
                "tissue": tissue,
                "n_sections": int(len(vals)),
                "mean_adjusted_beta": float(vals.mean()),
                "median_adjusted_beta": float(np.median(vals)),
                "n_positive": int((vals > 0).sum()),
                "n_negative": int((vals < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["exact_sign_flip_two_sided_fdr"] = bh_fdr(summary["exact_sign_flip_two_sided_p"])

    paired_rows = []
    for predictor in sorted(section["predictor"].dropna().unique()):
        sub = section[section["predictor"] == predictor]
        wide = sub.pivot_table(index="patient", columns="tissue", values="beta", aggfunc="mean").dropna(subset=["APA", "Adjacent"], how="any")
        delta = wide["APA"] - wide["Adjacent"]
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        paired_rows.append(
            {
                "predictor": predictor,
                "n_pairs": int(delta.shape[0]),
                "mean_APA_minus_Adjacent_adjusted_beta": float(delta.mean()),
                "median_APA_minus_Adjacent_adjusted_beta": float(delta.median()),
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta).pvalue if len(delta) > 1 and np.all(delta != 0) else np.nan,
            }
        )
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["exact_sign_flip_two_sided_fdr"] = bh_fdr(paired["exact_sign_flip_two_sided_p"])
    return section, summary, paired


def plot_ci_panel(ax: plt.Axes, data: pd.DataFrame, y_col: str, x_col: str, low_col: str, high_col: str, title: str, color: str) -> None:
    plot = data.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    for yi, (_, row) in enumerate(plot.iterrows()):
        low = row[low_col] if low_col in row and pd.notna(row[low_col]) else row[x_col]
        high = row[high_col] if high_col in row and pd.notna(row[high_col]) else row[x_col]
        ax.hlines(yi, low, high, color=color, lw=2.0, alpha=0.9)
        ax.vlines([low, high], yi - 0.08, yi + 0.08, color=color, lw=1.0, alpha=0.9)
        ax.scatter(row[x_col], yi, s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(high + 0.03, yi, f"{int(row['n_positive'])}/{int(row['n_pairs'])}", va="center", fontsize=7)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(plot[y_col])
    ax.set_title(title)
    despine(ax)


def make_figure(axis: pd.DataFrame, pathway: pd.DataFrame, overlap: pd.DataFrame, adj_summary: pd.DataFrame, specificity: pd.DataFrame) -> None:
    set_style()
    fig = plt.figure(figsize=(8.0, 7.4))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.62)

    ax1 = fig.add_subplot(gs[0, 0])
    show_axis = axis.copy()
    show_axis["label"] = show_axis["variant"].replace({"Original signature": "Original"})
    plot_ci_panel(
        ax1,
        show_axis,
        "label",
        "mean_delta",
        "ci95_low_empirical",
        "ci95_high_empirical",
        "Primary axis marker sensitivity",
        PALETTE["apa"],
    )
    ax1.set_xlabel("APA - adjacent mean delta")
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    show_path = pathway.copy()
    show_path["label"] = show_path["variant"].replace({"Original signature": "Original"})
    plot_ci_panel(
        ax2,
        show_path,
        "label",
        "mean_delta",
        "ci95_low_empirical",
        "ci95_high_empirical",
        "Pathway-step index sensitivity",
        PALETTE["gold"],
    )
    ax2.set_xlabel("APA - adjacent mean delta")
    add_panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    focus = adj_summary[adj_summary["predictor"].isin(["neighbor_wnt_rspo_module", "neighbor_fibroblast_stromal", "WNT4_to_FZD_neighbor_product"])].copy()
    label_map = {
        "neighbor_wnt_rspo_module": "Neighbor WNT/RSPO",
        "neighbor_fibroblast_stromal": "Neighbor fibroblast/stromal",
        "WNT4_to_FZD_neighbor_product": "WNT4-FZD product",
    }
    focus["label"] = focus["predictor"].map(label_map)
    focus = focus.sort_values(["label", "tissue"])
    x_offsets = {"APA": -0.05, "Adjacent": 0.05}
    colors = {"APA": PALETTE["apa"], "Adjacent": PALETTE["adjacent"]}
    labels = list(dict.fromkeys(focus["label"].tolist()))
    y_pos = {lab: i for i, lab in enumerate(labels[::-1])}
    for _, row in focus.iterrows():
        yi = y_pos[row["label"]] + x_offsets.get(row["tissue"], 0)
        ax3.scatter(row["mean_adjusted_beta"], yi, s=42, color=colors.get(row["tissue"], PALETTE["gray"]), edgecolor="white", linewidth=0.6, zorder=3)
        ax3.text(row["mean_adjusted_beta"] + 0.03, yi, f"{int(row['n_positive'])}/{int(row['n_sections'])}", va="center", fontsize=7)
    ax3.axvline(0, color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax3.set_yticks(list(y_pos.values()))
    ax3.set_yticklabels(list(y_pos.keys()))
    ax3.set_xlabel("Mean adjusted beta")
    ax3.set_title("Adjusted neighborhood coupling")
    ax3.scatter([], [], color=PALETTE["apa"], label="APA")
    ax3.scatter([], [], color=PALETTE["adjacent"], label="Adjacent")
    ax3.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax3)
    add_panel_label(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    spec = specificity.copy().iloc[::-1].reset_index(drop=True)
    spec["label"] = spec["predictor"].map(label_map)
    y = np.arange(len(spec))
    ax4.scatter(spec["mean_APA_minus_Adjacent_adjusted_beta"], y, s=46, color=PALETTE["teal"], edgecolor="white", linewidth=0.6, zorder=3)
    for yi, (_, row) in enumerate(spec.iterrows()):
        ax4.text(row["mean_APA_minus_Adjacent_adjusted_beta"] + 0.025, yi, f"{int(row['n_positive'])}/{int(row['n_pairs'])}", va="center", fontsize=7)
    ax4.axvline(0, color=PALETTE["ink"], lw=0.8, linestyle=(0, (3, 2)))
    ax4.set_yticks(y)
    ax4.set_yticklabels(spec["label"])
    ax4.set_xlabel("APA - adjacent adjusted beta")
    ax4.set_title("Coupling specificity")
    despine(ax4)
    add_panel_label(ax4, "D")

    fig.savefig(FIG / "Supp_submission_hardening_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Supp_submission_hardening_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # A compact, table-like overlap figure is more readable as CSV/Word table; still save a one-panel summary.
    fig2, ax = plt.subplots(figsize=(4.6, 2.4))
    ov = overlap.iloc[:3].copy()
    ax.barh(np.arange(len(ov)), ov["jaccard"], color=[PALETTE["teal"], PALETTE["apa"], PALETTE["adjacent"]])
    ax.set_yticks(np.arange(len(ov)))
    ax.set_yticklabels(ov["set_a"].str.replace("_", " "))
    ax.set_xlabel("Jaccard with pathway-step genes")
    ax.set_title("Axis/pathway gene-set overlap")
    despine(ax)
    fig2.savefig(FIG / "Supp_axis_pathway_gene_overlap.png", dpi=600, bbox_inches="tight")
    fig2.savefig(FIG / "Supp_axis_pathway_gene_overlap.pdf", bbox_inches="tight")
    plt.close(fig2)


def write_report(axis: pd.DataFrame, pathway: pd.DataFrame, overlap: pd.DataFrame, adj_summary: pd.DataFrame, specificity: pd.DataFrame) -> None:
    cyp_free = axis[axis["variant"] == "CYP11B2-free"].iloc[0]
    path_free = pathway[pathway["variant"] == "CYP11B2-free"].iloc[0]
    wnt_apa = adj_summary[(adj_summary["predictor"] == "neighbor_wnt_rspo_module") & (adj_summary["tissue"] == "APA")]
    wnt_apa = wnt_apa.iloc[0] if not wnt_apa.empty else None
    wnt_spec = specificity[specificity["predictor"] == "neighbor_wnt_rspo_module"]
    wnt_spec = wnt_spec.iloc[0] if not wnt_spec.empty else None
    main_overlap = overlap.iloc[0]
    lines = [
        "# Submission hardening sensitivity report 2026-06-03",
        "",
        "## Defensive-analysis status",
        "",
        f"- CYP11B2-free primary axis: mean APA-adjacent delta = {cyp_free['mean_delta']:.4g}; {int(cyp_free['n_positive'])}/{int(cyp_free['n_pairs'])} positive; exact p = {cyp_free['exact_sign_flip_two_sided_p']:.4g}.",
        f"- CYP11B2-free pathway-step index: mean APA-adjacent delta = {path_free['mean_delta']:.4g}; {int(path_free['n_positive'])}/{int(path_free['n_pairs'])} positive; exact p = {path_free['exact_sign_flip_two_sided_p']:.4g}. This variant omits the single-gene terminal CYP11B2 step and should be described as a CYP11B2-free pathway-step sensitivity index.",
        f"- Primary axis vs pathway-step gene overlap: {int(main_overlap['n_overlap'])} overlapping genes; Jaccard = {main_overlap['jaccard']:.3g}.",
    ]
    if wnt_apa is not None:
        lines.append(
            f"- Adjusted APA neighbor WNT/RSPO coupling: mean adjusted beta = {wnt_apa['mean_adjusted_beta']:.4g}; "
            f"{int(wnt_apa['n_positive'])}/{int(wnt_apa['n_sections'])} sections positive; exact p = {wnt_apa['exact_sign_flip_two_sided_p']:.4g}."
        )
    if wnt_spec is not None:
        lines.append(
            f"- APA-vs-adjacent WNT/RSPO coupling specificity: mean paired adjusted-beta delta = {wnt_spec['mean_APA_minus_Adjacent_adjusted_beta']:.4g}; "
            f"{int(wnt_spec['n_positive'])}/{int(wnt_spec['n_pairs'])} pairs positive; exact p = {wnt_spec['exact_sign_flip_two_sided_p']:.4g}."
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- These analyses strengthen robustness against single-marker and local-neighborhood confounding critiques.",
        "- The adjusted neighborhood models support spatial coupling, not causal WNT/RSPO or stromal signaling.",
        "- The pathway-step sensitivity is transcript-derived and should not be described as biochemical flux.",
        "",
        "## Output files",
        "",
        "- `results/submission_hardening_20260603/submission_hardening_axis_marker_sensitivity.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_pathway_marker_sensitivity.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_marker_availability.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_axis_pathway_gene_overlap.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_adjusted_neighborhood_section_betas.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_adjusted_neighborhood_summary.csv`",
        "- `results/submission_hardening_20260603/submission_hardening_adjusted_neighborhood_apa_adjacent_specificity.csv`",
        "- `results/submission_hardening_20260603/figures/Supp_submission_hardening_sensitivity.png/.pdf`",
        "- `results/submission_hardening_20260603/figures/Supp_axis_pathway_gene_overlap.png/.pdf`",
    ]
    report_text = "\n".join(lines) + "\n"
    (PROJECT / "submission_hardening_report_20260603.md").write_text(report_text, encoding="utf-8")
    (OUT / "submission_hardening_report_20260603.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    spots = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    niche = pd.read_csv(NICHE / "GSE274314_spatial_niche_spot_level.csv")
    axis, pathway, marker_avail = compute_marker_sensitivity(spots)
    overlap = compute_gene_overlap()
    section_betas, adj_summary, specificity = adjusted_neighborhood_models(niche)

    axis.to_csv(OUT / "submission_hardening_axis_marker_sensitivity.csv", index=False, encoding="utf-8-sig")
    pathway.to_csv(OUT / "submission_hardening_pathway_marker_sensitivity.csv", index=False, encoding="utf-8-sig")
    marker_avail.to_csv(OUT / "submission_hardening_marker_availability.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(OUT / "submission_hardening_axis_pathway_gene_overlap.csv", index=False, encoding="utf-8-sig")
    section_betas.to_csv(OUT / "submission_hardening_adjusted_neighborhood_section_betas.csv", index=False, encoding="utf-8-sig")
    adj_summary.to_csv(OUT / "submission_hardening_adjusted_neighborhood_summary.csv", index=False, encoding="utf-8-sig")
    specificity.to_csv(OUT / "submission_hardening_adjusted_neighborhood_apa_adjacent_specificity.csv", index=False, encoding="utf-8-sig")

    make_figure(axis, pathway, overlap, adj_summary, specificity)
    write_report(axis, pathway, overlap, adj_summary, specificity)

    print("Submission hardening sensitivity analyses complete.")
    print(OUT)


if __name__ == "__main__":
    main()
