from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmread
from scipy.optimize import nnls


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
RESULTS = PROJECT / "results"
OUT_DIR = RESULTS / "seurat_reference"
FIGURES = RESULTS / "figures"

PANELS = {
    "cortical_steroidogenic": ["CYP11A1", "STAR", "HSD3B2", "CYP21A2", "CYP11B1", "CYP17A1", "CYP11B2"],
    "ZG_like": ["CYP11B2", "AGTR1", "KCNJ5", "DACH1", "LGR5", "VSNL1", "ANO4", "PDE2A"],
    "ZF_ZR_like": ["CYP11B1", "CYP17A1", "MC2R", "SULT2A1", "CYB5A", "PAPSS2"],
    "endothelial": ["PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"],
    "fibroblast_stromal": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "immune": ["PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1"],
    "smooth_muscle_pericyte": ["ACTA2", "MYH11", "RGS5", "MCAM", "PDGFRB", "CSPG4"],
    "medulla_chromaffin": ["TH", "DBH", "PNMT", "CHGA", "CHGB", "SLC18A1"],
}

DECONV_GENES = sorted(
    set(sum(PANELS.values(), []))
    | {
        "CYP11A1",
        "CYP11B2",
        "CYP11B1",
        "CYP17A1",
        "HSD3B2",
        "STAR",
        "CYP21A2",
        "SULT2A1",
        "CYB5A",
        "PAPSS2",
        "COL4A1",
        "COL4A2",
        "TAGLN",
        "ADGRL4",
        "LYVE1",
    }
)


def load_cluster_utils():
    script = PROJECT / "scripts" / "08_cluster_annotation_deconvolution.py"
    spec = importlib.util.spec_from_file_location("cluster_utils", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_reference_expression(utils) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    X, meta, genes = utils.load_aligned_matrices()
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    detected_genes = X.getnnz(axis=1)
    mt_idx = np.array([i for i, gene in enumerate(genes) if gene.startswith("MT-")], dtype=int)
    if len(mt_idx):
        mt_counts = np.asarray(X[:, mt_idx].sum(axis=1)).ravel()
    else:
        mt_counts = np.zeros(X.shape[0])
    pct_mt = np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts > 0) * 100
    keep = (total_counts >= 500) & (detected_genes >= 300) & (pct_mt <= 20)
    Xf = X[keep].tocsr()
    mf = meta.loc[keep].reset_index(drop=True).copy()
    mf["total_counts"] = total_counts[keep]
    mf["detected_genes"] = detected_genes[keep]
    mf["pct_mt"] = pct_mt[keep]

    ann = pd.read_csv(OUT_DIR / "seurat_cell_cluster_annotations.csv", usecols=["cell_id", "deconv_label", "seurat_cluster"])
    mf = mf.merge(ann, on="cell_id", how="inner")
    if len(mf) != Xf.shape[0]:
        cell_to_row = {cell_id: i for i, cell_id in enumerate(meta.loc[keep, "cell_id"])}
        idx = [cell_to_row[c] for c in mf["cell_id"]]
        Xf = Xf[idx].tocsr()
    return Xf, mf, genes


def lognorm_selected_genes(X: sparse.csr_matrix, total_counts: np.ndarray) -> sparse.csr_matrix:
    scale = np.divide(1e4, total_counts, out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0)
    Xn = sparse.diags(scale).dot(X).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn


def build_profiles(
    Xlog: sparse.csr_matrix,
    meta: pd.DataFrame,
    genes: list[str],
    scenario: str,
    mask: np.ndarray,
    min_cells: int = 20,
) -> pd.DataFrame:
    labels = ["ZG_like", "ZF_ZR_like", "intermediate_steroidogenic", "endothelial", "fibroblast_stromal", "immune", "smooth_muscle_pericyte", "medulla_chromaffin"]
    rows = []
    for label in labels:
        label_mask = mask & (meta["deconv_label"].to_numpy() == label)
        n_cells = int(label_mask.sum())
        if n_cells < min_cells:
            continue
        values = np.asarray(Xlog[label_mask].mean(axis=0)).ravel()
        row = {"scenario": scenario, "label": label, "n_reference_cells": n_cells}
        row.update({gene: val for gene, val in zip(genes, values)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_spatial_expression(genes: list[str], utils) -> list[dict[str, object]]:
    meta = utils.parse_series_matrix_metadata(RAW / "GSE274314" / "GSE274314_series_matrix.txt.gz")
    meta_by_gsm = meta.set_index("geo_accession").to_dict(orient="index")
    samples = []
    tar_path = RAW / "GSE274314" / "GSE274314_RAW.tar"
    with tarfile.open(tar_path, "r") as tar:
        names = tar.getnames()
        prefixes = sorted({name.replace("_matrix.mtx.gz", "") for name in names if name.endswith("_matrix.mtx.gz")})
        for prefix in prefixes:
            gsm = prefix.split("_", 1)[0]
            mat = mmread(io.BytesIO(utils.read_tar_gz_bytes(tar, f"{prefix}_matrix.mtx.gz"))).tocsc()
            features = utils.read_features_text(utils.read_tar_gz_text(tar, f"{prefix}_features.tsv.gz"))
            barcodes = utils.read_tar_gz_text(tar, f"{prefix}_barcodes.tsv.gz").splitlines()
            positions = utils.read_positions_text(utils.read_tar_gz_text(tar, f"{prefix}_tissue_positions_list.csv.gz"))
            gene_symbols = features["gene_symbol"].astype(str).to_numpy()
            total = np.asarray(mat.sum(axis=0)).ravel().astype(float)
            total[total <= 0] = np.nan
            expr_cols: dict[str, np.ndarray] = {}
            for gene in genes:
                idx = np.where(gene_symbols == gene)[0]
                if len(idx):
                    counts = np.asarray(mat[idx, :].sum(axis=0)).ravel().astype(float)
                    expr_cols[gene] = np.log1p(counts / total * 1e4)
                else:
                    expr_cols[gene] = np.zeros(mat.shape[1])
            expr = pd.DataFrame(expr_cols)
            expr["barcode"] = barcodes
            tissue_barcodes = set(positions.loc[positions["in_tissue"].astype(int) == 1, "barcode"].astype(str))
            expr = expr.loc[expr["barcode"].isin(tissue_barcodes)].reset_index(drop=True)
            sample_meta = meta_by_gsm[gsm]
            samples.append({"sample_meta": sample_meta, "expr": expr})
    return samples


def deconvolve_scenario(profile: pd.DataFrame, genes: list[str], spatial_samples: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = profile["scenario"].iloc[0]
    labels = profile["label"].tolist()
    ref = profile[genes].to_numpy(dtype=float).T
    sample_means = []
    for sample in spatial_samples:
        expr = sample["expr"]
        x = expr[genes].to_numpy(dtype=float)
        props = []
        for i in range(x.shape[0]):
            coef, _ = nnls(ref, x[i, :])
            if coef.sum() > 0:
                coef = coef / coef.sum()
            props.append(coef)
        prop_df = pd.DataFrame(props, columns=labels)
        mean_props = prop_df.mean(axis=0).to_dict()
        row = {"scenario": scenario}
        row.update({f"prop_{label}": mean_props[label] for label in labels})
        for key in ["geo_accession", "title", "patient", "tissue", "genotype"]:
            row[key] = sample["sample_meta"].get(key, "")
        sample_means.append(row)
    sample_df = pd.DataFrame(sample_means)

    paired_rows = []
    test_rows = []
    for label in labels:
        col = f"prop_{label}"
        pivot = sample_df.pivot_table(index="patient", columns="tissue", values=col, aggfunc="mean")
        genotype = sample_df.loc[sample_df["tissue"] == "APA", ["patient", "genotype"]].drop_duplicates().set_index("patient")["genotype"]
        if "APA" not in pivot or "Adjacent" not in pivot:
            continue
        paired_values = pivot[["APA", "Adjacent"]].dropna()
        delta = paired_values["APA"] - paired_values["Adjacent"]
        _stat, p = stats.ttest_rel(paired_values["APA"], paired_values["Adjacent"]) if len(delta) >= 2 else (np.nan, np.nan)
        n_positive = int((delta > 0).sum())
        n_negative = int((delta < 0).sum())
        sign_p = stats.binomtest(n_positive, len(delta), 0.5, alternative="greater").pvalue if len(delta) else np.nan
        test_rows.append(
            {
                "scenario": scenario,
                "label": label,
                "n_pairs": len(delta),
                "mean_delta_APA_minus_adjacent": delta.mean(),
                "median_delta_APA_minus_adjacent": delta.median(),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "paired_t_p": p,
                "one_sided_sign_p_APA_gt_adjacent": sign_p,
            }
        )
        for patient, value in delta.items():
            paired_rows.append(
                {
                    "scenario": scenario,
                    "patient": patient,
                    "genotype": genotype.get(patient, ""),
                    "label": label,
                    "APA_minus_adjacent": value,
                    "APA_mean": pivot.loc[patient, "APA"],
                    "Adjacent_mean": pivot.loc[patient, "Adjacent"],
                }
            )
    return pd.DataFrame(paired_rows), pd.DataFrame(test_rows)


def plot_sensitivity(tests: pd.DataFrame) -> None:
    focus = tests.loc[tests["label"].isin(["ZG_like", "intermediate_steroidogenic", "ZF_ZR_like"])].copy()
    order = [
        "all_cells",
        "normal_only",
        "all_minus_NAd4",
        "normal_minus_NAd4",
        "normal_leaveout_NAd1",
        "normal_leaveout_NAd2",
        "normal_leaveout_NAd3",
        "normal_leaveout_NAd4",
        "APCC_only",
    ]
    focus["scenario"] = pd.Categorical(focus["scenario"], categories=order, ordered=True)
    labels = ["ZG_like", "intermediate_steroidogenic", "ZF_ZR_like"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 7), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = focus.loc[focus["label"] == label].sort_values("scenario")
        colors = ["#2ca02c" if p < 0.05 and d > 0 else "#777777" for p, d in zip(sub["paired_t_p"], sub["mean_delta_APA_minus_adjacent"])]
        ax.bar(sub["scenario"].astype(str), sub["mean_delta_APA_minus_adjacent"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label.replace("_", "\n"), rotation=0, labelpad=55, va="center")
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, row["mean_delta_APA_minus_adjacent"], f"{int(row['n_positive'])}/7", ha="center", va="bottom" if row["mean_delta_APA_minus_adjacent"] >= 0 else "top", fontsize=7)
    axes[-1].tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("Seurat-reference deconvolution sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "GSE274314_seurat_reference_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], n: int = 30) -> list[str]:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.head(n).iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
        lines.append("|" + "|".join(vals) + "|")
    return lines


def write_report(profile_counts: pd.DataFrame, tests: pd.DataFrame) -> None:
    focus = tests.loc[tests["label"].isin(["ZG_like", "intermediate_steroidogenic", "ZF_ZR_like"])].copy()
    focus = focus.sort_values(["scenario", "label"])
    zg = tests.loc[tests["label"] == "ZG_like"].sort_values("scenario")
    lines = [
        "# Seurat reference sensitivity analysis",
        "",
        "Date: 2026-06-01",
        "",
        "Status: PASS for internal reference sensitivity. This is still NNLS, not RCTD/SPOTlight/CARD.",
        "",
        "## Reference scenarios",
        "",
        *md_table(profile_counts, ["scenario", "label", "n_reference_cells"], n=80),
        "",
        "## Focused paired tests",
        "",
        *md_table(focus, ["scenario", "label", "n_pairs", "mean_delta_APA_minus_adjacent", "n_positive", "n_negative", "paired_t_p", "one_sided_sign_p_APA_gt_adjacent"], n=80),
        "",
        "## ZG-like sensitivity summary",
        "",
        *md_table(zg, ["scenario", "n_pairs", "mean_delta_APA_minus_adjacent", "n_positive", "n_negative", "paired_t_p", "one_sided_sign_p_APA_gt_adjacent"], n=20),
        "",
        "## Interpretation boundary",
        "",
        "This analysis asks whether the Seurat `ZG_like` spatial signal depends on APCC cells or on the NAd4-heavy ZG-like reference cluster. If the signal weakens after removing APCC or NAd4, the manuscript should frame it as a ZG/intermediate-axis program rather than a clean normal-ZG abundance result.",
        "",
        "## Key outputs",
        "",
        "- `results/seurat_reference/seurat_reference_sensitivity_profiles.csv`",
        "- `results/seurat_reference/GSE274314_seurat_reference_sensitivity_paired_delta.csv`",
        "- `results/seurat_reference/GSE274314_seurat_reference_sensitivity_paired_tests.csv`",
        "- `results/figures/GSE274314_seurat_reference_sensitivity.png`",
        "",
    ]
    (PROJECT / "seurat_reference_sensitivity_report_20260601.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    utils = load_cluster_utils()
    X, meta, all_genes = prepare_reference_expression(utils)
    gene_idx = [all_genes.index(g) for g in DECONV_GENES if g in all_genes]
    genes = [all_genes[i] for i in gene_idx]
    Xd = X[:, gene_idx].tocsr()
    Xlog = lognorm_selected_genes(Xd, meta["total_counts"].to_numpy(dtype=float))

    sample = meta["sample_id"].to_numpy()
    dataset = meta["dataset"].to_numpy()
    scenario_masks = {
        "all_cells": np.ones(len(meta), dtype=bool),
        "normal_only": dataset == "normal_snRNA",
        "all_minus_NAd4": sample != "NAd4",
        "normal_minus_NAd4": (dataset == "normal_snRNA") & (sample != "NAd4"),
        "normal_leaveout_NAd1": (dataset == "normal_snRNA") & (sample != "NAd1"),
        "normal_leaveout_NAd2": (dataset == "normal_snRNA") & (sample != "NAd2"),
        "normal_leaveout_NAd3": (dataset == "normal_snRNA") & (sample != "NAd3"),
        "normal_leaveout_NAd4": (dataset == "normal_snRNA") & (sample != "NAd4"),
        "APCC_only": dataset == "APCC_adrenal_scRNA",
    }

    profiles = []
    for scenario, mask in scenario_masks.items():
        prof = build_profiles(Xlog, meta, genes, scenario, mask)
        if not prof.empty:
            profiles.append(prof)
    profile_df = pd.concat(profiles, ignore_index=True)
    profile_df.to_csv(OUT_DIR / "seurat_reference_sensitivity_profiles.csv", index=False)
    profile_counts = profile_df[["scenario", "label", "n_reference_cells"]].copy()

    spatial_samples = load_spatial_expression(genes, utils)
    paired_frames = []
    test_frames = []
    for scenario, sub_profile in profile_df.groupby("scenario", sort=False):
        paired, tests = deconvolve_scenario(sub_profile, genes, spatial_samples)
        paired_frames.append(paired)
        test_frames.append(tests)

    paired_all = pd.concat(paired_frames, ignore_index=True)
    tests_all = pd.concat(test_frames, ignore_index=True)
    paired_all.to_csv(OUT_DIR / "GSE274314_seurat_reference_sensitivity_paired_delta.csv", index=False)
    tests_all.to_csv(OUT_DIR / "GSE274314_seurat_reference_sensitivity_paired_tests.csv", index=False)
    plot_sensitivity(tests_all)
    write_report(profile_counts, tests_all)
    print("DONE")
    print(tests_all.loc[tests_all["label"].isin(["ZG_like", "intermediate_steroidogenic", "ZF_ZR_like"])].sort_values(["scenario", "label"]).to_string(index=False))


if __name__ == "__main__":
    main()
