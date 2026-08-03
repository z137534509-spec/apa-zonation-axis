from __future__ import annotations

import gzip
import io
import json
import math
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
from scipy.sparse import coo_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
OUT = RESULTS / "stage2_zonation_axis"
FIG = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260601)
N_RANDOM_SETS = 1000


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
    "medulla_chromaffin_control": ["TH", "DBH", "PNMT", "CHGA", "CHGB", "SLC18A1"],
    "immune_control": ["PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1"],
    "stromal_control": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "endothelial_control": ["PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"],
    "housekeeping_control": ["ACTB", "GAPDH", "RPLP0", "RPL13A", "RPS18", "B2M", "HPRT1", "PGK1", "TBP", "PPIA"],
}

KEY_SCORE_ORDER = [
    "ZG_aldosterone_program",
    "ZF_cortisol_program",
    "ZR_androgen_program",
    "intermediate_steroidogenic_program",
    "generic_steroidogenesis_program",
    "ZG_vs_ZF_axis_score",
    "ZG_intermediate_vs_ZF_axis_score",
    "aldosterone_specificity_score",
    "medulla_chromaffin_control",
    "immune_control",
    "stromal_control",
    "housekeeping_control",
]


def read_tar_gz_bytes(tar: tarfile.TarFile, member: str) -> bytes:
    raw = tar.extractfile(member)
    if raw is None:
        raise FileNotFoundError(member)
    return gzip.decompress(raw.read())


def read_tar_gz_text(tar: tarfile.TarFile, member: str) -> str:
    return read_tar_gz_bytes(tar, member).decode("utf-8", errors="replace")


def read_features(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="\t", header=None)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["feature_id", "gene_symbol"]
    else:
        df.columns = ["gene_symbol"]
        df["feature_id"] = df["gene_symbol"]
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    return df


def read_positions(text: str) -> pd.DataFrame:
    first = text.splitlines()[0].split(",")[0].strip().lower()
    if first == "barcode":
        pos = pd.read_csv(io.StringIO(text))
    else:
        pos = pd.read_csv(
            io.StringIO(text),
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    pos["barcode"] = pos["barcode"].astype(str)
    return pos


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


def clean_gene_pool(genes: list[str]) -> list[str]:
    out = []
    for gene in genes:
        gene = str(gene)
        if not gene or gene.startswith("MT-") or gene.startswith("RP11-"):
            continue
        if gene.startswith("AC0") or gene.startswith("AL0"):
            continue
        out.append(gene)
    return sorted(set(out))


def load_gene_pool() -> list[str]:
    hvg_path = RESULTS / "cluster_annotation" / "selected_hvg_plus_marker_genes.csv"
    if hvg_path.exists():
        hvg = pd.read_csv(hvg_path)["gene"].astype(str).tolist()
        return clean_gene_pool(hvg)
    tar_path = RAW / "GSE274314" / "GSE274314_RAW.tar"
    with tarfile.open(tar_path, "r") as tar:
        features_member = next(name for name in tar.getnames() if name.endswith("_features.tsv.gz"))
        genes = read_features(read_tar_gz_text(tar, features_member))["gene_symbol"].astype(str).tolist()
    return clean_gene_pool(genes)


def build_random_sets(axis_genes: list[str], pool: list[str], n_sets: int, size: int) -> tuple[list[list[str]], list[str]]:
    excluded = set(axis_genes)
    available = [g for g in pool if g not in excluded]
    if len(available) < size:
        raise ValueError("Not enough genes for random matched sets")
    random_sets = [RNG.choice(available, size=size, replace=False).tolist() for _ in range(n_sets)]
    union = sorted(set(g for genes in random_sets for g in genes))
    return random_sets, union


def collapse_by_gene(mat, gene_symbols: np.ndarray) -> tuple:
    valid = np.array([bool(g) and g != "nan" for g in gene_symbols])
    mat = mat[valid, :]
    gene_symbols = gene_symbols[valid]
    unique_genes, inverse = np.unique(gene_symbols, return_inverse=True)
    coo = mat.tocoo()
    collapsed = coo_matrix((coo.data, (inverse[coo.row], coo.col)), shape=(len(unique_genes), mat.shape[1])).tocsr()
    return collapsed, unique_genes


def load_spotlight_props() -> pd.DataFrame:
    props = pd.read_csv(RESULTS / "spotlight" / "GSE274314_SPOTlight_props_full_scran.csv")
    props["geo_accession"] = props["spot_id"].str.split(":").str[0]
    props["barcode"] = props["spot_id"].str.split(":", n=1).str[1]
    meta = pd.read_csv(PROCESSED / "GSE274314_sample_metadata.csv")
    meta["genotype"] = meta["genotype"].fillna("NA").replace("", "NA")
    return props.merge(meta, on="geo_accession", how="left")


def load_spatial_expression(spot_genes: list[str], random_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot_genes = sorted(set(spot_genes))
    section_genes = sorted(set(spot_genes + random_genes))
    spot_rows = []
    section_rows = []
    tar_path = RAW / "GSE274314" / "GSE274314_RAW.tar"
    with tarfile.open(tar_path, "r") as tar:
        names = tar.getnames()
        prefixes = sorted({name.replace("_matrix.mtx.gz", "") for name in names if name.endswith("_matrix.mtx.gz")})
        for prefix in prefixes:
            gsm = prefix.split("_", 1)[0]
            mat = mmread(io.BytesIO(read_tar_gz_bytes(tar, f"{prefix}_matrix.mtx.gz"))).tocsc()
            features = read_features(read_tar_gz_text(tar, f"{prefix}_features.tsv.gz"))
            barcodes = np.array(read_tar_gz_text(tar, f"{prefix}_barcodes.tsv.gz").splitlines(), dtype=str)
            pos = read_positions(read_tar_gz_text(tar, f"{prefix}_tissue_positions_list.csv.gz"))
            scale = json.loads(read_tar_gz_text(tar, f"{prefix}_scalefactors_json.json.gz"))
            tissue_barcodes = pos.loc[pos["in_tissue"].astype(int) == 1, "barcode"].astype(str).tolist()
            barcode_index = pd.Series(np.arange(len(barcodes)), index=barcodes)
            tissue_idx = barcode_index.loc[tissue_barcodes].to_numpy()
            mat = mat[:, tissue_idx]
            tissue_barcodes = barcodes[tissue_idx]
            total_umi = np.asarray(mat.sum(axis=0)).ravel().astype(float)
            total_umi[total_umi <= 0] = np.nan
            n_detected = np.asarray((mat > 0).sum(axis=0)).ravel().astype(float)
            collapsed, unique_genes = collapse_by_gene(mat, features["gene_symbol"].to_numpy(dtype=str))
            gene_index = pd.Series(np.arange(len(unique_genes)), index=unique_genes)
            present_spot = [g for g in spot_genes if g in gene_index.index]
            present_section = [g for g in section_genes if g in gene_index.index]

            spot_expr = {}
            for gene in present_spot:
                counts = np.asarray(collapsed[gene_index[gene], :].sum(axis=0)).ravel().astype(float)
                spot_expr[gene] = np.log1p(counts / total_umi * 1e4)
            spot_df = pd.DataFrame(spot_expr)
            spot_df["barcode"] = tissue_barcodes
            spot_df["geo_accession"] = gsm
            spot_df["total_umi"] = total_umi
            spot_df["n_detected_genes"] = n_detected
            pos_sub = pos.loc[pos["barcode"].isin(tissue_barcodes)].copy()
            pos_sub["barcode"] = pos_sub["barcode"].astype(str)
            spot_df = spot_df.merge(pos_sub, on="barcode", how="left")
            spot_df["lowres_x"] = spot_df["pxl_col_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            spot_df["lowres_y"] = spot_df["pxl_row_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            spot_rows.append(spot_df)

            section_record = {"geo_accession": gsm}
            for gene in present_section:
                counts = np.asarray(collapsed[gene_index[gene], :].sum(axis=0)).ravel().astype(float)
                expr = np.log1p(counts / total_umi * 1e4)
                section_record[gene] = float(np.nanmean(expr))
            section_rows.append(section_record)
    return pd.concat(spot_rows, ignore_index=True), pd.DataFrame(section_rows)


def module_scores_from_expr(expr: pd.DataFrame, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = sorted(set(g for genes in signatures.values() for g in genes if g in expr.columns))
    scaled = pd.DataFrame(index=expr.index)
    for gene in gene_cols:
        vals = expr[gene].astype(float)
        sd = vals.std(ddof=0)
        scaled[gene] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else 0.0
    scores = pd.DataFrame(index=expr.index)
    availability = []
    for name, genes in signatures.items():
        available = [g for g in genes if g in scaled.columns]
        missing = [g for g in genes if g not in scaled.columns]
        scores[name] = scaled[available].mean(axis=1) if available else np.nan
        availability.append(
            {
                "signature": name,
                "n_requested": len(genes),
                "n_available": len(available),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
            }
        )
    scores["ZG_vs_ZF_axis_score"] = scores["ZG_aldosterone_program"] - scores["ZF_cortisol_program"]
    scores["ZG_intermediate_vs_ZF_axis_score"] = (
        scores[["ZG_aldosterone_program", "intermediate_steroidogenic_program"]].mean(axis=1) - scores["ZF_cortisol_program"]
    )
    scores["aldosterone_specificity_score"] = scores["ZG_aldosterone_program"] - scores["generic_steroidogenesis_program"]
    return scores, pd.DataFrame(availability)


def add_depth_adjusted_scores(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["log10_total_umi"] = np.log10(df["total_umi"].astype(float).fillna(0) + 1)
    covars = pd.DataFrame(
        {
            "log10_total_umi": df["log10_total_umi"],
            "n_detected_genes": df["n_detected_genes"].astype(float).fillna(0),
        }
    )
    x = StandardScaler().fit_transform(covars)
    for score in score_cols:
        y = df[score].astype(float)
        ok = y.notna().to_numpy() & np.isfinite(x).all(axis=1)
        adjusted = np.full(len(df), np.nan)
        if ok.sum() > 10 and y[ok].std(ddof=0) > 0:
            model = LinearRegression().fit(x[ok], y[ok])
            adjusted[ok] = y[ok] - model.predict(x[ok])
        df[f"{score}_depth_adjusted"] = adjusted
    return df


def section_summary(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    agg = {
        "spot_id": "count",
        "total_umi": "median",
        "n_detected_genes": "median",
    }
    for score in score_cols:
        agg[score] = ["mean", "median"]
        adj = f"{score}_depth_adjusted"
        if adj in df.columns:
            agg[adj] = ["mean", "median"]
    grouped = df.groupby(["geo_accession", "patient", "tissue", "genotype"], dropna=False).agg(agg)
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.rename(columns={"spot_id_count": "n_spots", "total_umi_median": "median_total_umi", "n_detected_genes_median": "median_n_detected_genes"})
    return grouped.reset_index()


def paired_tests(summary: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    deltas = []
    for metric in metric_cols:
        wide = summary.pivot_table(index="patient", columns="tissue", values=metric, aggfunc="mean")
        if not {"APA", "Adjacent"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["APA", "Adjacent"])
        delta = (wide["APA"] - wide["Adjacent"]).astype(float)
        if len(delta) >= 2 and delta.std(ddof=1) > 0:
            paired_t_p = stats.ttest_1samp(delta, 0.0).pvalue
            wilcoxon_p = stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan
            dz = delta.mean() / delta.std(ddof=1)
        else:
            paired_t_p = np.nan
            wilcoxon_p = np.nan
            dz = np.nan
        npos = int((delta > 0).sum())
        sign_p = stats.binomtest(npos, len(delta), p=0.5, alternative="greater").pvalue if len(delta) else np.nan
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(delta),
                "mean_APA_minus_adjacent": delta.mean(),
                "median_APA_minus_adjacent": delta.median(),
                "n_positive": npos,
                "n_negative": int((delta < 0).sum()),
                "paired_t_p": paired_t_p,
                "wilcoxon_p": wilcoxon_p,
                "one_sided_sign_p_APA_gt_adjacent": sign_p,
                "paired_effect_size_dz": dz,
            }
        )
        for patient, value in delta.items():
            deltas.append(
                {
                    "patient": patient,
                    "metric": metric,
                    "APA": wide.loc[patient, "APA"],
                    "Adjacent": wide.loc[patient, "Adjacent"],
                    "APA_minus_adjacent": value,
                }
            )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["paired_t_fdr"] = bh_fdr(tests["paired_t_p"])
    return tests, pd.DataFrame(deltas)


def leave_one_patient_out(summary: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    patients = sorted(summary["patient"].dropna().unique())
    for metric in metric_cols:
        for left_out in patients:
            sub = summary[summary["patient"] != left_out]
            tests, _ = paired_tests(sub, [metric])
            if tests.empty:
                continue
            row = tests.iloc[0].to_dict()
            rows.append(
                {
                    "metric": metric,
                    "left_out_patient": left_out,
                    "n_pairs": row.get("n_pairs"),
                    "mean_APA_minus_adjacent": row.get("mean_APA_minus_adjacent"),
                    "n_positive": row.get("n_positive"),
                    "paired_t_p": row.get("paired_t_p"),
                }
            )
    return pd.DataFrame(rows)


def random_gene_set_null(section_gene_means: pd.DataFrame, random_sets: list[list[str]], observed_metric: pd.Series) -> pd.DataFrame:
    gene_cols = [c for c in section_gene_means.columns if c != "geo_accession"]
    z = section_gene_means.copy()
    for gene in gene_cols:
        sd = z[gene].std(ddof=0)
        z[gene] = (z[gene] - z[gene].mean()) / sd if sd and np.isfinite(sd) else 0.0
    meta = pd.read_csv(PROCESSED / "GSE274314_sample_metadata.csv")
    z = z.merge(meta[["geo_accession", "patient", "tissue"]], on="geo_accession", how="left")

    rows = []
    for i, genes in enumerate(random_sets, start=1):
        available = [g for g in genes if g in z.columns]
        if len(available) < max(3, len(genes) // 2):
            continue
        z["random_score"] = z[available].mean(axis=1)
        wide = z.pivot_table(index="patient", columns="tissue", values="random_score", aggfunc="mean").dropna()
        if {"APA", "Adjacent"}.issubset(wide.columns):
            delta = wide["APA"] - wide["Adjacent"]
            rows.append({"random_set": i, "n_genes": len(available), "mean_APA_minus_adjacent": delta.mean(), "n_positive": int((delta > 0).sum())})
    out = pd.DataFrame(rows)
    obs = float(observed_metric.mean())
    out["observed_axis_mean_delta"] = obs
    if not out.empty:
        out["empirical_abs_ge_observed"] = (out["mean_APA_minus_adjacent"].abs() >= abs(obs)).astype(int)
        empirical_p = (1 + out["empirical_abs_ge_observed"].sum()) / (len(out) + 1)
        out.attrs["empirical_p"] = empirical_p
    return out


def roi_concordance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if "cyp11b2_top_quartile_roi" not in df.columns:
        df["cyp11b2_top_quartile_roi"] = df.groupby("geo_accession")["CYP11B2"].rank(method="first", pct=True) > 0.75
    df["hsd3b2_top_quartile_roi"] = df.groupby("geo_accession")["HSD3B2"].rank(method="first", pct=True) > 0.75
    df["axis_top_quartile_roi"] = df.groupby("geo_accession")["ZG_intermediate_vs_ZF_axis_score"].rank(method="first", pct=True) > 0.75
    if "image_domain_top_cyp11b2" not in df.columns:
        df["image_domain_top_cyp11b2"] = False
    roi_cols = ["cyp11b2_top_quartile_roi", "hsd3b2_top_quartile_roi", "axis_top_quartile_roi", "image_domain_top_cyp11b2"]
    section_rows = []
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        for a in roi_cols:
            for b in roi_cols:
                av = sub[a].astype(bool).to_numpy()
                bv = sub[b].astype(bool).to_numpy()
                inter = np.logical_and(av, bv).sum()
                union = np.logical_or(av, bv).sum()
                section_rows.append(
                    {
                        "geo_accession": gsm,
                        "patient": patient,
                        "tissue": tissue,
                        "roi_a": a,
                        "roi_b": b,
                        "jaccard": inter / union if union else np.nan,
                        "overlap_fraction_of_a": inter / av.sum() if av.sum() else np.nan,
                    }
                )
    section = pd.DataFrame(section_rows)
    overall = section.groupby(["tissue", "roi_a", "roi_b"], dropna=False).agg(mean_jaccard=("jaccard", "mean"), median_jaccard=("jaccard", "median")).reset_index()
    return section, overall


def per_section_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        x_cov = sub[["generic_steroidogenesis_program", "log10_total_umi", "n_detected_genes"]].replace([np.inf, -np.inf], np.nan)
        y = sub["CYP11B2"].astype(float)
        z = sub["ZG_like"].astype(float)
        ok = x_cov.notna().all(axis=1) & y.notna() & z.notna()
        if ok.sum() < 20:
            continue
        x_scaled = StandardScaler().fit_transform(x_cov.loc[ok])
        y_res = y.loc[ok] - LinearRegression().fit(x_scaled, y.loc[ok]).predict(x_scaled)
        z_res = z.loc[ok] - LinearRegression().fit(x_scaled, z.loc[ok]).predict(x_scaled)
        pearson = stats.pearsonr(z_res, y_res)
        spearman_raw = stats.spearmanr(sub.loc[ok, "ZG_like"], sub.loc[ok, "CYP11B2"])
        rows.append(
            {
                "geo_accession": gsm,
                "patient": patient,
                "tissue": tissue,
                "n_spots": int(ok.sum()),
                "raw_spearman_ZG_like_CYP11B2": spearman_raw.statistic,
                "raw_spearman_p": spearman_raw.pvalue,
                "partial_pearson_ZG_like_CYP11B2_adj_generic_depth": pearson.statistic,
                "partial_pearson_p": pearson.pvalue,
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbor_enrichment(df: pd.DataFrame, roi_col: str, n_perm: int = 500) -> pd.DataFrame:
    rows = []
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        coords = sub[["lowres_x", "lowres_y"]].to_numpy(float)
        high = sub[roi_col].astype(bool).to_numpy()
        if len(sub) < 30 or high.sum() < 5:
            continue
        k = min(7, len(sub))
        idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
        obs = high[idx[high]].mean()
        exp = []
        for _ in range(n_perm):
            perm = RNG.permutation(high)
            exp.append(perm[idx[perm]].mean())
        exp = np.asarray(exp)
        rows.append(
            {
                "geo_accession": gsm,
                "patient": patient,
                "tissue": tissue,
                "roi": roi_col,
                "n_spots": len(sub),
                "n_high": int(high.sum()),
                "observed_high_neighbor_fraction": obs,
                "expected_high_neighbor_fraction_mean": float(exp.mean()),
                "enrichment_ratio": obs / exp.mean() if exp.mean() > 0 else np.nan,
                "permutation_p": (1 + np.sum(exp >= obs)) / (len(exp) + 1),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["permutation_fdr"] = bh_fdr(out["permutation_p"])
    return out


def plot_paired(deltas: pd.DataFrame, metrics: list[str], outfile: Path, title: str) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(4.0 * n, 8), 4.0), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = deltas[deltas["metric"] == metric]
        for _, row in sub.iterrows():
            ax.plot([0, 1], [row["Adjacent"], row["APA"]], color="#888888", alpha=0.75, lw=1)
            ax.scatter([0, 1], [row["Adjacent"], row["APA"]], color="#1f77b4", s=18, zorder=3)
        ax.set_title(metric.replace("_", "\n"), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Adjacent", "APA"])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("section mean score")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def plot_random_null(random_null: pd.DataFrame, outfile: Path) -> None:
    if random_null.empty:
        return
    obs = random_null["observed_axis_mean_delta"].iloc[0]
    p = random_null.attrs.get("empirical_p", np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(random_null["mean_APA_minus_adjacent"], bins=40, color="#9ecae1", edgecolor="#ffffff")
    ax.axvline(obs, color="#d62728", lw=2, label=f"observed={obs:.3f}\nempirical p={p:.4f}")
    ax.axvline(-obs, color="#d62728", lw=1, linestyle="--")
    ax.set_xlabel("random gene-set mean APA-adjacent delta")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def plot_roi_concordance(overall: pd.DataFrame, outfile: Path) -> None:
    sub = overall[overall["tissue"] == "APA"].copy()
    if sub.empty:
        return
    mat = sub.pivot(index="roi_a", columns="roi_b", values="mean_jaccard")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("APA ROI concordance, mean Jaccard")
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def safe_p(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4g}"


def report_lines(
    availability: pd.DataFrame,
    paired: pd.DataFrame,
    loo: pd.DataFrame,
    random_null: pd.DataFrame,
    partial: pd.DataFrame,
    roi_overall: pd.DataFrame,
    axis_nn: pd.DataFrame,
) -> list[str]:
    def paired_row(metric: str) -> str:
        sub = paired[paired["metric"] == f"{metric}_mean"]
        if sub.empty:
            sub = paired[paired["metric"] == metric]
        if sub.empty:
            return f"|{metric}|NA|NA|NA|NA|NA|NA|"
        r = sub.iloc[0]
        return (
            f"|{metric}|{int(r['n_pairs'])}|{r['mean_APA_minus_adjacent']:.4g}|{int(r['n_positive'])}|"
            f"{int(r['n_negative'])}|{safe_p(r['paired_t_p'])}|{safe_p(r['wilcoxon_p'])}|"
        )

    key_metrics = [
        "ZG_aldosterone_program",
        "ZF_cortisol_program",
        "ZR_androgen_program",
        "intermediate_steroidogenic_program",
        "generic_steroidogenesis_program",
        "ZG_vs_ZF_axis_score",
        "ZG_intermediate_vs_ZF_axis_score",
        "aldosterone_specificity_score",
        "medulla_chromaffin_control",
        "immune_control",
        "stromal_control",
        "housekeeping_control",
    ]
    emp_p = random_null.attrs.get("empirical_p", np.nan) if not random_null.empty else np.nan
    partial_summary = partial.groupby("tissue").agg(
        n_sections=("geo_accession", "count"),
        mean_raw_rho=("raw_spearman_ZG_like_CYP11B2", "mean"),
        mean_partial_r=("partial_pearson_ZG_like_CYP11B2_adj_generic_depth", "mean"),
        n_partial_positive=("partial_pearson_ZG_like_CYP11B2_adj_generic_depth", lambda x: int((x > 0).sum())),
    ).reset_index() if not partial.empty else pd.DataFrame()
    lines = [
        "# Stage 2 zonation-axis strengthening analysis",
        "",
        "Date: 2026-06-01",
        "",
        "Status: PASS for formal zonation-axis scores, negative controls, depth adjustment, random gene-set nulls, ROI concordance, and state-vs-abundance sensitivity.",
        "",
        "## Why this was run",
        "",
        "This stage tests whether the central claim should be framed as `aldosterone-dominant ZG/intermediate-axis spatial steroidogenic rewiring` rather than normal ZG abundance increase or adrenal zonation collapse.",
        "",
        "Primary statistics are section/patient-level paired comparisons. Spot-level analyses are used only for spatial localization, ROI concordance, and within-section sensitivity.",
        "",
        "## Gene signature availability",
        "",
        "|signature|available/requested|available genes|missing genes|",
        "|---|---:|---|---|",
    ]
    for _, r in availability.iterrows():
        lines.append(f"|{r['signature']}|{int(r['n_available'])}/{int(r['n_requested'])}|{r['available_genes']}|{r['missing_genes']}|")
    lines.extend(
        [
            "",
            "## Patient-paired APA-vs-adjacent score tests",
            "",
            "|metric|n_pairs|mean_APA_minus_adjacent|n_positive|n_negative|paired_t_p|wilcoxon_p|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in key_metrics:
        lines.append(paired_row(f"{metric}_mean"))
    depth_metrics = [f"{m}_depth_adjusted_mean" for m in ["ZG_intermediate_vs_ZF_axis_score", "ZG_vs_ZF_axis_score", "aldosterone_specificity_score"]]
    lines.extend(["", "## Count-depth adjusted primary scores", "", "|metric|n_pairs|mean_APA_minus_adjacent|n_positive|n_negative|paired_t_p|wilcoxon_p|", "|---|---:|---:|---:|---:|---:|---:|"])
    for metric in depth_metrics:
        lines.append(paired_row(metric))
    lines.extend(
        [
            "",
            "## Random matched gene-set null",
            "",
            f"- Random sets: {len(random_null)}.",
            f"- Observed `ZG_intermediate_vs_ZF_axis_score` mean paired delta empirical two-sided p: {safe_p(emp_p)}.",
            "",
            "## Leave-one-patient-out sensitivity",
            "",
            "|metric|min_mean_delta|max_mean_delta|min_n_positive|max_n_positive|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in ["ZG_intermediate_vs_ZF_axis_score_mean", "ZG_vs_ZF_axis_score_mean", "aldosterone_specificity_score_mean"]:
        sub = loo[loo["metric"] == metric]
        if not sub.empty:
            lines.append(
                f"|{metric}|{sub['mean_APA_minus_adjacent'].min():.4g}|{sub['mean_APA_minus_adjacent'].max():.4g}|"
                f"{int(sub['n_positive'].min())}|{int(sub['n_positive'].max())}|"
            )
    lines.extend(
        [
            "",
            "## State-vs-abundance sensitivity",
            "",
            "Within each section, `ZG_like` was tested against CYP11B2 before and after adjusting for generic steroidogenesis, total UMI, and detected genes.",
            "",
            "|tissue|n_sections|mean_raw_spearman|mean_partial_r|n_partial_positive|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, r in partial_summary.iterrows():
        lines.append(f"|{r['tissue']}|{int(r['n_sections'])}|{r['mean_raw_rho']:.4g}|{r['mean_partial_r']:.4g}|{int(r['n_partial_positive'])}|")
    roi_apa = roi_overall[(roi_overall["tissue"] == "APA") & (roi_overall["roi_a"] == "cyp11b2_top_quartile_roi")]
    lines.extend(["", "## ROI concordance and spatial contiguity", ""])
    for _, r in roi_apa.iterrows():
        lines.append(f"- APA mean Jaccard, CYP11B2 transcript ROI vs `{r['roi_b']}`: {r['mean_jaccard']:.3g}.")
    if not axis_nn.empty:
        nn_sum = axis_nn.groupby("tissue").agg(mean_enrichment=("enrichment_ratio", "mean"), median_enrichment=("enrichment_ratio", "median"), n_sections=("geo_accession", "count")).reset_index()
        for _, r in nn_sum.iterrows():
            lines.append(f"- {r['tissue']} axis-high neighbor enrichment ratio: mean = {r['mean_enrichment']:.3g}, median = {r['median_enrichment']:.3g}, n = {int(r['n_sections'])}.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the formal ZG/intermediate-vs-ZF axis remains positive after patient-paired summarization, depth adjustment, negative controls, and random gene-set null comparison, the manuscript can more safely claim a spatial steroidogenic axis rewiring model. The result should still not be described as normal ZG cell abundance or complete zonation collapse.",
            "",
            "## Key outputs",
            "",
            "- `results/stage2_zonation_axis/GSE274314_stage2_spot_scores.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_section_score_summary.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_paired_tests.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_leave_one_patient_out.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_random_gene_set_null.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_roi_concordance_section.csv`",
            "- `results/stage2_zonation_axis/GSE274314_stage2_partial_correlations.csv`",
            "- `results/figures/GSE274314_stage2_zonation_axis_paired_scores.png`",
            "- `results/figures/GSE274314_stage2_negative_control_paired_scores.png`",
            "- `results/figures/GSE274314_stage2_random_gene_set_null.png`",
            "- `results/figures/GSE274314_stage2_roi_concordance_heatmap.png`",
            "",
            "## Claim boundary",
            "",
            "Can say: formal section-level scores and negative controls support an aldosterone-dominant ZG/intermediate-vs-ZF spatial steroidogenic axis in APA.",
            "",
            "Cannot say: APA causes adrenal zonation collapse, APA contains more histologic ZG cells, or the deconvolution coefficient is a direct abundance measurement.",
            "",
        ]
    )
    return lines


def main() -> None:
    axis_gene_union = sorted(set(g for key in ["ZG_aldosterone_program", "ZF_cortisol_program", "intermediate_steroidogenic_program"] for g in SIGNATURES[key]))
    all_signature_genes = sorted(set(g for genes in SIGNATURES.values() for g in genes))
    random_pool = load_gene_pool()
    random_sets, random_genes = build_random_sets(axis_gene_union, random_pool, N_RANDOM_SETS, size=len(axis_gene_union))
    spot_expr, section_gene_means = load_spatial_expression(all_signature_genes + ["CYP11B2", "CYP11B1", "CYP17A1", "HSD3B2"], random_genes)
    props = load_spotlight_props()
    props["genotype"] = props["genotype"].fillna("NA").replace("", "NA")
    spot_expr["spot_id"] = spot_expr["geo_accession"] + ":" + spot_expr["barcode"]
    df = props.merge(spot_expr, on=["geo_accession", "barcode", "spot_id"], how="left", suffixes=("", "_expr"))
    for col in ["patient", "tissue", "genotype", "sample_type", "title"]:
        expr_col = f"{col}_expr"
        if expr_col in df.columns:
            df[col] = df[col].fillna(df[expr_col])
            df = df.drop(columns=[expr_col])

    scores, availability = module_scores_from_expr(df, SIGNATURES)
    for col in scores.columns:
        df[col] = scores[col].to_numpy()

    score_cols = list(scores.columns)
    df = add_depth_adjusted_scores(df, score_cols)
    roi_path = RESULTS / "roi_anchor" / "GSE274314_roi_anchor_spot_level.csv"
    if roi_path.exists():
        roi = pd.read_csv(roi_path, usecols=lambda c: c in {"spot_id", "cyp11b2_top_quartile_roi", "image_domain_top_cyp11b2", "image_domain"})
        df = df.merge(roi, on="spot_id", how="left")
    df["cyp11b2_top_quartile_roi"] = df.get("cyp11b2_top_quartile_roi", False).fillna(False).astype(bool)
    df["image_domain_top_cyp11b2"] = df.get("image_domain_top_cyp11b2", False).fillna(False).astype(bool)
    df["axis_top_quartile_roi"] = df.groupby("geo_accession")["ZG_intermediate_vs_ZF_axis_score"].rank(method="first", pct=True) > 0.75
    df["hsd3b2_top_quartile_roi"] = df.groupby("geo_accession")["HSD3B2"].rank(method="first", pct=True) > 0.75

    summary = section_summary(df, score_cols)
    metric_cols = [f"{score}_mean" for score in score_cols] + [f"{score}_depth_adjusted_mean" for score in score_cols]
    paired, paired_delta = paired_tests(summary, metric_cols)
    loo = leave_one_patient_out(summary, [f"{score}_mean" for score in ["ZG_intermediate_vs_ZF_axis_score", "ZG_vs_ZF_axis_score", "aldosterone_specificity_score"]])
    observed_delta = paired_delta[paired_delta["metric"] == "ZG_intermediate_vs_ZF_axis_score_mean"]["APA_minus_adjacent"]
    random_null = random_gene_set_null(section_gene_means, random_sets, observed_delta)
    partial = per_section_partial_correlations(df)
    roi_section, roi_overall = roi_concordance(df)
    axis_nn = nearest_neighbor_enrichment(df, "axis_top_quartile_roi")

    availability.to_csv(OUT / "GSE274314_stage2_signature_gene_availability.csv", index=False)
    df.to_csv(OUT / "GSE274314_stage2_spot_scores.csv", index=False)
    summary.to_csv(OUT / "GSE274314_stage2_section_score_summary.csv", index=False)
    paired.to_csv(OUT / "GSE274314_stage2_paired_tests.csv", index=False)
    paired_delta.to_csv(OUT / "GSE274314_stage2_paired_delta.csv", index=False)
    loo.to_csv(OUT / "GSE274314_stage2_leave_one_patient_out.csv", index=False)
    random_null.to_csv(OUT / "GSE274314_stage2_random_gene_set_null.csv", index=False)
    roi_section.to_csv(OUT / "GSE274314_stage2_roi_concordance_section.csv", index=False)
    roi_overall.to_csv(OUT / "GSE274314_stage2_roi_concordance_overall.csv", index=False)
    partial.to_csv(OUT / "GSE274314_stage2_partial_correlations.csv", index=False)
    axis_nn.to_csv(OUT / "GSE274314_stage2_axis_high_neighbor_enrichment.csv", index=False)

    plot_paired(
        paired_delta,
        ["ZG_intermediate_vs_ZF_axis_score_mean", "ZG_vs_ZF_axis_score_mean", "aldosterone_specificity_score_mean", "generic_steroidogenesis_program_mean"],
        FIG / "GSE274314_stage2_zonation_axis_paired_scores.png",
        "Formal zonation-axis scores, patient-paired",
    )
    plot_paired(
        paired_delta,
        ["medulla_chromaffin_control_mean", "immune_control_mean", "stromal_control_mean", "housekeeping_control_mean"],
        FIG / "GSE274314_stage2_negative_control_paired_scores.png",
        "Negative-control signatures, patient-paired",
    )
    plot_random_null(random_null, FIG / "GSE274314_stage2_random_gene_set_null.png")
    plot_roi_concordance(roi_overall, FIG / "GSE274314_stage2_roi_concordance_heatmap.png")

    (PROJECT / "stage2_zonation_axis_strengthening_report_20260601.md").write_text(
        "\n".join(report_lines(availability, paired, loo, random_null, partial, roi_overall, axis_nn)),
        encoding="utf-8-sig",
    )

    show = paired[paired["metric"].isin([f"{m}_mean" for m in KEY_SCORE_ORDER])].copy()
    print(show[["metric", "n_pairs", "mean_APA_minus_adjacent", "n_positive", "n_negative", "paired_t_p", "wilcoxon_p"]].to_string(index=False))
    print("random_empirical_p", random_null.attrs.get("empirical_p", np.nan))
    print("DONE")


if __name__ == "__main__":
    main()
