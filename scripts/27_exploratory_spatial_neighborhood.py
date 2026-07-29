from __future__ import annotations

import gzip
import io
import itertools
import json
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
STAGE2 = RESULTS / "stage2_zonation_axis"
OUT = RESULTS / "spatial_niche_neighborhood_20260602"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260602)
N_PERM = 999
PRIMARY_AXIS = "ZG_intermediate_vs_ZF_axis_score"

NICHE_SIGNATURES = {
    "endothelial_marker_module": ["PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"],
    "stromal_marker_module": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "wnt_rspo_module": ["RSPO3", "RSPO2", "WNT4", "WNT2", "WNT5A", "LGR5", "FZD1", "FZD2", "FZD4", "FZD5", "FZD6", "LRP5", "LRP6"],
    "cxcl12_axis_module": ["CXCL12", "CXCR4", "ACKR3"],
    "vascular_stromal_niche_module": [
        "PECAM1",
        "VWF",
        "KDR",
        "COL1A1",
        "COL1A2",
        "DCN",
        "LUM",
        "RSPO3",
        "WNT4",
        "CXCL12",
    ],
    "aldosterone_receptor_context_module": ["LGR5", "AGTR1", "MC2R", "NR4A1", "NR4A2"],
}

TARGET_GENES = sorted(set(g for genes in NICHE_SIGNATURES.values() for g in genes))

NEIGHBOR_VARIABLES = [
    "neighbor_endothelial",
    "neighbor_fibroblast_stromal",
    "neighbor_endothelial_marker_module",
    "neighbor_stromal_marker_module",
    "neighbor_wnt_rspo_module",
    "neighbor_cxcl12_axis_module",
    "neighbor_vascular_stromal_niche_module",
]

LR_PAIRS = {
    "RSPO3_to_LGR5_neighbor_product": ("RSPO3", "LGR5"),
    "WNT4_to_FZD_neighbor_product": ("WNT4", "FZD_receptor_module"),
    "CXCL12_to_CXCR4_neighbor_product": ("CXCL12", "CXCR4"),
    "CXCL12_to_ACKR3_neighbor_product": ("CXCL12", "ACKR3"),
}


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


def collapse_by_gene(mat, gene_symbols: np.ndarray) -> tuple:
    valid = np.array([bool(g) and g != "nan" for g in gene_symbols])
    mat = mat[valid, :]
    gene_symbols = gene_symbols[valid]
    unique_genes, inverse = np.unique(gene_symbols, return_inverse=True)
    coo = mat.tocoo()
    collapsed = coo_matrix((coo.data, (inverse[coo.row], coo.col)), shape=(len(unique_genes), mat.shape[1])).tocsr()
    return collapsed, unique_genes


def extract_targeted_spatial_expression(target_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    availability_rows = []
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
            totals = np.asarray(mat.sum(axis=0)).ravel().astype(float)
            totals[totals <= 0] = np.nan
            collapsed, unique_genes = collapse_by_gene(mat, features["gene_symbol"].to_numpy(dtype=str))
            gene_index = pd.Series(np.arange(len(unique_genes)), index=unique_genes)
            present = [g for g in target_genes if g in gene_index.index]
            missing = [g for g in target_genes if g not in gene_index.index]
            availability_rows.append(
                {
                    "geo_accession": gsm,
                    "n_requested": len(target_genes),
                    "n_available": len(present),
                    "available_genes": ";".join(present),
                    "missing_genes": ";".join(missing),
                }
            )
            expr = {}
            for gene in present:
                counts = np.asarray(collapsed[gene_index[gene], :].sum(axis=0)).ravel().astype(float)
                expr[gene] = np.log1p(counts / totals * 1e4)
            df = pd.DataFrame(expr)
            df["barcode"] = tissue_barcodes
            df["geo_accession"] = gsm
            pos_sub = pos.loc[pos["barcode"].isin(tissue_barcodes)].copy()
            pos_sub["barcode"] = pos_sub["barcode"].astype(str)
            df = df.merge(pos_sub, on="barcode", how="left")
            df["lowres_x_raw"] = df["pxl_col_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            df["lowres_y_raw"] = df["pxl_row_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["spot_id"] = out["geo_accession"] + ":" + out["barcode"]
    return out, pd.DataFrame(availability_rows)


def zscore_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for col in cols:
        vals = df[col].astype(float)
        sd = vals.std(ddof=0)
        z[col] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return z


def add_niche_modules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    availability = []
    all_genes = sorted({g for genes in NICHE_SIGNATURES.values() for g in genes if g in df.columns})
    z = zscore_columns(df, all_genes)
    for name, genes in NICHE_SIGNATURES.items():
        present = [g for g in genes if g in z.columns]
        missing = [g for g in genes if g not in z.columns]
        df[name] = z[present].mean(axis=1) if present else np.nan
        availability.append(
            {
                "module": name,
                "n_requested": len(genes),
                "n_available": len(present),
                "available_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )
    fzd_genes = [g for g in ["FZD1", "FZD2", "FZD4", "FZD5", "FZD6"] if g in z.columns]
    df["FZD_receptor_module"] = z[fzd_genes].mean(axis=1) if fzd_genes else np.nan
    availability.append(
        {
            "module": "FZD_receptor_module",
            "n_requested": 5,
            "n_available": len(fzd_genes),
            "available_genes": ";".join(fzd_genes),
            "missing_genes": ";".join([g for g in ["FZD1", "FZD2", "FZD4", "FZD5", "FZD6"] if g not in fzd_genes]),
        }
    )
    return df, pd.DataFrame(availability)


def add_neighbor_features(df: pd.DataFrame, variables: list[str], k: int = 6) -> pd.DataFrame:
    df = df.copy()
    for var in variables:
        df[f"neighbor_{var}"] = np.nan
    df["mean_neighbor_distance"] = np.nan
    for gsm, sub_idx in df.groupby("geo_accession").groups.items():
        idx = np.asarray(list(sub_idx))
        sub = df.loc[idx]
        coords = sub[["lowres_x", "lowres_y"]].to_numpy(float)
        if len(sub) <= k + 1:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        dist, neigh = nn.kneighbors(coords, return_distance=True)
        neigh = neigh[:, 1:]
        dist = dist[:, 1:]
        df.loc[idx, "mean_neighbor_distance"] = dist.mean(axis=1)
        for var in variables:
            vals = sub[var].to_numpy(float)
            df.loc[idx, f"neighbor_{var}"] = np.nanmean(vals[neigh], axis=1)
    return df


def add_lr_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, (ligand, receptor) in LR_PAIRS.items():
        if ligand not in df.columns or receptor not in df.columns:
            df[name] = np.nan
            continue
        if f"neighbor_{ligand}" not in df.columns:
            df[name] = np.nan
            continue
        df[name] = df[f"neighbor_{ligand}"].astype(float) * df[receptor].astype(float)
    return df


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


def section_spatial_tests(df: pd.DataFrame, variables: list[str], axis_col: str = PRIMARY_AXIS) -> pd.DataFrame:
    rows = []
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        high = sub["axis_top_quartile_roi"].astype(bool).to_numpy()
        axis = sub[axis_col].to_numpy(float)
        for var in variables:
            vals = sub[var].to_numpy(float)
            ok = np.isfinite(axis) & np.isfinite(vals)
            if ok.sum() < 30 or np.nanstd(vals[ok]) == 0 or np.nanstd(axis[ok]) == 0:
                continue
            rho, rho_p = stats.spearmanr(axis[ok], vals[ok])
            high_ok = high & ok
            low_ok = (~high) & ok
            if high_ok.sum() < 5 or low_ok.sum() < 5:
                diff = np.nan
                perm_p = np.nan
            else:
                diff = float(vals[high_ok].mean() - vals[low_ok].mean())
                null = []
                ok_idx = np.where(ok)[0]
                n_high = int(high_ok.sum())
                for _ in range(N_PERM):
                    perm_high_idx = RNG.choice(ok_idx, size=n_high, replace=False)
                    perm_high = np.zeros(len(sub), dtype=bool)
                    perm_high[perm_high_idx] = True
                    null.append(float(vals[perm_high & ok].mean() - vals[(~perm_high) & ok].mean()))
                null = np.asarray(null)
                perm_p = (1 + int(np.sum(np.abs(null) >= abs(diff)))) / (N_PERM + 1)
            rows.append(
                {
                    "geo_accession": gsm,
                    "patient": int(patient),
                    "tissue": tissue,
                    "variable": var,
                    "n_spots": int(ok.sum()),
                    "axis_spearman_rho": float(rho),
                    "axis_spearman_p": float(rho_p),
                    "axis_high_minus_other_mean": diff,
                    "axis_high_permutation_p": perm_p,
                    "n_axis_high": int(high_ok.sum()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["axis_high_permutation_fdr"] = bh_fdr(out["axis_high_permutation_p"])
    return out


def summarize_section_tests(section_tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variable, tissue), sub in section_tests.groupby(["variable", "tissue"], dropna=False):
        for metric in ["axis_spearman_rho", "axis_high_minus_other_mean"]:
            vals = sub[metric].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            exact_two, exact_greater = exact_sign_flip(vals)
            rows.append(
                {
                    "variable": variable,
                    "tissue": tissue,
                    "section_metric": metric,
                    "n_sections": int(len(vals)),
                    "mean_value": float(vals.mean()),
                    "median_value": float(np.median(vals)),
                    "n_positive": int((vals > 0).sum()),
                    "n_negative": int((vals < 0).sum()),
                    "exact_sign_flip_two_sided_p": exact_two,
                    "exact_sign_flip_greater_p": exact_greater,
                    "one_sample_t_p": stats.ttest_1samp(vals, 0.0).pvalue if len(vals) > 1 and vals.std(ddof=1) > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exact_sign_flip_two_sided_fdr"] = bh_fdr(out["exact_sign_flip_two_sided_p"])
    return out


def paired_apa_adjacent_section_deltas(section_tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in sorted(section_tests["variable"].dropna().unique()):
        for metric in ["axis_spearman_rho", "axis_high_minus_other_mean"]:
            sub = section_tests[section_tests["variable"] == variable]
            wide = sub.pivot_table(index="patient", columns="tissue", values=metric, aggfunc="mean").dropna(subset=["APA", "Adjacent"], how="any")
            if wide.empty:
                continue
            delta = wide["APA"] - wide["Adjacent"]
            exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
            rows.append(
                {
                    "variable": variable,
                    "section_metric": metric,
                    "n_pairs": int(delta.shape[0]),
                    "mean_APA_minus_Adjacent": float(delta.mean()),
                    "median_APA_minus_Adjacent": float(delta.median()),
                    "n_positive": int((delta > 0).sum()),
                    "n_negative": int((delta < 0).sum()),
                    "exact_sign_flip_two_sided_p": exact_two,
                    "exact_sign_flip_greater_p": exact_greater,
                    "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exact_sign_flip_two_sided_fdr"] = bh_fdr(out["exact_sign_flip_two_sided_p"])
    return out


def cv_r2(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    ok = np.isfinite(y.to_numpy(float))
    for col in X.columns:
        ok &= np.isfinite(X[col].to_numpy(float))
    X = X.loc[ok]
    y = y.loc[ok]
    if len(y) < 60 or y.std() == 0:
        return np.nan
    n_splits = min(5, max(2, len(y) // 40))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)
    scaler = StandardScaler()
    X_arr = X.to_numpy(float)
    y_arr = y.to_numpy(float)
    for train, test in kf.split(X_arr):
        X_train = scaler.fit_transform(X_arr[train])
        X_test = scaler.transform(X_arr[test])
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_arr[train])
        preds[test] = model.predict(X_test)
    return float(r2_score(y_arr[np.isfinite(preds)], preds[np.isfinite(preds)]))


def neighborhood_regression(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cols = ["log10_total_umi", "ZG_like", "ZF_ZR_like", "intermediate_steroidogenic"]
    local_niche_cols = ["endothelial", "fibroblast_stromal", "wnt_rspo_module", "cxcl12_axis_module", "vascular_stromal_niche_module"]
    neighbor_cols = ["neighbor_endothelial", "neighbor_fibroblast_stromal", "neighbor_wnt_rspo_module", "neighbor_cxcl12_axis_module", "neighbor_vascular_stromal_niche_module"]
    for (gsm, patient, tissue), sub in df.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        y = sub[PRIMARY_AXIS].astype(float)
        base = sub[[c for c in base_cols if c in sub.columns]].astype(float)
        local = sub[[c for c in base_cols + local_niche_cols if c in sub.columns]].astype(float)
        neigh = sub[[c for c in base_cols + local_niche_cols + neighbor_cols if c in sub.columns]].astype(float)
        base_r2 = cv_r2(base, y, seed=100 + int(patient))
        local_r2 = cv_r2(local, y, seed=200 + int(patient))
        neighbor_r2 = cv_r2(neigh, y, seed=300 + int(patient))
        rows.append(
            {
                "geo_accession": gsm,
                "patient": int(patient),
                "tissue": tissue,
                "n_spots": int(sub.shape[0]),
                "base_cv_r2": base_r2,
                "local_niche_cv_r2": local_r2,
                "neighbor_niche_cv_r2": neighbor_r2,
                "local_minus_base_cv_r2": local_r2 - base_r2 if np.isfinite(local_r2) and np.isfinite(base_r2) else np.nan,
                "neighbor_minus_local_cv_r2": neighbor_r2 - local_r2 if np.isfinite(neighbor_r2) and np.isfinite(local_r2) else np.nan,
                "neighbor_minus_base_cv_r2": neighbor_r2 - base_r2 if np.isfinite(neighbor_r2) and np.isfinite(base_r2) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_regression(reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tissue, sub in reg.groupby("tissue"):
        for metric in ["local_minus_base_cv_r2", "neighbor_minus_local_cv_r2", "neighbor_minus_base_cv_r2"]:
            vals = sub[metric].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            exact_two, exact_greater = exact_sign_flip(vals)
            rows.append(
                {
                    "tissue": tissue,
                    "metric": metric,
                    "n_sections": int(len(vals)),
                    "mean_value": float(vals.mean()),
                    "median_value": float(np.median(vals)),
                    "n_positive": int((vals > 0).sum()),
                    "n_negative": int((vals < 0).sum()),
                    "exact_sign_flip_two_sided_p": exact_two,
                    "exact_sign_flip_greater_p": exact_greater,
                }
            )
    return pd.DataFrame(rows)


def safe_fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def make_figure(summary: pd.DataFrame, reg: pd.DataFrame, reg_summary: pd.DataFrame, section_tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    focus = [
        "neighbor_endothelial",
        "neighbor_fibroblast_stromal",
        "neighbor_wnt_rspo_module",
        "neighbor_cxcl12_axis_module",
        "neighbor_vascular_stromal_niche_module",
    ]
    s1 = summary[(summary["tissue"] == "APA") & (summary["section_metric"] == "axis_spearman_rho") & (summary["variable"].isin(focus))].copy()
    s1["variable"] = pd.Categorical(s1["variable"], categories=focus[::-1], ordered=True)
    axes[0, 0].barh(s1["variable"].astype(str), s1["mean_value"], color="#3a7ca5")
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_title("A. APA axis-neighborhood correlations")
    axes[0, 0].set_xlabel("Mean section Spearman rho")
    axes[0, 0].set_ylabel("")

    s2 = summary[(summary["tissue"] == "APA") & (summary["section_metric"] == "axis_high_minus_other_mean") & (summary["variable"].isin(focus))].copy()
    s2["variable"] = pd.Categorical(s2["variable"], categories=focus[::-1], ordered=True)
    colors = ["#2f8f71" if x > 0 else "#b8554f" for x in s2["mean_value"]]
    axes[0, 1].barh(s2["variable"].astype(str), s2["mean_value"], color=colors)
    axes[0, 1].axvline(0, color="black", lw=0.8)
    axes[0, 1].set_title("B. APA axis-high neighborhood enrichment")
    axes[0, 1].set_xlabel("Axis-high minus other spots")
    axes[0, 1].set_ylabel("")

    reg_long = reg.melt(
        id_vars=["geo_accession", "patient", "tissue"],
        value_vars=["local_minus_base_cv_r2", "neighbor_minus_local_cv_r2", "neighbor_minus_base_cv_r2"],
        var_name="metric",
        value_name="delta_cv_r2",
    )
    sns.stripplot(data=reg_long, x="metric", y="delta_cv_r2", hue="tissue", dodge=True, ax=axes[1, 0], size=4, alpha=0.85)
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_title("C. Simplified neighborhood-regression gain")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Delta cross-validated R2")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(title="", fontsize=7)

    lr_focus = [v for v in LR_PAIRS.keys() if v in section_tests["variable"].unique()]
    s4 = summary[(summary["tissue"] == "APA") & (summary["section_metric"] == "axis_high_minus_other_mean") & (summary["variable"].isin(lr_focus))].copy()
    if not s4.empty:
        s4["variable"] = pd.Categorical(s4["variable"], categories=lr_focus[::-1], ordered=True)
        axes[1, 1].barh(s4["variable"].astype(str), s4["mean_value"], color="#8b6bb1")
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("D. Focused LR spatial product enrichment")
    axes[1, 1].set_xlabel("Axis-high minus other spots")
    axes[1, 1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIG / "Fig5_spatial_niche_neighborhood.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Fig5_spatial_niche_neighborhood.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(module_avail: pd.DataFrame, section_summary: pd.DataFrame, paired_delta: pd.DataFrame, reg_summary: pd.DataFrame) -> None:
    def summary_line(variable: str, metric: str, tissue: str = "APA") -> str:
        sub = section_summary[(section_summary["variable"] == variable) & (section_summary["section_metric"] == metric) & (section_summary["tissue"] == tissue)]
        if sub.empty:
            return f"- {variable}: NA."
        r = sub.iloc[0]
        return (
            f"- {variable}: mean {metric} = {safe_fmt(r['mean_value'])}; {int(r['n_positive'])}/{int(r['n_sections'])} sections positive; "
            f"exact p = {safe_fmt(r['exact_sign_flip_two_sided_p'])}."
        )

    reg_line = ""
    sub = reg_summary[(reg_summary["tissue"] == "APA") & (reg_summary["metric"] == "neighbor_minus_local_cv_r2")]
    if not sub.empty:
        r = sub.iloc[0]
        reg_line = (
            f"- Adding neighbor niche features beyond local features changed cross-validated R2 by mean {safe_fmt(r['mean_value'])}; "
            f"{int(r['n_positive'])}/{int(r['n_sections'])} APA sections positive; exact p = {safe_fmt(r['exact_sign_flip_two_sided_p'])}."
        )

    lines = [
        "# Spatial niche neighborhood report 2026-06-02",
        "",
        "## Purpose",
        "",
        "Test whether the APA ZG/intermediate-vs-ZF spatial steroidogenic axis is coupled to neighboring endothelial/stromal and focused WNT/RSPO/CXCL12 niche signals.",
        "",
        "Evidence role: mechanism-generating spatial context. This is not causal cell-cell signaling proof.",
        "",
        "## Targeted gene/module availability",
        "",
    ]
    for _, row in module_avail.iterrows():
        lines.append(f"- {row['module']}: {int(row['n_available'])}/{int(row['n_requested'])} genes available.")
    lines.extend(
        [
            "",
            "## APA section-level axis-neighborhood correlations",
            "",
            summary_line("neighbor_endothelial", "axis_spearman_rho"),
            summary_line("neighbor_fibroblast_stromal", "axis_spearman_rho"),
            summary_line("neighbor_wnt_rspo_module", "axis_spearman_rho"),
            summary_line("neighbor_cxcl12_axis_module", "axis_spearman_rho"),
            summary_line("neighbor_vascular_stromal_niche_module", "axis_spearman_rho"),
            "",
            "## APA axis-high neighborhood enrichment",
            "",
            summary_line("neighbor_endothelial", "axis_high_minus_other_mean"),
            summary_line("neighbor_fibroblast_stromal", "axis_high_minus_other_mean"),
            summary_line("neighbor_wnt_rspo_module", "axis_high_minus_other_mean"),
            summary_line("neighbor_cxcl12_axis_module", "axis_high_minus_other_mean"),
            summary_line("neighbor_vascular_stromal_niche_module", "axis_high_minus_other_mean"),
            "",
            "## Simplified neighborhood regression",
            "",
            reg_line or "- Regression summary unavailable.",
            "",
            "## Claim boundary",
            "",
            "Recommended wording:",
            "",
            "> APA axis-high spatial domains were associated with neighboring fibroblast/stromal deconvolution weights and a focused WNT/RSPO spatial module, with weaker exploratory support for CXCL12-axis coupling, nominating a spatial microenvironment context for the steroidogenic axis.",
            "",
            "Avoid:",
            "",
            "- Endothelial/stromal signaling causes APA steroidogenic rewiring.",
            "- Endothelial marker expression is a robust positive result in this analysis.",
            "- Ligand-receptor products prove active signaling.",
            "- Deconvolution weights equal histologic cell abundance.",
            "",
            "## Outputs",
            "",
            "- `results/spatial_niche_neighborhood_20260602/GSE274314_spatial_niche_spot_level.csv`",
            "- `results/spatial_niche_neighborhood_20260602/GSE274314_spatial_niche_section_tests.csv`",
            "- `results/spatial_niche_neighborhood_20260602/GSE274314_spatial_niche_section_summary.csv`",
            "- `results/spatial_niche_neighborhood_20260602/GSE274314_spatial_niche_neighborhood_regression.csv`",
            "- `results/spatial_niche_neighborhood_20260602/figures/Fig5_spatial_niche_neighborhood.png/.pdf`",
        ]
    )
    (PROJECT / "spatial_niche_neighborhood_report_20260602.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage2 = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    stage2["log10_total_umi"] = np.log10(stage2["total_umi"].astype(float).clip(lower=1.0))
    expr, gene_avail = extract_targeted_spatial_expression(TARGET_GENES)
    merged = stage2.merge(
        expr.drop(columns=["lowres_x_raw", "lowres_y_raw"], errors="ignore"),
        on=["spot_id", "geo_accession", "barcode"],
        how="left",
        suffixes=("", "_target"),
    )
    for gene in TARGET_GENES:
        target_col = f"{gene}_target"
        if target_col in merged.columns:
            merged[gene] = merged[target_col]
            merged = merged.drop(columns=[target_col])
    merged, module_avail = add_niche_modules(merged)
    neighbor_input = [
        "endothelial",
        "fibroblast_stromal",
        "immune",
        "ZG_like",
        "ZF_ZR_like",
        "intermediate_steroidogenic",
        "endothelial_marker_module",
        "stromal_marker_module",
        "wnt_rspo_module",
        "cxcl12_axis_module",
        "vascular_stromal_niche_module",
        "RSPO3",
        "WNT4",
        "CXCL12",
    ]
    neighbor_input = [c for c in neighbor_input if c in merged.columns]
    merged = add_neighbor_features(merged, neighbor_input, k=6)
    merged = add_lr_products(merged)

    tested_vars = [v for v in NEIGHBOR_VARIABLES if v in merged.columns] + [v for v in LR_PAIRS if v in merged.columns]
    section_tests = section_spatial_tests(merged, tested_vars)
    section_summary = summarize_section_tests(section_tests)
    paired_delta = paired_apa_adjacent_section_deltas(section_tests)
    reg = neighborhood_regression(merged)
    reg_summary = summarize_regression(reg)

    gene_avail.to_csv(OUT / "GSE274314_spatial_niche_target_gene_availability_by_section.csv", index=False, encoding="utf-8-sig")
    module_avail.to_csv(OUT / "GSE274314_spatial_niche_module_gene_availability.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT / "GSE274314_spatial_niche_spot_level.csv", index=False, encoding="utf-8-sig")
    section_tests.to_csv(OUT / "GSE274314_spatial_niche_section_tests.csv", index=False, encoding="utf-8-sig")
    section_summary.to_csv(OUT / "GSE274314_spatial_niche_section_summary.csv", index=False, encoding="utf-8-sig")
    paired_delta.to_csv(OUT / "GSE274314_spatial_niche_APA_adjacent_paired_section_delta.csv", index=False, encoding="utf-8-sig")
    reg.to_csv(OUT / "GSE274314_spatial_niche_neighborhood_regression.csv", index=False, encoding="utf-8-sig")
    reg_summary.to_csv(OUT / "GSE274314_spatial_niche_neighborhood_regression_summary.csv", index=False, encoding="utf-8-sig")

    make_figure(section_summary, reg, reg_summary, section_tests)
    write_report(module_avail, section_summary, paired_delta, reg_summary)

    print("APA section summary, focused variables:")
    focus = section_summary[(section_summary["tissue"] == "APA") & (section_summary["variable"].isin(NEIGHBOR_VARIABLES))]
    print(focus.sort_values(["section_metric", "exact_sign_flip_two_sided_p"]).to_string(index=False))
    print("Regression summary:")
    print(reg_summary.to_string(index=False))
    print("Spatial niche neighborhood analysis complete.")


if __name__ == "__main__":
    main()
