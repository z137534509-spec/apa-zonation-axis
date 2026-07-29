from __future__ import annotations

import gzip
import io
import itertools
import json
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.io import mmread
from scipy.sparse import coo_matrix


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
STAGE2 = RESULTS / "stage2_zonation_axis"
OUT = RESULTS / "candidate_regulon_activity_20260602"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

REGULON_URL = (
    "https://omnipathdb.org/interactions?"
    "datasets=dorothea&organisms=9606&genesymbols=yes&fields=sources,references,curation_effort"
)
REGULON_RAW = RAW / "omnipath" / "omnipath_dorothea_interactions_20260602.tsv"

PRIMARY_AXIS = "ZG_intermediate_vs_ZF_axis_score"
MIN_TARGETS_AVAILABLE = 5

CANDIDATE_TFS = [
    "NR5A1",
    "NR4A1",
    "NR4A2",
    "NR4A3",
    "GATA6",
    "PBX1",
    "WT1",
    "TCF7",
    "TCF7L1",
    "TCF7L2",
    "LEF1",
    "CREB1",
    "ATF3",
    "ATF4",
    "NR0B1",
    "FOXO1",
    "CEBPB",
]


def ensure_regulon_file() -> None:
    REGULON_RAW.parent.mkdir(parents=True, exist_ok=True)
    if REGULON_RAW.exists() and REGULON_RAW.stat().st_size > 0:
        return
    print("Downloading OmniPath/DoRothEA interactions...")
    urlretrieve(REGULON_URL, REGULON_RAW)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def infer_dorothea_confidence(sources: str) -> str:
    sources = str(sources)
    for conf in ["A", "B", "C", "D", "E"]:
        if f"DoRothEA-{conf}" in sources:
            return conf
    if "DoRothEA" in sources:
        return "unlabeled"
    return "not_explicit"


def load_candidate_regulons() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_regulon_file()
    raw = pd.read_csv(REGULON_RAW, sep="\t")
    raw.columns = [str(c) for c in raw.columns]
    tf_col = "source_genesymbol" if "source_genesymbol" in raw.columns else "source"
    target_col = "target_genesymbol" if "target_genesymbol" in raw.columns else "target"
    raw["tf"] = raw[tf_col].astype(str).str.upper()
    raw["target_gene"] = raw[target_col].astype(str).str.upper()
    raw["dorothea_confidence_inferred"] = raw.get("sources", "").astype(str).map(infer_dorothea_confidence)
    raw = raw[raw["tf"].isin(CANDIDATE_TFS)].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw["is_stimulation_bool"] = raw["is_stimulation"].map(parse_bool) if "is_stimulation" in raw.columns else False
    raw["is_inhibition_bool"] = raw["is_inhibition"].map(parse_bool) if "is_inhibition" in raw.columns else False
    raw["target_sign"] = np.where(raw["is_inhibition_bool"] & ~raw["is_stimulation_bool"], -1, np.where(raw["is_stimulation_bool"] & ~raw["is_inhibition_bool"], 1, np.nan))
    raw = raw.dropna(subset=["target_sign"])
    raw = raw[raw["target_gene"].str.match(r"^[A-Z0-9_.-]+$")]
    raw = raw.drop_duplicates(subset=["tf", "target_gene", "target_sign"])
    # If the same TF-target appears with conflicting signs, drop it rather than forcing direction.
    conflict = raw.groupby(["tf", "target_gene"])["target_sign"].nunique().reset_index()
    conflict = conflict[conflict["target_sign"] > 1][["tf", "target_gene"]]
    if not conflict.empty:
        raw = raw.merge(conflict.assign(conflict=True), on=["tf", "target_gene"], how="left")
        raw = raw[raw["conflict"].isna()].drop(columns=["conflict"])
    summary = (
        raw.groupby("tf", as_index=False)
        .agg(
            n_signed_targets=("target_gene", "nunique"),
            n_activating_targets=("target_sign", lambda x: int((x > 0).sum())),
            n_inhibitory_targets=("target_sign", lambda x: int((x < 0).sum())),
            inferred_confidence=("dorothea_confidence_inferred", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values("n_signed_targets", ascending=False)
    )
    return raw, summary


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


def extract_spatial_gene_expression(target_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    avail_rows = []
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
            avail_rows.append({"geo_accession": gsm, "n_requested": len(target_genes), "n_available": len(present), "missing_genes": ";".join(missing)})
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
            df["lowres_x"] = df["pxl_col_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            df["lowres_y"] = df["pxl_row_in_fullres"] * float(scale.get("tissue_lowres_scalef", 1.0))
            df["spot_id"] = df["geo_accession"] + ":" + df["barcode"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(avail_rows)


def zscore_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for col in cols:
        vals = df[col].astype(float)
        sd = vals.std(ddof=0)
        out[col] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else pd.Series(0.0, index=df.index)
    return pd.DataFrame(out, index=df.index)


def score_regulons(expr: pd.DataFrame, regulon: pd.DataFrame, id_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_genes = sorted(set(regulon["target_gene"]).intersection(expr.columns))
    z = zscore_columns(expr, target_genes)
    score = expr[id_cols].copy()
    rows = []
    for tf in CANDIDATE_TFS:
        sub = regulon[regulon["tf"] == tf].copy()
        present = sub[sub["target_gene"].isin(z.columns)].copy()
        if present["target_gene"].nunique() < MIN_TARGETS_AVAILABLE:
            score[f"{tf}_signed_regulon_score"] = np.nan
            rows.append(
                {
                    "tf": tf,
                    "n_targets_total": int(sub["target_gene"].nunique()),
                    "n_targets_available": int(present["target_gene"].nunique()),
                    "status": "too_few_available_targets",
                }
            )
            continue
        vals = []
        weights = []
        for _, row in present.drop_duplicates(["target_gene"]).iterrows():
            vals.append(z[row["target_gene"]].to_numpy(float) * float(row["target_sign"]))
            weights.append(1.0)
        mat = np.vstack(vals)
        score[f"{tf}_signed_regulon_score"] = np.nanmean(mat, axis=0)
        rows.append(
            {
                "tf": tf,
                "n_targets_total": int(sub["target_gene"].nunique()),
                "n_targets_available": int(present["target_gene"].nunique()),
                "status": "scored",
                "available_targets": ";".join(sorted(present["target_gene"].unique())),
            }
        )
    return score, pd.DataFrame(rows)


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


def section_regulon_tests(spots: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    section_mean_rows = []
    for (gsm, patient, tissue), sub in spots.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        axis = sub[PRIMARY_AXIS].to_numpy(float)
        high = sub["axis_top_quartile_roi"].astype(bool).to_numpy()
        section_record = {"geo_accession": gsm, "patient": int(patient), "tissue": tissue}
        for score_col in score_cols:
            vals = sub[score_col].to_numpy(float)
            section_record[score_col] = float(np.nanmean(vals))
            ok = np.isfinite(axis) & np.isfinite(vals)
            if ok.sum() < 30 or np.nanstd(vals[ok]) == 0 or np.nanstd(axis[ok]) == 0:
                continue
            rho, rho_p = stats.spearmanr(axis[ok], vals[ok])
            high_ok = high & ok
            low_ok = (~high) & ok
            diff = np.nan
            if high_ok.sum() >= 5 and low_ok.sum() >= 5:
                diff = float(vals[high_ok].mean() - vals[low_ok].mean())
            rows.append(
                {
                    "geo_accession": gsm,
                    "patient": int(patient),
                    "tissue": tissue,
                    "tf": score_col.replace("_signed_regulon_score", ""),
                    "score_col": score_col,
                    "n_spots": int(ok.sum()),
                    "axis_spearman_rho": float(rho),
                    "axis_spearman_p": float(rho_p),
                    "axis_high_minus_other_mean": diff,
                    "n_axis_high": int(high_ok.sum()),
                }
            )
        section_mean_rows.append(section_record)
    section_tests = pd.DataFrame(rows)
    section_means = pd.DataFrame(section_mean_rows)

    paired_rows = []
    for score_col in score_cols:
        wide = section_means.pivot_table(index="patient", columns="tissue", values=score_col, aggfunc="mean").dropna(subset=["APA", "Adjacent"], how="any")
        if wide.empty:
            continue
        delta = wide["APA"] - wide["Adjacent"]
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        paired_rows.append(
            {
                "tf": score_col.replace("_signed_regulon_score", ""),
                "score_col": score_col,
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
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["exact_sign_flip_two_sided_fdr"] = bh_fdr(paired["exact_sign_flip_two_sided_p"])
    return section_tests, section_means, paired


def summarize_section_tests(section_tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, tissue), sub in section_tests.groupby(["tf", "tissue"], dropna=False):
        for metric in ["axis_spearman_rho", "axis_high_minus_other_mean"]:
            vals = sub[metric].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            exact_two, exact_greater = exact_sign_flip(vals)
            rows.append(
                {
                    "tf": tf,
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


def bulk_regulon_tests(regulon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expr_path = PROCESSED / "GSE60042_gene_expression_mean_collapsed.csv"
    meta_path = PROCESSED / "GSE60042_sample_metadata.csv"
    if not expr_path.exists() or not meta_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    expr = pd.read_csv(expr_path)
    expr["gene_symbol"] = expr["gene_symbol"].astype(str).str.upper()
    expr = expr.groupby("gene_symbol", as_index=False).mean(numeric_only=True).set_index("gene_symbol")
    meta = pd.read_csv(meta_path)
    sample_cols = [c for c in expr.columns if c in set(meta["geo_accession"])]
    sample_expr = expr[sample_cols].T
    sample_expr.index.name = "geo_accession"
    sample_expr = sample_expr.reset_index().merge(meta[["geo_accession", "case", "tissue"]], on="geo_accession", how="left")
    score, coverage = score_regulons(sample_expr, regulon, id_cols=["geo_accession", "case", "tissue"])
    score_cols = [c for c in score.columns if c.endswith("_signed_regulon_score")]
    rows = []
    for col in score_cols:
        wide = score.pivot_table(index="case", columns="tissue", values=col, aggfunc="mean")
        if not {"APA", "AAG"}.issubset(set(wide.columns)):
            continue
        wide = wide.dropna(subset=["APA", "AAG"], how="any")
        if wide.empty:
            continue
        delta = wide["APA"] - wide["AAG"]
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        rows.append(
            {
                "tf": col.replace("_signed_regulon_score", ""),
                "score_col": col,
                "n_pairs": int(delta.shape[0]),
                "mean_APA_minus_AAG": float(delta.mean()),
                "median_APA_minus_AAG": float(delta.median()),
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
            }
        )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["exact_sign_flip_two_sided_fdr"] = bh_fdr(tests["exact_sign_flip_two_sided_p"])
    return score, coverage, tests


def safe_fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def make_figure(summary: pd.DataFrame, paired: pd.DataFrame, bulk_tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    apa_corr = summary[(summary["tissue"] == "APA") & (summary["section_metric"] == "axis_spearman_rho")].copy()
    apa_corr = apa_corr.sort_values(["exact_sign_flip_two_sided_p", "mean_value"]).head(8)
    axes[0, 0].barh(apa_corr["tf"], apa_corr["mean_value"], color="#3a7ca5")
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_title("A. APA regulon-axis correlation")
    axes[0, 0].set_xlabel("Mean section Spearman rho")
    axes[0, 0].set_ylabel("")

    apa_high = summary[(summary["tissue"] == "APA") & (summary["section_metric"] == "axis_high_minus_other_mean")].copy()
    apa_high = apa_high.sort_values(["exact_sign_flip_two_sided_p", "mean_value"]).head(8)
    axes[0, 1].barh(apa_high["tf"], apa_high["mean_value"], color="#2f8f71")
    axes[0, 1].axvline(0, color="black", lw=0.8)
    axes[0, 1].set_title("B. APA axis-high regulon enrichment")
    axes[0, 1].set_xlabel("Axis-high minus other spots")
    axes[0, 1].set_ylabel("")

    p = paired.sort_values(["exact_sign_flip_two_sided_p", "mean_APA_minus_Adjacent"]).head(8).copy()
    axes[1, 0].barh(p["tf"], p["mean_APA_minus_Adjacent"], color="#8b6bb1")
    axes[1, 0].axvline(0, color="black", lw=0.8)
    axes[1, 0].set_title("C. Spatial APA-adjacent regulon delta")
    axes[1, 0].set_xlabel("APA - Adjacent section mean")
    axes[1, 0].set_ylabel("")

    if not bulk_tests.empty:
        b = bulk_tests.sort_values(["exact_sign_flip_two_sided_p", "mean_APA_minus_AAG"]).head(8).copy()
        axes[1, 1].barh(b["tf"], b["mean_APA_minus_AAG"], color="#d08b3e")
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("D. GSE60042 bulk regulon transfer")
    axes[1, 1].set_xlabel("APA - AAG paired mean")
    axes[1, 1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIG / "Supp_candidate_regulon_activity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Supp_candidate_regulon_activity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(reg_summary: pd.DataFrame, coverage: pd.DataFrame, section_summary: pd.DataFrame, paired: pd.DataFrame, bulk_tests: pd.DataFrame) -> None:
    def top_lines(df: pd.DataFrame, label: str, value_col: str) -> list[str]:
        if df.empty:
            return [f"- {label}: NA."]
        top = df.sort_values(["exact_sign_flip_two_sided_p", value_col], ascending=[True, False]).head(5)
        out = []
        for _, r in top.iterrows():
            out.append(
                f"- {label} {r['tf']}: mean = {safe_fmt(r[value_col])}; positives = {int(r['n_positive'])}/{int(r.get('n_sections', r.get('n_pairs')))}; exact p = {safe_fmt(r['exact_sign_flip_two_sided_p'])}."
            )
        return out

    apa_corr = section_summary[(section_summary["tissue"] == "APA") & (section_summary["section_metric"] == "axis_spearman_rho")]
    apa_high = section_summary[(section_summary["tissue"] == "APA") & (section_summary["section_metric"] == "axis_high_minus_other_mean")]

    lines = [
        "# Candidate regulon activity report 2026-06-02",
        "",
        "## Purpose",
        "",
        "Estimate candidate transcription-factor regulon activity using signed OmniPath/DoRothEA target sets and test whether these activities spatially covary with the APA ZG/intermediate-vs-ZF axis.",
        "",
        "Evidence role: candidate regulatory context. This is not ChIP-seq, ATAC-seq, perturbation, or direct TF regulation proof.",
        "",
        "## Regulon resource audit",
        "",
        f"- Candidate TFs requested: {len(CANDIDATE_TFS)}.",
        f"- TFs with signed target sets in OmniPath/DoRothEA: {reg_summary.shape[0]}.",
        f"- TFs scored in spatial data: {int((coverage['status'] == 'scored').sum())}.",
        "",
        "## APA regulon-axis spatial association",
        "",
    ]
    lines.extend(top_lines(apa_corr, "axis correlation", "mean_value"))
    lines.extend(
        [
            "",
            "Interpretation: the positive spatial signal is mainly Wnt/TCF-family signed target activity (LEF1/TCF7L2). NR5A1 and NR4A1 signed target scores are inversely associated with the axis, so they should not be written as activated master regulators in the APA axis-high domain. NR4A2/NR4A3 did not have usable signed DoRothEA target sets under this workflow.",
        ]
    )
    lines.extend(["", "## APA axis-high regulon enrichment", ""])
    lines.extend(top_lines(apa_high, "axis-high enrichment", "mean_value"))
    lines.extend(["", "## Spatial paired APA-adjacent section deltas", ""])
    lines.extend(top_lines(paired, "APA-adjacent delta", "mean_APA_minus_Adjacent"))
    lines.extend(["", "## Independent GSE60042 bulk regulon transfer", ""])
    if bulk_tests.empty:
        lines.append("- Bulk regulon transfer unavailable.")
    else:
        lines.extend(top_lines(bulk_tests, "bulk APA-AAG delta", "mean_APA_minus_AAG"))
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Recommended wording:",
            "",
            "> Signed target-set scoring nominated spatially coordinated candidate regulon activities that covary with the APA steroidogenic axis; these results generate regulatory hypotheses rather than proving direct TF control.",
            "",
            "Avoid:",
            "",
            "- NR4A2 directly regulates CYP11B2 in APA.",
            "- SCENIC/DoRothEA proves a master switch.",
            "- CTNNB1 is a transcription factor regulon in this analysis.",
            "",
            "## Outputs",
            "",
            "- `results/candidate_regulon_activity_20260602/GSE274314_candidate_regulon_spot_scores.csv`",
            "- `results/candidate_regulon_activity_20260602/GSE274314_candidate_regulon_section_summary.csv`",
            "- `results/candidate_regulon_activity_20260602/GSE274314_candidate_regulon_APA_adjacent_paired_tests.csv`",
            "- `results/candidate_regulon_activity_20260602/GSE60042_candidate_regulon_bulk_tests.csv`",
            "- `results/candidate_regulon_activity_20260602/figures/Supp_candidate_regulon_activity.png/.pdf`",
        ]
    )
    (PROJECT / "candidate_regulon_activity_report_20260602.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    regulon, regulon_summary = load_candidate_regulons()
    if regulon.empty:
        raise RuntimeError("No candidate regulons were loaded.")
    regulon.to_csv(OUT / "omnipath_dorothea_candidate_signed_targets.csv", index=False, encoding="utf-8-sig")
    regulon_summary.to_csv(OUT / "omnipath_dorothea_candidate_regulon_summary.csv", index=False, encoding="utf-8-sig")

    target_genes = sorted(set(regulon["target_gene"]).union(CANDIDATE_TFS))
    expr, spatial_gene_avail = extract_spatial_gene_expression(target_genes)
    stage2 = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    keep_stage2 = [
        "spot_id",
        "geo_accession",
        "barcode",
        "patient",
        "tissue",
        "genotype",
        PRIMARY_AXIS,
        "axis_top_quartile_roi",
        "total_umi",
        "n_detected_genes",
    ]
    merged = stage2[keep_stage2].merge(expr.drop(columns=["lowres_x", "lowres_y"], errors="ignore"), on=["spot_id", "geo_accession", "barcode"], how="left")
    scores, coverage = score_regulons(merged, regulon, id_cols=keep_stage2)
    score_cols = [c for c in scores.columns if c.endswith("_signed_regulon_score") and scores[c].notna().any()]
    section_tests, section_means, paired = section_regulon_tests(scores, score_cols)
    section_summary = summarize_section_tests(section_tests)

    bulk_scores, bulk_coverage, bulk_tests = bulk_regulon_tests(regulon)

    spatial_gene_avail.to_csv(OUT / "GSE274314_candidate_regulon_target_gene_availability_by_section.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT / "GSE274314_candidate_regulon_coverage.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT / "GSE274314_candidate_regulon_spot_scores.csv", index=False, encoding="utf-8-sig")
    section_tests.to_csv(OUT / "GSE274314_candidate_regulon_section_tests.csv", index=False, encoding="utf-8-sig")
    section_summary.to_csv(OUT / "GSE274314_candidate_regulon_section_summary.csv", index=False, encoding="utf-8-sig")
    section_means.to_csv(OUT / "GSE274314_candidate_regulon_section_mean_scores.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "GSE274314_candidate_regulon_APA_adjacent_paired_tests.csv", index=False, encoding="utf-8-sig")
    if not bulk_scores.empty:
        bulk_scores.to_csv(OUT / "GSE60042_candidate_regulon_bulk_scores.csv", index=False, encoding="utf-8-sig")
        bulk_coverage.to_csv(OUT / "GSE60042_candidate_regulon_coverage.csv", index=False, encoding="utf-8-sig")
        bulk_tests.to_csv(OUT / "GSE60042_candidate_regulon_bulk_tests.csv", index=False, encoding="utf-8-sig")

    make_figure(section_summary, paired, bulk_tests)
    write_report(regulon_summary, coverage, section_summary, paired, bulk_tests)

    print("Candidate regulon coverage:")
    print(coverage.to_string(index=False))
    print("APA section summary top:")
    top = section_summary[section_summary["tissue"] == "APA"].sort_values(["exact_sign_flip_two_sided_p", "mean_value"], ascending=[True, False]).head(20)
    print(top.to_string(index=False))
    print("Bulk transfer top:")
    if not bulk_tests.empty:
        print(bulk_tests.sort_values(["exact_sign_flip_two_sided_p", "mean_APA_minus_AAG"], ascending=[True, False]).head(20).to_string(index=False))
    print("Candidate regulon activity analysis complete.")


if __name__ == "__main__":
    main()
