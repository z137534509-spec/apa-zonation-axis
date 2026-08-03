from __future__ import annotations

import gzip
import io
import itertools
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
OUT = RESULTS / "external_transcriptome_methylome_20260602"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260602)
N_BOOT = 10000

URLS = {
    RAW / "GSE60042" / "GSE60042_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60042/matrix/GSE60042_series_matrix.txt.gz",
    RAW / "GSE60043" / "GSE60043_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60043/matrix/GSE60043_series_matrix.txt.gz",
    RAW / "GSE60042" / "GPL14550_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL14nnn/GPL14550/soft/GPL14550_family.soft.gz",
    RAW / "GSE60043" / "GPL13534_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/soft/GPL13534_family.soft.gz",
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
    "medulla_chromaffin_control": ["TH", "DBH", "PNMT", "CHGA", "CHGB", "SLC18A1"],
    "immune_control": ["PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1"],
    "stromal_control": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "endothelial_control": ["PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"],
    "housekeeping_control": ["ACTB", "GAPDH", "RPLP0", "RPL13A", "RPS18", "B2M", "HPRT1", "PGK1", "TBP", "PPIA"],
}

KEY_METRICS = [
    "ZG_aldosterone_program",
    "ZF_cortisol_program",
    "ZR_androgen_program",
    "intermediate_steroidogenic_program",
    "generic_steroidogenesis_program",
    "ZG_vs_ZF_axis_score",
    "ZG_intermediate_vs_ZF_axis_score",
    "aldosterone_specificity_score",
    "housekeeping_control",
]

NICHE_GENES = [
    "RSPO3",
    "RSPO2",
    "LGR5",
    "WNT4",
    "WNT2",
    "WNT5A",
    "FZD1",
    "FZD2",
    "FZD4",
    "FZD5",
    "FZD6",
    "LRP5",
    "LRP6",
    "CXCL12",
    "CXCR4",
    "ACKR3",
    "PECAM1",
    "VWF",
    "KDR",
    "COL1A1",
    "COL1A2",
    "DCN",
    "LUM",
]

PRIORITY_GENES = sorted(
    set(
        SIGNATURES["ZG_aldosterone_program"]
        + SIGNATURES["intermediate_steroidogenic_program"]
        + ["CYP11B2", "MC2R", "HSD3B2", "STAR", "NR4A2"]
        + NICHE_GENES
    )
)

PROMOTER_GROUPS_STRICT = {"TSS1500", "TSS200"}
PROMOTER_GROUPS_EXTENDED = {"TSS1500", "TSS200", "5'UTR", "1STEXON"}


def ensure_input_files() -> None:
    for path, url in URLS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {path.name}...")
        urlretrieve(url, path)


def clean_geo_value(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def split_geo_line(line: str) -> list[str]:
    return [clean_geo_value(x) for x in line.rstrip("\n").split("\t")[1:]]


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: dict[str, list[list[str]]] = defaultdict(list)
    table_lines: list[str] = []
    in_table = False

    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
                continue
            if line.startswith("!Sample_"):
                key = line.split("\t", 1)[0].replace("!", "")
                meta_rows[key].append(split_geo_line(line))

    accessions = meta_rows.get("Sample_geo_accession", [[]])[0]
    titles = meta_rows.get("Sample_title", [[]])[0]
    meta = pd.DataFrame({"geo_accession": accessions, "title": titles})
    for char_values in meta_rows.get("Sample_characteristics_ch1", []):
        parsed = []
        key_name = None
        for item in char_values:
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("/", "_")
                value = value.strip()
            else:
                key, value = "characteristic", item.strip()
            if key_name is None:
                key_name = key
            parsed.append(value)
        if key_name:
            meta[key_name] = parsed

    if not table_lines:
        raise RuntimeError(f"No expression table found in {path}")
    matrix = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t")
    matrix.columns = [str(c).strip('"') for c in matrix.columns]
    matrix["ID_REF"] = matrix["ID_REF"].astype(str).str.strip('"')
    matrix = matrix.set_index("ID_REF")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return meta, matrix


def add_case_tissue(meta: pd.DataFrame, dataset: str) -> pd.DataFrame:
    meta = meta.copy()
    case = meta["title"].astype(str).str.extract(r"(Case\d+)", expand=False)
    meta["case"] = case.str.replace("Case", "", regex=False).astype(int)
    title = meta["title"].astype(str).str.lower()
    meta["tissue"] = pd.Series(index=meta.index, dtype="object")
    meta.loc[title.str.contains("adjacent adrenal gland|aag", regex=True), "tissue"] = "AAG"
    meta.loc[title.str.contains("apa|adenoma", regex=True), "tissue"] = "APA"
    meta["dataset"] = dataset
    return meta


def split_gene_symbols(value: str) -> list[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value or value in {"---", "-", "NA", "nan", "None"}:
        return []
    parts = re.split(r"[;,/|]+", value)
    genes = []
    for part in parts:
        gene = part.strip().upper()
        if not gene or gene in {"---", "NA", "NAN", "NONE", "CONTROL", "POS", "NEG"}:
            continue
        if re.match(r"^[A-Z0-9_.-]+$", gene):
            genes.append(gene)
    return sorted(set(genes))


def parse_agilent_gene_map(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        in_table = False
        header: list[str] | None = None
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                header = next(handle).rstrip("\n").split("\t")
                continue
            if not in_table:
                continue
            if line.startswith("!platform_table_end"):
                break
            values = line.rstrip("\n").split("\t")
            if header is None or len(values) != len(header):
                continue
            row = dict(zip(header, values))
            if row.get("CONTROL_TYPE", "").lower() == "pos":
                continue
            symbols = split_gene_symbols(row.get("GENE_SYMBOL", ""))
            if not symbols:
                symbols = split_gene_symbols(row.get("GENE", ""))
            for gene in symbols:
                rows.append({"ID_REF": str(row.get("ID", "")), "gene_symbol": gene})
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(PROCESSED / "GPL14550_probe_gene_map.csv", index=False, encoding="utf-8-sig")
    return out


def collapse_probe_to_gene(matrix: pd.DataFrame, gene_map: pd.DataFrame) -> pd.DataFrame:
    joined = matrix.join(gene_map.set_index("ID_REF"), how="inner")
    value_cols = [c for c in joined.columns if c != "gene_symbol"]
    gene_expr = joined.groupby("gene_symbol", sort=True)[value_cols].mean()
    return gene_expr


def zscore_by_gene(gene_expr: pd.DataFrame) -> pd.DataFrame:
    z = gene_expr.T.copy()
    for gene in z.columns:
        vals = z[gene].astype(float)
        sd = vals.std(ddof=0)
        z[gene] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return z


def score_expression(gene_expr: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_by_gene(gene_expr)
    scores = pd.DataFrame(index=z.index)
    availability = []
    for name, genes in SIGNATURES.items():
        genes_u = [g.upper() for g in genes]
        available = [g for g in genes_u if g in z.columns]
        missing = [g for g in genes_u if g not in z.columns]
        scores[name] = z[available].mean(axis=1) if available else np.nan
        availability.append(
            {
                "dataset": "GSE60042",
                "signature": name,
                "n_requested": len(genes_u),
                "n_available": len(available),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
            }
        )
    scores["ZG_vs_ZF_axis_score"] = scores["ZG_aldosterone_program"] - scores["ZF_cortisol_program"]
    scores["ZG_intermediate_vs_ZF_axis_score"] = scores[["ZG_aldosterone_program", "intermediate_steroidogenic_program"]].mean(axis=1) - scores[
        "ZF_cortisol_program"
    ]
    scores["aldosterone_specificity_score"] = scores["ZG_aldosterone_program"] - scores["generic_steroidogenesis_program"]
    scores = scores.reset_index().rename(columns={"index": "geo_accession"}).merge(meta, on="geo_accession", how="left")
    availability_df = pd.DataFrame(availability)
    return scores, availability_df


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


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    boots = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


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


def paired_metric_tests(scores: pd.DataFrame, metrics: list[str], dataset: str, tissue_a: str = "APA", tissue_b: str = "AAG") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    delta_rows = []
    for metric in metrics:
        if metric not in scores.columns:
            continue
        wide = scores.pivot_table(index="case", columns="tissue", values=metric, aggfunc="mean")
        if not {tissue_a, tissue_b}.issubset(set(wide.columns)):
            continue
        wide = wide.dropna(subset=[tissue_a, tissue_b])
        delta = (wide[tissue_a] - wide[tissue_b]).astype(float)
        if delta.empty:
            continue
        ci_l, ci_u = bootstrap_ci(delta.to_numpy())
        exact_two, exact_greater = exact_sign_flip(delta.to_numpy())
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "comparison": f"{tissue_a} vs {tissue_b}",
                "n_pairs": int(delta.shape[0]),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "ci95_low": ci_l,
                "ci95_high": ci_u,
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
                "paired_t_p": stats.ttest_1samp(delta, 0.0).pvalue if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
            }
        )
        for case, value in delta.items():
            delta_rows.append({"dataset": dataset, "metric": metric, "case": case, "delta_APA_minus_AAG": float(value)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exact_sign_flip_two_sided_fdr"] = bh_fdr(out["exact_sign_flip_two_sided_p"])
        out["wilcoxon_fdr"] = bh_fdr(out["wilcoxon_p"])
    return out, pd.DataFrame(delta_rows)


def parse_methylation_probe_map(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        in_table = False
        header: list[str] | None = None
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                header = next(handle).rstrip("\n").split("\t")
                continue
            if not in_table:
                continue
            if line.startswith("!platform_table_end"):
                break
            values = line.rstrip("\n").split("\t")
            if header is None or len(values) != len(header):
                continue
            row = dict(zip(header, values))
            probe_id = str(row.get("ID", "")).strip()
            gene_field = str(row.get("UCSC_RefGene_Name", "")).strip()
            group_field = str(row.get("UCSC_RefGene_Group", "")).strip()
            if not probe_id or not gene_field:
                continue
            genes = [g.strip().upper() for g in gene_field.split(";") if g.strip()]
            groups = [g.strip().upper() for g in group_field.split(";") if g.strip()]
            if not groups:
                groups = [""]
            if len(groups) == len(genes):
                pairs = zip(genes, groups)
            else:
                pairs = itertools.product(genes, groups)
            for gene, group in pairs:
                if not gene or gene in {"---", "NA", "NAN"}:
                    continue
                rows.append(
                    {
                        "ID_REF": probe_id,
                        "gene_symbol": gene,
                        "ucsc_refgene_group": group,
                        "strict_tss_promoter": group in PROMOTER_GROUPS_STRICT,
                        "extended_promoter": group in PROMOTER_GROUPS_EXTENDED,
                    }
                )
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(PROCESSED / "GPL13534_probe_gene_promoter_map.csv", index=False, encoding="utf-8-sig")
    return out


def aggregate_methylation_by_gene(beta: pd.DataFrame, probe_map: pd.DataFrame, promoter_mode: str = "strict") -> tuple[pd.DataFrame, pd.DataFrame]:
    if promoter_mode == "strict":
        keep = probe_map[probe_map["strict_tss_promoter"]].copy()
    elif promoter_mode == "extended":
        keep = probe_map[probe_map["extended_promoter"]].copy()
    else:
        keep = probe_map.copy()
    keep = keep[keep["ID_REF"].isin(beta.index)].copy()
    joined = beta.join(keep.set_index("ID_REF"), how="inner")
    value_cols = [c for c in beta.columns if c in joined.columns]
    gene_beta = joined.groupby("gene_symbol", sort=True)[value_cols].mean()
    coverage = (
        joined.reset_index()
        .groupby("gene_symbol", as_index=False)
        .agg(n_probes=("ID_REF", "nunique"), promoter_groups=("ucsc_refgene_group", lambda x: ";".join(sorted(set(map(str, x))))))
    )
    return gene_beta, coverage


def methylation_hypomethylation_scores(gene_beta: pd.DataFrame, meta: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_by_gene(gene_beta)
    scores = pd.DataFrame(index=z.index)
    scores["all_strict_promoter_hypomethylation_background"] = -z.mean(axis=1)
    availability = []
    meth_signatures = {
        "ZG_intermediate_core_promoter_hypomethylation": sorted(set(SIGNATURES["ZG_aldosterone_program"] + SIGNATURES["intermediate_steroidogenic_program"])),
        "ZG_aldosterone_promoter_hypomethylation": SIGNATURES["ZG_aldosterone_program"],
        "intermediate_steroidogenic_promoter_hypomethylation": SIGNATURES["intermediate_steroidogenic_program"],
        "ZF_cortisol_promoter_hypomethylation": SIGNATURES["ZF_cortisol_program"],
        "priority_gene_promoter_hypomethylation": PRIORITY_GENES,
        "housekeeping_promoter_hypomethylation_control": SIGNATURES["housekeeping_control"],
    }
    available_gene_set = set(gene_beta.index.astype(str).str.upper())
    for name, genes in meth_signatures.items():
        genes_u = [g.upper() for g in genes]
        available = [g for g in genes_u if g in z.columns]
        missing = [g for g in genes_u if g not in z.columns]
        scores[name] = -z[available].mean(axis=1) if available else np.nan
        availability.append(
            {
                "dataset": "GSE60043",
                "signature": name,
                "n_requested": len(genes_u),
                "n_available": len(available),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
                "n_with_promoter_probe": len(set(available).intersection(available_gene_set)),
            }
        )
    scores["ZG_intermediate_core_promoter_hypomethylation_minus_global"] = (
        scores["ZG_intermediate_core_promoter_hypomethylation"] - scores["all_strict_promoter_hypomethylation_background"]
    )
    scores["ZG_aldosterone_promoter_hypomethylation_minus_global"] = (
        scores["ZG_aldosterone_promoter_hypomethylation"] - scores["all_strict_promoter_hypomethylation_background"]
    )
    scores["priority_gene_promoter_hypomethylation_minus_global"] = (
        scores["priority_gene_promoter_hypomethylation"] - scores["all_strict_promoter_hypomethylation_background"]
    )
    scores["ZG_intermediate_core_promoter_hypomethylation_minus_housekeeping"] = (
        scores["ZG_intermediate_core_promoter_hypomethylation"] - scores["housekeeping_promoter_hypomethylation_control"]
    )
    scores = scores.reset_index().rename(columns={"index": "geo_accession"}).merge(meta, on="geo_accession", how="left")
    availability_df = pd.DataFrame(availability)
    return scores, availability_df


def gene_level_expression_delta(gene_expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, values in gene_expr.iterrows():
        df = pd.DataFrame({"geo_accession": values.index, "value": values.to_numpy(dtype=float)}).merge(meta[["geo_accession", "case", "tissue"]], on="geo_accession")
        wide = df.pivot_table(index="case", columns="tissue", values="value", aggfunc="mean")
        if not {"APA", "AAG"}.issubset(set(wide.columns)):
            continue
        wide = wide.dropna(subset=["APA", "AAG"])
        if wide.empty:
            continue
        delta = wide["APA"] - wide["AAG"]
        rows.append(
            {
                "gene_symbol": gene,
                "expression_n_pairs": int(delta.shape[0]),
                "expression_mean_delta_APA_minus_AAG": float(delta.mean()),
                "expression_median_delta_APA_minus_AAG": float(delta.median()),
                "expression_n_positive": int((delta > 0).sum()),
                "expression_exact_two_sided_p": exact_sign_flip(delta.to_numpy())[0],
                "expression_wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gene_level_methylation_delta(gene_beta: pd.DataFrame, meta: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coverage_map = coverage.set_index("gene_symbol").to_dict("index") if not coverage.empty else {}
    for gene, values in gene_beta.iterrows():
        df = pd.DataFrame({"geo_accession": values.index, "beta": values.to_numpy(dtype=float)}).merge(meta[["geo_accession", "case", "tissue"]], on="geo_accession")
        wide = df.pivot_table(index="case", columns="tissue", values="beta", aggfunc="mean")
        if not {"APA", "AAG"}.issubset(set(wide.columns)):
            continue
        wide = wide.dropna(subset=["APA", "AAG"])
        if wide.empty:
            continue
        delta = wide["APA"] - wide["AAG"]
        cov = coverage_map.get(gene, {})
        rows.append(
            {
                "gene_symbol": gene,
                "methylation_n_pairs": int(delta.shape[0]),
                "methylation_mean_delta_APA_minus_AAG": float(delta.mean()),
                "methylation_median_delta_APA_minus_AAG": float(delta.median()),
                "methylation_n_hypomethylated": int((delta < 0).sum()),
                "methylation_exact_two_sided_p": exact_sign_flip(delta.to_numpy())[0],
                "methylation_wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
                "n_promoter_probes": int(cov.get("n_probes", 0)) if cov else 0,
                "promoter_groups": cov.get("promoter_groups", ""),
            }
        )
    return pd.DataFrame(rows)


def build_gene_concordance(expr_delta: pd.DataFrame, meth_delta: pd.DataFrame) -> pd.DataFrame:
    out = expr_delta.merge(meth_delta, on="gene_symbol", how="inner")
    sig_lookup = []
    for sig, genes in SIGNATURES.items():
        for gene in genes:
            sig_lookup.append({"gene_symbol": gene.upper(), "signature": sig})
    for gene in NICHE_GENES:
        sig_lookup.append({"gene_symbol": gene.upper(), "signature": "niche_candidate"})
    sig_df = pd.DataFrame(sig_lookup).drop_duplicates()
    sig_agg = sig_df.groupby("gene_symbol")["signature"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()
    out = out.merge(sig_agg, on="gene_symbol", how="left")
    out["priority_gene"] = out["gene_symbol"].isin(PRIORITY_GENES)
    out["expression_up_and_promoter_hypomethylated"] = (out["expression_mean_delta_APA_minus_AAG"] > 0) & (
        out["methylation_mean_delta_APA_minus_AAG"] < 0
    )
    out["expression_down_and_promoter_hypermethylated"] = (out["expression_mean_delta_APA_minus_AAG"] < 0) & (
        out["methylation_mean_delta_APA_minus_AAG"] > 0
    )
    out["concordant_inverse_direction"] = out["expression_up_and_promoter_hypomethylated"] | out["expression_down_and_promoter_hypermethylated"]
    out["negative_methylation_delta"] = -out["methylation_mean_delta_APA_minus_AAG"]
    return out


def summarize_concordance(conc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    background = conc.dropna(subset=["concordant_inverse_direction"])
    groups = {
        "all_gene_with_expression_and_promoter_methylation": conc,
        "priority_zg_intermediate_niche_genes": conc[conc["priority_gene"]],
        "housekeeping_control_genes": conc[conc["signature"].astype(str).str.contains("housekeeping_control", na=False)],
        "ZG_aldosterone_program_genes": conc[conc["signature"].astype(str).str.contains("ZG_aldosterone_program", na=False)],
        "intermediate_steroidogenic_program_genes": conc[conc["signature"].astype(str).str.contains("intermediate_steroidogenic_program", na=False)],
    }
    for name, df in groups.items():
        df = df.dropna(subset=["expression_mean_delta_APA_minus_AAG", "methylation_mean_delta_APA_minus_AAG"])
        if df.empty:
            continue
        rho, p_value = (np.nan, np.nan)
        if df.shape[0] >= 3:
            rho, p_value = stats.spearmanr(df["expression_mean_delta_APA_minus_AAG"], df["negative_methylation_delta"], nan_policy="omit")
        fisher_p = np.nan
        odds_ratio = np.nan
        if name != "all_gene_with_expression_and_promoter_methylation" and not background.empty:
            in_group = background["gene_symbol"].isin(df["gene_symbol"])
            a = int(background.loc[in_group, "concordant_inverse_direction"].sum())
            b = int(in_group.sum() - a)
            c = int(background.loc[~in_group, "concordant_inverse_direction"].sum())
            d = int((~in_group).sum() - c)
            if min(a, b, c, d) >= 0 and (a + b) > 0 and (c + d) > 0:
                odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "gene_set": name,
                "n_genes": int(df.shape[0]),
                "n_expression_up": int((df["expression_mean_delta_APA_minus_AAG"] > 0).sum()),
                "n_promoter_hypomethylated": int((df["methylation_mean_delta_APA_minus_AAG"] < 0).sum()),
                "n_inverse_concordant": int(df["concordant_inverse_direction"].sum()),
                "fraction_inverse_concordant": float(df["concordant_inverse_direction"].mean()),
                "mean_expression_delta": float(df["expression_mean_delta_APA_minus_AAG"].mean()),
                "mean_methylation_delta": float(df["methylation_mean_delta_APA_minus_AAG"].mean()),
                "spearman_expression_vs_negative_methylation_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p_value) if np.isfinite(p_value) else np.nan,
                "inverse_concordance_enrichment_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan,
                "inverse_concordance_enrichment_fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_paired_lines(ax: plt.Axes, scores: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    sub = scores[["case", "tissue", metric]].dropna()
    order = ["AAG", "APA"]
    sns.stripplot(data=sub, x="tissue", y=metric, order=order, ax=ax, color="#222222", size=5, jitter=0.04)
    for _, g in sub.groupby("case"):
        wide = g.set_index("tissue")[metric]
        if {"AAG", "APA"}.issubset(set(wide.index)):
            ax.plot([0, 1], [wide["AAG"], wide["APA"]], color="#5f6b7a", lw=1.1, alpha=0.75)
    means = sub.groupby("tissue")[metric].mean().reindex(order)
    ax.plot([0, 1], means.to_numpy(), color="#d14f3f", lw=2.2, marker="o")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def make_figure(expr_scores: pd.DataFrame, meth_scores: pd.DataFrame, conc: pd.DataFrame, expr_tests: pd.DataFrame, meth_tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))

    plot_paired_lines(
        axes[0, 0],
        expr_scores,
        "ZG_intermediate_vs_ZF_axis_score",
        "A. GSE60042 expression axis transfer",
        "ZG/intermediate-vs-ZF axis score",
    )
    plot_paired_lines(
        axes[0, 1],
        meth_scores,
        "ZG_intermediate_core_promoter_hypomethylation",
        "B. GSE60043 promoter hypomethylation",
        "Core promoter hypomethylation score",
    )

    show_metrics = [
        "ZG_intermediate_vs_ZF_axis_score",
        "ZG_aldosterone_program",
        "ZF_cortisol_program",
        "intermediate_steroidogenic_program",
        "aldosterone_specificity_score",
        "housekeeping_control",
    ]
    tmp = expr_tests[expr_tests["metric"].isin(show_metrics)].copy()
    tmp["metric"] = pd.Categorical(tmp["metric"], categories=show_metrics[::-1], ordered=True)
    colors = ["#3178b8" if x > 0 else "#b8554f" for x in tmp["mean_delta"]]
    axes[1, 0].barh(tmp["metric"].astype(str), tmp["mean_delta"], color=colors, alpha=0.88)
    axes[1, 0].axvline(0, color="black", lw=0.8)
    axes[1, 0].set_title("C. Paired expression deltas")
    axes[1, 0].set_xlabel("APA - AAG mean delta")
    axes[1, 0].set_ylabel("")

    plot_df = conc[conc["priority_gene"]].copy()
    if plot_df.empty:
        plot_df = conc.copy().head(40)
    axes[1, 1].axhline(0, color="#777777", lw=0.8)
    axes[1, 1].axvline(0, color="#777777", lw=0.8)
    sns.scatterplot(
        data=plot_df,
        x="expression_mean_delta_APA_minus_AAG",
        y="negative_methylation_delta",
        hue="concordant_inverse_direction",
        style="priority_gene",
        palette={True: "#287c62", False: "#8d8d8d"},
        ax=axes[1, 1],
        s=42,
        edgecolor="white",
        linewidth=0.5,
    )
    label_genes = ["CYP11B2", "MC2R", "HSD3B2", "STAR", "NR4A2", "LGR5", "CXCL12", "RSPO3", "WNT4"]
    for _, row in plot_df[plot_df["gene_symbol"].isin(label_genes)].iterrows():
        axes[1, 1].text(row["expression_mean_delta_APA_minus_AAG"], row["negative_methylation_delta"], row["gene_symbol"], fontsize=7)
    axes[1, 1].set_title("D. Expression-methylation concordance")
    axes[1, 1].set_xlabel("Expression delta (APA - AAG)")
    axes[1, 1].set_ylabel("- promoter methylation delta")
    axes[1, 1].legend(loc="best", fontsize=7, frameon=True)

    fig.tight_layout()
    fig.savefig(FIG / "Fig6_external_transcriptome_methylome_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Fig6_external_transcriptome_methylome_validation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def write_report(
    expr_meta: pd.DataFrame,
    meth_meta: pd.DataFrame,
    expr_avail: pd.DataFrame,
    meth_avail: pd.DataFrame,
    expr_tests: pd.DataFrame,
    meth_tests: pd.DataFrame,
    conc_summary: pd.DataFrame,
    conc: pd.DataFrame,
) -> None:
    primary = expr_tests[expr_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score"].iloc[0]
    meth_primary = meth_tests[meth_tests["metric"] == "ZG_intermediate_core_promoter_hypomethylation"].iloc[0]
    meth_background = meth_tests[meth_tests["metric"] == "all_strict_promoter_hypomethylation_background"].iloc[0]
    meth_global_adjusted = meth_tests[meth_tests["metric"] == "ZG_intermediate_core_promoter_hypomethylation_minus_global"].iloc[0]
    meth_housekeeping_adjusted = meth_tests[meth_tests["metric"] == "ZG_intermediate_core_promoter_hypomethylation_minus_housekeeping"].iloc[0]
    priority = conc_summary[conc_summary["gene_set"] == "priority_zg_intermediate_niche_genes"]
    priority_row = priority.iloc[0] if not priority.empty else None
    zg_row = conc_summary[conc_summary["gene_set"] == "ZG_aldosterone_program_genes"]
    zg_row = zg_row.iloc[0] if not zg_row.empty else None
    cyp = conc[conc["gene_symbol"] == "CYP11B2"]

    lines = [
        "# External transcriptome-methylome validation report 2026-06-02",
        "",
        "## Purpose",
        "",
        "Upgrade the APA zonation-axis manuscript with an independent paired transcriptome-methylome layer from the GSE60044 superseries.",
        "",
        "Evidence roles:",
        "",
        "- `GSE60042`: paired APA/AAG bulk microarray expression transfer.",
        "- `GSE60043`: paired APA/AAG promoter methylation concordance.",
        "- `GSE60044`: superseries connecting both layers from the same seven cases.",
        "",
        "## Dataset audit",
        "",
        f"- GSE60042 expression samples: {expr_meta.shape[0]} samples, {expr_meta['case'].nunique()} cases, tissue counts {expr_meta['tissue'].value_counts().to_dict()}.",
        f"- GSE60043 methylation samples: {meth_meta.shape[0]} samples, {meth_meta['case'].nunique()} cases, tissue counts {meth_meta['tissue'].value_counts().to_dict()}.",
        "",
        "## Expression transfer: GSE60042",
        "",
        f"- Primary ZG/intermediate-vs-ZF axis: mean APA-AAG delta = {safe_fmt(primary['mean_delta'])}; median delta = {safe_fmt(primary['median_delta'])}; {int(primary['n_positive'])}/{int(primary['n_pairs'])} pairs positive; exact sign-flip two-sided p = {safe_fmt(primary['exact_sign_flip_two_sided_p'])}; Wilcoxon p = {safe_fmt(primary['wilcoxon_p'])}.",
        "",
        "Signature availability:",
        "",
    ]
    for _, row in expr_avail.iterrows():
        if row["signature"] in KEY_METRICS or row["signature"] in SIGNATURES:
            lines.append(f"- {row['signature']}: {int(row['n_available'])}/{int(row['n_requested'])} genes available.")
    lines.extend(
        [
            "",
            "## Promoter methylation concordance: GSE60043",
            "",
            f"- ZG/intermediate core promoter hypomethylation score: mean APA-AAG delta = {safe_fmt(meth_primary['mean_delta'])}; median delta = {safe_fmt(meth_primary['median_delta'])}; {int(meth_primary['n_positive'])}/{int(meth_primary['n_pairs'])} pairs higher hypomethylation in APA; exact sign-flip two-sided p = {safe_fmt(meth_primary['exact_sign_flip_two_sided_p'])}; Wilcoxon p = {safe_fmt(meth_primary['wilcoxon_p'])}.",
            f"- Global strict-promoter hypomethylation background is also present: mean APA-AAG delta = {safe_fmt(meth_background['mean_delta'])}; {int(meth_background['n_positive'])}/{int(meth_background['n_pairs'])} pairs positive; exact p = {safe_fmt(meth_background['exact_sign_flip_two_sided_p'])}.",
            f"- Core promoter hypomethylation after subtracting the global promoter background is directionally positive but not a strong paired result: mean delta = {safe_fmt(meth_global_adjusted['mean_delta'])}; {int(meth_global_adjusted['n_positive'])}/{int(meth_global_adjusted['n_pairs'])} pairs positive; exact p = {safe_fmt(meth_global_adjusted['exact_sign_flip_two_sided_p'])}.",
            f"- Core promoter hypomethylation after subtracting housekeeping promoter hypomethylation remains positive in {int(meth_housekeeping_adjusted['n_positive'])}/{int(meth_housekeeping_adjusted['n_pairs'])} pairs; exact p = {safe_fmt(meth_housekeeping_adjusted['exact_sign_flip_two_sided_p'])}.",
            "",
            "Methylation signature availability:",
            "",
        ]
    )
    for _, row in meth_avail.iterrows():
        lines.append(f"- {row['signature']}: {int(row['n_available'])}/{int(row['n_requested'])} genes with strict TSS promoter probes.")
    lines.extend(["", "## Gene-level expression-methylation concordance", ""])
    if priority_row is not None:
        lines.append(
            f"- Priority ZG/intermediate/niche genes: {int(priority_row['n_inverse_concordant'])}/{int(priority_row['n_genes'])} genes show inverse expression-methylation direction; mean expression delta = {safe_fmt(priority_row['mean_expression_delta'])}; mean methylation delta = {safe_fmt(priority_row['mean_methylation_delta'])}; Fisher enrichment vs all mapped genes p = {safe_fmt(priority_row['inverse_concordance_enrichment_fisher_p'])}."
        )
    if zg_row is not None:
        lines.append(
            f"- ZG aldosterone genes: {int(zg_row['n_inverse_concordant'])}/{int(zg_row['n_genes'])} inverse-concordant; Fisher enrichment p = {safe_fmt(zg_row['inverse_concordance_enrichment_fisher_p'])}."
        )
    if not cyp.empty:
        row = cyp.iloc[0]
        lines.append(
            f"- CYP11B2: expression delta = {safe_fmt(row['expression_mean_delta_APA_minus_AAG'])}; promoter methylation delta = {safe_fmt(row['methylation_mean_delta_APA_minus_AAG'])}; n promoter probes = {int(row['n_promoter_probes'])}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This layer strongly supports independent paired expression transfer of the spatially defined APA zonation-axis program. The methylation layer supports directionally concordant promoter hypomethylation and inverse expression-methylation enrichment, but it occurs against a global APA promoter hypomethylation background and should not be written as gene-specific causal epigenetic proof.",
            "",
            "Recommended wording:",
            "",
            "> The spatially defined APA ZG/intermediate-vs-ZF steroidogenic axis was independently transferred to a paired bulk transcriptome superseries and showed directionally concordant promoter methylation support, including inverse expression-methylation enrichment among ZG/aldosterone genes, while recognizing a broader APA promoter hypomethylation background.",
            "",
            "Avoid:",
            "",
            "- DNA methylation causes APA zonation-axis rewiring.",
            "- Promoter hypomethylation proves CYP11B2 regulation.",
            "- Bulk expression/methylation validates spatial cell abundance.",
            "",
            "## Outputs",
            "",
            "- `results/external_transcriptome_methylome_20260602/GSE60042_expression_axis_scores.csv`",
            "- `results/external_transcriptome_methylome_20260602/GSE60042_expression_paired_tests.csv`",
            "- `results/external_transcriptome_methylome_20260602/GSE60043_strict_promoter_methylation_scores.csv`",
            "- `results/external_transcriptome_methylome_20260602/GSE60043_promoter_methylation_paired_tests.csv`",
            "- `results/external_transcriptome_methylome_20260602/GSE60042_GSE60043_gene_expression_methylation_concordance.csv`",
            "- `results/external_transcriptome_methylome_20260602/figures/Fig6_external_transcriptome_methylome_validation.png/.pdf`",
        ]
    )
    (PROJECT / "external_transcriptome_methylome_validation_report_20260602.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_input_files()

    expr_meta, expr_probe = parse_series_matrix(RAW / "GSE60042" / "GSE60042_series_matrix.txt.gz")
    expr_meta = add_case_tissue(expr_meta, "GSE60042")
    expr_map = parse_agilent_gene_map(RAW / "GSE60042" / "GPL14550_family.soft.gz")
    expr_gene = collapse_probe_to_gene(expr_probe, expr_map)
    expr_scores, expr_avail = score_expression(expr_gene, expr_meta)
    expr_tests, expr_delta = paired_metric_tests(expr_scores, KEY_METRICS, "GSE60042")

    expr_meta.to_csv(PROCESSED / "GSE60042_sample_metadata.csv", index=False, encoding="utf-8-sig")
    expr_gene.reset_index().to_csv(PROCESSED / "GSE60042_gene_expression_mean_collapsed.csv", index=False, encoding="utf-8-sig")
    expr_scores.to_csv(OUT / "GSE60042_expression_axis_scores.csv", index=False, encoding="utf-8-sig")
    expr_avail.to_csv(OUT / "GSE60042_expression_signature_gene_availability.csv", index=False, encoding="utf-8-sig")
    expr_tests.to_csv(OUT / "GSE60042_expression_paired_tests.csv", index=False, encoding="utf-8-sig")
    expr_delta.to_csv(OUT / "GSE60042_expression_paired_delta.csv", index=False, encoding="utf-8-sig")

    meth_meta, beta_probe = parse_series_matrix(RAW / "GSE60043" / "GSE60043_series_matrix.txt.gz")
    meth_meta = add_case_tissue(meth_meta, "GSE60043")
    probe_map = parse_methylation_probe_map(RAW / "GSE60043" / "GPL13534_family.soft.gz")
    meth_gene_strict, meth_cov_strict = aggregate_methylation_by_gene(beta_probe, probe_map, promoter_mode="strict")
    meth_scores, meth_avail = methylation_hypomethylation_scores(meth_gene_strict, meth_meta, meth_cov_strict)
    meth_metrics = [
        "all_strict_promoter_hypomethylation_background",
        "ZG_intermediate_core_promoter_hypomethylation",
        "ZG_intermediate_core_promoter_hypomethylation_minus_global",
        "ZG_intermediate_core_promoter_hypomethylation_minus_housekeeping",
        "ZG_aldosterone_promoter_hypomethylation",
        "ZG_aldosterone_promoter_hypomethylation_minus_global",
        "intermediate_steroidogenic_promoter_hypomethylation",
        "ZF_cortisol_promoter_hypomethylation",
        "priority_gene_promoter_hypomethylation",
        "priority_gene_promoter_hypomethylation_minus_global",
        "housekeeping_promoter_hypomethylation_control",
    ]
    meth_tests, meth_delta = paired_metric_tests(meth_scores, meth_metrics, "GSE60043")

    meth_meta.to_csv(PROCESSED / "GSE60043_sample_metadata.csv", index=False, encoding="utf-8-sig")
    meth_gene_strict.reset_index().to_csv(PROCESSED / "GSE60043_gene_strict_promoter_methylation_mean_beta.csv", index=False, encoding="utf-8-sig")
    meth_cov_strict.to_csv(OUT / "GSE60043_strict_promoter_probe_gene_coverage.csv", index=False, encoding="utf-8-sig")
    meth_scores.to_csv(OUT / "GSE60043_strict_promoter_methylation_scores.csv", index=False, encoding="utf-8-sig")
    meth_avail.to_csv(OUT / "GSE60043_methylation_signature_gene_availability.csv", index=False, encoding="utf-8-sig")
    meth_tests.to_csv(OUT / "GSE60043_promoter_methylation_paired_tests.csv", index=False, encoding="utf-8-sig")
    meth_delta.to_csv(OUT / "GSE60043_promoter_methylation_paired_delta.csv", index=False, encoding="utf-8-sig")

    expr_gene_delta = gene_level_expression_delta(expr_gene, expr_meta)
    meth_gene_delta = gene_level_methylation_delta(meth_gene_strict, meth_meta, meth_cov_strict)
    conc = build_gene_concordance(expr_gene_delta, meth_gene_delta)
    conc_summary = summarize_concordance(conc)
    expr_gene_delta.to_csv(OUT / "GSE60042_gene_expression_paired_delta.csv", index=False, encoding="utf-8-sig")
    meth_gene_delta.to_csv(OUT / "GSE60043_gene_promoter_methylation_paired_delta.csv", index=False, encoding="utf-8-sig")
    conc.to_csv(OUT / "GSE60042_GSE60043_gene_expression_methylation_concordance.csv", index=False, encoding="utf-8-sig")
    conc_summary.to_csv(OUT / "GSE60042_GSE60043_concordance_summary.csv", index=False, encoding="utf-8-sig")

    make_figure(expr_scores, meth_scores, conc, expr_tests, meth_tests)
    write_report(expr_meta, meth_meta, expr_avail, meth_avail, expr_tests, meth_tests, conc_summary, conc)

    print("GSE60042 expression primary axis:")
    print(expr_tests[expr_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score"].to_string(index=False))
    print("GSE60043 methylation core promoter hypomethylation:")
    print(meth_tests[meth_tests["metric"] == "ZG_intermediate_core_promoter_hypomethylation"].to_string(index=False))
    print("External transcriptome-methylome validation complete.")


if __name__ == "__main__":
    main()
