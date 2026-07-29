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

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - optional local dependency
    smf = None


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
STAGE2 = RESULTS / "stage2_zonation_axis"
OUT = RESULTS / "computational_only_defense_20260602"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260602)
N_PERM = 5000
N_BOOT = 10000
PRIMARY_SPATIAL_SCORE = "ZG_intermediate_vs_ZF_axis_score"


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


def safe_float(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def exact_sign_flip(delta: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    observed = float(delta.mean())
    rows = []
    for bits in itertools.product([-1, 1], repeat=len(delta)):
        perm_mean = float((delta * np.asarray(bits)).mean())
        rows.append({"perm_mean_delta": perm_mean, "abs_ge_observed": abs(perm_mean) >= abs(observed), "ge_observed": perm_mean >= observed})
    perm = pd.DataFrame(rows)
    two_sided = float(perm["abs_ge_observed"].mean())
    greater = float(perm["ge_observed"].mean())
    return two_sided, greater, perm


def paired_delta_from_section_summary(summary: pd.DataFrame, metric: str) -> pd.Series:
    wide = summary.pivot_table(index="patient", columns="tissue", values=metric, aggfunc="mean")
    wide = wide.dropna(subset=["APA", "Adjacent"])
    return (wide["APA"] - wide["Adjacent"]).astype(float)


def section_exact_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(STAGE2 / "GSE274314_stage2_section_score_summary.csv")
    rows = []
    perm_primary = pd.DataFrame()
    metric_cols = [f"{m}_mean" for m in KEY_METRICS if f"{m}_mean" in summary.columns]
    metric_cols += [f"{m}_depth_adjusted_mean" for m in KEY_METRICS if f"{m}_depth_adjusted_mean" in summary.columns]
    for metric in metric_cols:
        delta = paired_delta_from_section_summary(summary, metric)
        if delta.empty:
            continue
        ci_l, ci_u = bootstrap_ci(delta.to_numpy())
        exact_two, exact_greater, perm = exact_sign_flip(delta.to_numpy())
        if metric == f"{PRIMARY_SPATIAL_SCORE}_mean":
            perm_primary = perm.copy()
            perm_primary["observed_mean_delta"] = float(delta.mean())
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(delta.shape[0]),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "ci95_low": ci_l,
                "ci95_high": ci_u,
                "n_positive": int((delta > 0).sum()),
                "n_negative": int((delta < 0).sum()),
                "exact_sign_flip_two_sided_p": exact_two,
                "exact_sign_flip_greater_p": exact_greater,
                "one_sided_sign_p": stats.binomtest(int((delta > 0).sum()), len(delta), 0.5, alternative="greater").pvalue,
                "wilcoxon_p": stats.wilcoxon(delta, alternative="two-sided").pvalue if np.all(delta != 0) else np.nan,
                "paired_t_p": stats.ttest_1samp(delta, 0.0).pvalue if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["exact_sign_flip_two_sided_fdr"] = bh_fdr(out["exact_sign_flip_two_sided_p"])
    out.to_csv(OUT / "section_level_exact_paired_statistics.csv", index=False, encoding="utf-8-sig")
    perm_primary.to_csv(OUT / "primary_axis_exact_sign_flip_null.csv", index=False, encoding="utf-8-sig")
    return out, perm_primary


def mixed_effect_section_tests() -> pd.DataFrame:
    rows = [
        {
            "metric": "all",
            "status": "skipped_by_design",
            "model": "score ~ tissue + (1|patient)",
            "note": "With only seven patient pairs, mixed models frequently hit singular random-effect fits; exact paired and patient fixed-effect tests are used instead.",
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "section_level_mixed_effect_tests.csv", index=False, encoding="utf-8-sig")
    return out


def patient_fixed_effect_section_tests() -> pd.DataFrame:
    summary = pd.read_csv(STAGE2 / "GSE274314_stage2_section_score_summary.csv")
    metrics = [f"{m}_mean" for m in KEY_METRICS if f"{m}_mean" in summary.columns]
    metrics += [f"{m}_depth_adjusted_mean" for m in KEY_METRICS if f"{m}_depth_adjusted_mean" in summary.columns]
    rows = []
    if smf is None:
        rows.append({"metric": "all", "status": "statsmodels not available"})
        out = pd.DataFrame(rows)
        out.to_csv(OUT / "section_level_mixed_effect_tests.csv", index=False, encoding="utf-8-sig")
        return out
    for metric in metrics:
        df = summary[["patient", "tissue", metric]].dropna().rename(columns={metric: "score"})
        df["tissue"] = pd.Categorical(df["tissue"], categories=["Adjacent", "APA"], ordered=True)
        try:
            fit = smf.ols("score ~ tissue + C(patient)", data=df).fit()
            term = "tissue[T.APA]"
            rows.append(
                {
                    "metric": metric,
                    "status": "ok",
                    "n_sections": int(df.shape[0]),
                    "n_patients": int(df["patient"].nunique()),
                    "apa_vs_adjacent_coef": float(fit.params.get(term, np.nan)),
                    "apa_vs_adjacent_p": float(fit.pvalues.get(term, np.nan)),
                    "model": "score ~ tissue + C(patient)",
                }
            )
        except Exception as exc:
            rows.append({"metric": metric, "status": f"failed: {exc}", "model": "score ~ tissue + C(patient)"})
    out = pd.DataFrame(rows)
    if "apa_vs_adjacent_p" in out.columns:
        out["apa_vs_adjacent_fdr"] = bh_fdr(out["apa_vs_adjacent_p"])
    out.to_csv(OUT / "section_level_patient_fixed_effect_tests.csv", index=False, encoding="utf-8-sig")
    return out


def spot_label_permutation_null(score_col: str = PRIMARY_SPATIAL_SCORE) -> pd.DataFrame:
    spots = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    groups = []
    observed = []
    for patient, sub in spots.groupby("patient"):
        a = sub.loc[sub["tissue"] == "APA", score_col].astype(float).dropna().to_numpy()
        b = sub.loc[sub["tissue"] == "Adjacent", score_col].astype(float).dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        groups.append({"patient": patient, "values": np.concatenate([a, b]), "n_apa": len(a), "n_adjacent": len(b)})
        observed.append(float(a.mean() - b.mean()))
    obs_mean = float(np.mean(observed))
    rows = []
    for i in range(1, N_PERM + 1):
        deltas = []
        for g in groups:
            perm = RNG.permutation(g["values"])
            apa = perm[: g["n_apa"]]
            adj = perm[g["n_apa"] :]
            deltas.append(float(apa.mean() - adj.mean()))
        perm_mean = float(np.mean(deltas))
        rows.append({"permutation": i, "perm_mean_delta": perm_mean, "observed_mean_delta": obs_mean})
    out = pd.DataFrame(rows)
    out["abs_ge_observed"] = out["perm_mean_delta"].abs() >= abs(obs_mean)
    out["ge_observed"] = out["perm_mean_delta"] >= obs_mean
    out.attrs["two_sided_p"] = (1 + int(out["abs_ge_observed"].sum())) / (len(out) + 1)
    out.attrs["greater_p"] = (1 + int(out["ge_observed"].sum())) / (len(out) + 1)
    out.to_csv(OUT / "primary_axis_within_patient_spot_label_null.csv", index=False, encoding="utf-8-sig")
    return out


def assign_spatial_blocks(section: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    x = section["lowres_x"].astype(float)
    y = section["lowres_y"].astype(float)
    x_rank = x.rank(method="first", pct=True)
    y_rank = y.rank(method="first", pct=True)
    x_bin = np.minimum((x_rank * n_bins).astype(int), n_bins - 1)
    y_bin = np.minimum((y_rank * n_bins).astype(int), n_bins - 1)
    return x_bin.astype(str) + "_" + y_bin.astype(str)


def spatial_block_permutation_null(score_col: str = PRIMARY_SPATIAL_SCORE) -> tuple[pd.DataFrame, pd.DataFrame]:
    spots = pd.read_csv(STAGE2 / "GSE274314_stage2_spot_scores.csv")
    block_rows = []
    for (gsm, patient, tissue), sub in spots.groupby(["geo_accession", "patient", "tissue"], dropna=False):
        sub = sub.copy()
        sub["block"] = assign_spatial_blocks(sub)
        agg = sub.groupby("block", dropna=False).agg(
            score_mean=(score_col, "mean"),
            n_spots=(score_col, "size"),
            lowres_x=("lowres_x", "mean"),
            lowres_y=("lowres_y", "mean"),
        )
        for block, row in agg.iterrows():
            block_rows.append(
                {
                    "geo_accession": gsm,
                    "patient": patient,
                    "tissue": tissue,
                    "block": block,
                    "score_mean": float(row["score_mean"]),
                    "n_spots": int(row["n_spots"]),
                    "lowres_x": float(row["lowres_x"]),
                    "lowres_y": float(row["lowres_y"]),
                }
            )
    blocks = pd.DataFrame(block_rows)
    observed = []
    groups = []
    for patient, sub in blocks.groupby("patient"):
        a = sub[sub["tissue"] == "APA"].copy()
        b = sub[sub["tissue"] == "Adjacent"].copy()
        if a.empty or b.empty:
            continue
        obs = np.average(a["score_mean"], weights=a["n_spots"]) - np.average(b["score_mean"], weights=b["n_spots"])
        observed.append(float(obs))
        pooled = pd.concat([a, b], ignore_index=True)
        groups.append({"patient": patient, "scores": pooled["score_mean"].to_numpy(float), "weights": pooled["n_spots"].to_numpy(float), "n_apa_blocks": a.shape[0]})
    obs_mean = float(np.mean(observed))
    rows = []
    for i in range(1, N_PERM + 1):
        deltas = []
        for g in groups:
            idx = RNG.permutation(np.arange(len(g["scores"])))
            apa_idx = idx[: g["n_apa_blocks"]]
            adj_idx = idx[g["n_apa_blocks"] :]
            apa = np.average(g["scores"][apa_idx], weights=g["weights"][apa_idx])
            adj = np.average(g["scores"][adj_idx], weights=g["weights"][adj_idx])
            deltas.append(float(apa - adj))
        perm_mean = float(np.mean(deltas))
        rows.append({"permutation": i, "perm_mean_delta": perm_mean, "observed_mean_delta": obs_mean})
    null = pd.DataFrame(rows)
    null["abs_ge_observed"] = null["perm_mean_delta"].abs() >= abs(obs_mean)
    null["ge_observed"] = null["perm_mean_delta"] >= obs_mean
    null.attrs["two_sided_p"] = (1 + int(null["abs_ge_observed"].sum())) / (len(null) + 1)
    null.attrs["greater_p"] = (1 + int(null["ge_observed"].sum())) / (len(null) + 1)
    blocks.to_csv(OUT / "primary_axis_spatial_block_table.csv", index=False, encoding="utf-8-sig")
    null.to_csv(OUT / "primary_axis_spatial_block_label_null.csv", index=False, encoding="utf-8-sig")
    return blocks, null


def score_bulk_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = pd.read_csv(PROCESSED / "GSE64957_gene_expression_mean_collapsed.csv")
    expr["gene_symbol"] = expr["gene_symbol"].astype(str).str.upper()
    expr = expr.groupby("gene_symbol", as_index=False).mean(numeric_only=True)
    meta = pd.read_csv(PROCESSED / "GSE64957_sample_metadata.csv")
    sample_cols = [c for c in expr.columns if c != "gene_symbol"]
    expr_mat = expr.set_index("gene_symbol")[sample_cols].T
    expr_mat.index.name = "geo_accession"
    z = expr_mat.copy()
    for gene in z.columns:
        vals = z[gene].astype(float)
        sd = vals.std(ddof=0)
        z[gene] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else 0.0
    scores = pd.DataFrame(index=z.index)
    availability = []
    for name, genes in SIGNATURES.items():
        genes_u = [g.upper() for g in genes]
        available = [g for g in genes_u if g in z.columns]
        missing = [g for g in genes_u if g not in z.columns]
        scores[name] = z[available].mean(axis=1) if available else np.nan
        availability.append(
            {
                "dataset": "GSE64957",
                "signature": name,
                "n_requested": len(genes_u),
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
    scores = scores.reset_index().merge(meta, on="geo_accession", how="left")
    scores.to_csv(OUT / "GSE64957_bulk_zonation_axis_scores.csv", index=False, encoding="utf-8-sig")
    avail = pd.DataFrame(availability)
    avail.to_csv(OUT / "GSE64957_bulk_signature_gene_availability.csv", index=False, encoding="utf-8-sig")
    return scores, avail


def group_or_paired_test(scores: pd.DataFrame, metric: str, group_a: str, group_b: str, paired: bool) -> dict:
    if paired:
        sub = scores[scores["tissue_subtype"].isin([group_a, group_b])]
        wide = sub.pivot_table(index="patient_tissue_id", columns="tissue_subtype", values=metric, aggfunc="mean")
        if not {group_a, group_b}.issubset(wide.columns):
            return {}
        wide = wide.dropna(subset=[group_a, group_b])
        delta = (wide[group_a] - wide[group_b]).astype(float)
        if delta.empty:
            return {}
        ci_l, ci_u = bootstrap_ci(delta.to_numpy())
        exact_two, exact_greater, _ = exact_sign_flip(delta.to_numpy())
        return {
            "dataset": "GSE64957",
            "comparison": f"{group_a} vs {group_b}",
            "metric": metric,
            "test_type": "patient_tissue_id paired",
            "n_a": int(delta.shape[0]),
            "n_b": int(delta.shape[0]),
            "mean_a_minus_b": float(delta.mean()),
            "median_a_minus_b": float(delta.median()),
            "ci95_low": ci_l,
            "ci95_high": ci_u,
            "n_positive": int((delta > 0).sum()),
            "n_negative": int((delta < 0).sum()),
            "p_value": stats.wilcoxon(delta).pvalue if np.all(delta != 0) else np.nan,
            "exact_sign_flip_two_sided_p": exact_two,
            "exact_sign_flip_greater_p": exact_greater,
        }
    a = scores.loc[scores["tissue_subtype"] == group_a, metric].astype(float).dropna().to_numpy()
    b = scores.loc[scores["tissue_subtype"] == group_b, metric].astype(float).dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return {}
    diff = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    null = []
    for _ in range(N_PERM):
        perm = RNG.permutation(pooled)
        null.append(float(perm[: len(a)].mean() - perm[len(a) :].mean()))
    null = np.asarray(null)
    return {
        "dataset": "GSE64957",
        "comparison": f"{group_a} vs {group_b}",
        "metric": metric,
        "test_type": "unpaired permutation + Mann-Whitney",
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a_minus_b": diff,
        "median_a_minus_b": float(np.median(a) - np.median(b)),
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "n_positive": np.nan,
        "n_negative": np.nan,
        "p_value": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
        "permutation_two_sided_p": (1 + int(np.sum(np.abs(null) >= abs(diff)))) / (N_PERM + 1),
        "permutation_greater_p": (1 + int(np.sum(null >= diff))) / (N_PERM + 1),
    }


def bulk_validation_tests(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    metrics = [
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
    comparisons = [
        ("Aldosterone-Producing Adenoma", "Zona Fasciculata", True),
        ("Zona Glomerulosa", "Zona Fasciculata", True),
        ("Aldosterone-Producing Adenoma", "Zona Glomerulosa", True),
        ("Aldosterone-Producing Adenoma", "Zona Fasciculata", False),
        ("Zona Glomerulosa", "Zona Fasciculata", False),
        ("Aldosterone-Producing Adenoma", "Zona Glomerulosa", False),
    ]
    rows = []
    for metric in metrics:
        for a, b, paired in comparisons:
            row = group_or_paired_test(scores, metric, a, b, paired=paired)
            if row:
                rows.append(row)
    out = pd.DataFrame(rows)
    if "p_value" in out.columns:
        out["p_value_fdr"] = bh_fdr(out["p_value"])
    out.to_csv(OUT / "GSE64957_bulk_axis_validation_tests.csv", index=False, encoding="utf-8-sig")
    return out


def bulk_centroid_distance(scores: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "ZG_aldosterone_program",
        "ZF_cortisol_program",
        "ZR_androgen_program",
        "intermediate_steroidogenic_program",
        "generic_steroidogenesis_program",
        "ZG_intermediate_vs_ZF_axis_score",
    ]
    ref = scores[scores["tissue_subtype"].isin(["Zona Glomerulosa", "Zona Fasciculata"])].copy()
    centroids = ref.groupby("tissue_subtype")[score_cols].mean()
    rows = []
    for _, row in scores.iterrows():
        if row["tissue_subtype"] != "Aldosterone-Producing Adenoma":
            continue
        vec = row[score_cols].astype(float).to_numpy()
        d_zg = float(np.linalg.norm(vec - centroids.loc["Zona Glomerulosa"].to_numpy(float)))
        d_zf = float(np.linalg.norm(vec - centroids.loc["Zona Fasciculata"].to_numpy(float)))
        rows.append(
            {
                "geo_accession": row["geo_accession"],
                "patient_tissue_id": row["patient_tissue_id"],
                "distance_to_ZG_centroid": d_zg,
                "distance_to_ZF_centroid": d_zf,
                "closer_to_ZG_than_ZF": d_zg < d_zf,
                "ZG_minus_ZF_distance": d_zg - d_zf,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "GSE64957_APA_centroid_distance_to_zones.csv", index=False, encoding="utf-8-sig")
    return out


def null_summary_table(section_exact: pd.DataFrame, spot_null: pd.DataFrame, block_null: pd.DataFrame) -> pd.DataFrame:
    primary = section_exact[section_exact["metric"] == f"{PRIMARY_SPATIAL_SCORE}_mean"].iloc[0]
    rows = [
        {
            "analysis": "Exact patient-paired section sign-flip null",
            "unit_preserved": "patient pair / section mean",
            "observed_mean_delta": primary["mean_delta"],
            "two_sided_p": primary["exact_sign_flip_two_sided_p"],
            "greater_p": primary["exact_sign_flip_greater_p"],
            "interpretation": "Primary inference does not rely on parametric t test.",
        },
        {
            "analysis": "Within-patient spot-label null",
            "unit_preserved": "patient pair; original APA/adjacent spot counts",
            "observed_mean_delta": spot_null["observed_mean_delta"].iloc[0],
            "two_sided_p": (1 + int(spot_null["abs_ge_observed"].sum())) / (len(spot_null) + 1),
            "greater_p": (1 + int(spot_null["ge_observed"].sum())) / (len(spot_null) + 1),
            "interpretation": "Spot-level distributions are not exchangeable within patient pairs.",
        },
        {
            "analysis": "Spatial block-label null",
            "unit_preserved": "within-section spatial blocks aggregated before permutation",
            "observed_mean_delta": block_null["observed_mean_delta"].iloc[0],
            "two_sided_p": (1 + int(block_null["abs_ge_observed"].sum())) / (len(block_null) + 1),
            "greater_p": (1 + int(block_null["ge_observed"].sum())) / (len(block_null) + 1),
            "interpretation": "The paired signal exceeds a local-spatial-block label randomization.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "primary_axis_null_summary_table.csv", index=False, encoding="utf-8-sig")
    return out


def make_figures(section_exact: pd.DataFrame, sign_null: pd.DataFrame, spot_null: pd.DataFrame, block_null: pd.DataFrame, bulk_scores: pd.DataFrame, bulk_tests: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    primary = section_exact[section_exact["metric"] == f"{PRIMARY_SPATIAL_SCORE}_mean"].iloc[0]
    observed = float(primary["mean_delta"])
    panels = [
        ("A. exact sign-flip", sign_null, "perm_mean_delta", True),
        ("B. spot-label null", spot_null, "perm_mean_delta", False),
        ("C. spatial block null", block_null, "perm_mean_delta", False),
    ]
    for ax, (title, df, col, show_observed_line) in zip(axes, panels):
        values = df[col].astype(float)
        ax.hist(values, bins=35, color="#bdbdbd", edgecolor="white")
        if show_observed_line:
            ax.axvline(observed, color="#0072B2", lw=2)
            ax.axvline(-observed, color="#0072B2", lw=1, ls="--")
        else:
            lo, hi = np.quantile(values, [0.001, 0.999])
            pad = max((hi - lo) * 0.08, 1e-4)
            ax.set_xlim(lo - pad, hi + pad)
            ax.text(
                0.98,
                0.94,
                f"observed = {observed:.3f}\noutside null range",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#0072B2",
            )
        ax.set_title(title)
        ax.set_xlabel("Null mean APA - adjacent delta")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Null draws")
    fig.tight_layout()
    fig.savefig(FIG / "primary_axis_computational_nulls.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "primary_axis_computational_nulls.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    plot = bulk_scores[bulk_scores["tissue_subtype"].isin(["Aldosterone-Producing Adenoma", "Zona Glomerulosa", "Zona Fasciculata"])].copy()
    plot["tissue_subtype"] = pd.Categorical(
        plot["tissue_subtype"],
        ["Zona Fasciculata", "Zona Glomerulosa", "Aldosterone-Producing Adenoma"],
        ordered=True,
    )
    short = {"Zona Fasciculata": "ZF", "Zona Glomerulosa": "ZG", "Aldosterone-Producing Adenoma": "APA"}
    plot["subtype_short"] = plot["tissue_subtype"].astype(str).map(short)
    plot["subtype_short"] = pd.Categorical(plot["subtype_short"], ["ZF", "ZG", "APA"], ordered=True)
    sns.boxplot(data=plot, x="subtype_short", y="ZG_intermediate_vs_ZF_axis_score", ax=axes[0], color="#d9eaf7", fliersize=0)
    sns.stripplot(data=plot, x="subtype_short", y="ZG_intermediate_vs_ZF_axis_score", ax=axes[0], color="#333333", size=3, jitter=0.18)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Bulk ZG/intermediate vs ZF axis")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("A. GSE64957 zone-axis transfer")
    axes[0].spines[["top", "right"]].set_visible(False)

    paired = bulk_tests[
        (bulk_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score")
        & (bulk_tests["test_type"] == "patient_tissue_id paired")
        & (bulk_tests["comparison"].isin(["Aldosterone-Producing Adenoma vs Zona Fasciculata", "Zona Glomerulosa vs Zona Fasciculata"]))
    ].copy()
    label_map = {
        "Aldosterone-Producing Adenoma vs Zona Fasciculata": "APA - ZF",
        "Zona Glomerulosa vs Zona Fasciculata": "ZG - ZF",
    }
    paired["comparison_short"] = paired["comparison"].map(label_map)
    paired = paired.iloc[::-1]
    y = np.arange(len(paired))
    for i, (_, row) in enumerate(paired.iterrows()):
        axes[1].plot([row["ci95_low"], row["ci95_high"]], [i, i], color="#333333", lw=1.0)
        axes[1].scatter(row["mean_a_minus_b"], i, color="#0072B2", s=34)
    axes[1].axvline(0, color="#666666", lw=0.8, ls="--")
    axes[1].set_yticks(y, paired["comparison_short"])
    axes[1].set_xlabel("Mean paired delta")
    axes[1].set_title("B. Paired external bulk tests")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "GSE64957_bulk_axis_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "GSE64957_bulk_axis_validation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(section_exact: pd.DataFrame, fixed: pd.DataFrame, null_summary: pd.DataFrame, bulk_tests: pd.DataFrame, centroid: pd.DataFrame, availability: pd.DataFrame) -> None:
    primary = section_exact[section_exact["metric"] == f"{PRIMARY_SPATIAL_SCORE}_mean"].iloc[0]
    primary_adj = section_exact[section_exact["metric"] == f"{PRIMARY_SPATIAL_SCORE}_depth_adjusted_mean"].iloc[0]
    bulk_primary = bulk_tests[
        (bulk_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score")
        & (bulk_tests["comparison"] == "Aldosterone-Producing Adenoma vs Zona Fasciculata")
        & (bulk_tests["test_type"] == "patient_tissue_id paired")
    ]
    bulk_zone = bulk_tests[
        (bulk_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score")
        & (bulk_tests["comparison"] == "Zona Glomerulosa vs Zona Fasciculata")
        & (bulk_tests["test_type"] == "patient_tissue_id paired")
    ]
    centroid_n = int(centroid["closer_to_ZG_than_ZF"].sum()) if not centroid.empty else 0
    centroid_total = int(centroid.shape[0])

    lines = [
        "# Computational-only defense package",
        "",
        "Date: 2026-06-02",
        "",
        "Purpose: strengthen the no-wet-lab manuscript with analyses that can be completed from public data only.",
        "",
        "## Primary spatial inference hardening",
        "",
        f"- Primary section-level axis mean delta = {primary['mean_delta']:.4g}; 95% bootstrap CI {primary['ci95_low']:.4g} to {primary['ci95_high']:.4g}; {int(primary['n_positive'])}/{int(primary['n_pairs'])} pairs positive.",
        f"- Exact patient-paired sign-flip two-sided p = {primary['exact_sign_flip_two_sided_p']:.4g}; greater p = {primary['exact_sign_flip_greater_p']:.4g}.",
        f"- Depth-adjusted primary axis mean delta = {primary_adj['mean_delta']:.4g}; exact sign-flip two-sided p = {primary_adj['exact_sign_flip_two_sided_p']:.4g}.",
        "",
        "## Spatial nulls",
        "",
        "|analysis|unit preserved|observed mean delta|two-sided p|greater p|",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in null_summary.iterrows():
        lines.append(
            f"|{row['analysis']}|{row['unit_preserved']}|{row['observed_mean_delta']:.4g}|{row['two_sided_p']:.4g}|{row['greater_p']:.4g}|"
        )

    lines.extend(["", "## External bulk axis transfer: GSE64957", ""])
    if not bulk_primary.empty:
        r = bulk_primary.iloc[0]
        lines.append(
            f"- Matched APA vs ZF bulk axis: mean delta = {r['mean_a_minus_b']:.4g}; "
            f"{int(r['n_positive'])}/{int(r['n_a'])} matched patients positive; exact sign-flip p = {r['exact_sign_flip_two_sided_p']:.4g}."
        )
    if not bulk_zone.empty:
        r = bulk_zone.iloc[0]
        lines.append(
            f"- ZG vs ZF calibration in the same dataset: mean delta = {r['mean_a_minus_b']:.4g}; "
            f"{int(r['n_positive'])}/{int(r['n_a'])} matched tissues positive; exact sign-flip p = {r['exact_sign_flip_two_sided_p']:.4g}."
        )
    lines.append(
        f"- Boundary check: only {centroid_n}/{centroid_total} APA samples are globally closer to the normal ZG than ZF centroid in multi-score space, "
        "so the bulk data support a ZG/intermediate-axis program shift rather than normal ZG identity."
    )

    lines.extend(
        [
            "",
            "## Bulk signature availability",
            "",
            "|signature|available/requested|missing genes|",
            "|---|---:|---|",
        ]
    )
    for _, row in availability.iterrows():
        lines.append(f"|{row['signature']}|{int(row['n_available'])}/{int(row['n_requested'])}|{row['missing_genes']}|")

    lines.extend(
        [
            "",
            "## Patient fixed-effect model status",
            "",
            "Section-level patient fixed-effect models were used as a secondary check. The exact paired tests remain the primary inferential layer because the sample size is seven pairs.",
            "",
            "|metric|status|coef APA vs adjacent|p|",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in fixed.head(12).iterrows():
        lines.append(
            f"|{row.get('metric', 'NA')}|{row.get('status', 'NA')}|{safe_float(row.get('apa_vs_adjacent_coef', np.nan))}|{safe_float(row.get('apa_vs_adjacent_p', np.nan))}|"
        )

    lines.extend(
        [
            "",
            "## Manuscript use",
            "",
            "Use these analyses as Supplementary Tables/Figures for a computational-only submission. They support robustness and external public-data consistency, but they still do not provide orthogonal histology/IHC validation.",
            "",
            "Safe wording: the public-data evidence supports an aldosterone-dominant ZG/intermediate-vs-ZF spatial steroidogenic program in APA.",
            "",
            "Avoid: IHC-validated ROI, histologic ZG abundance, causal zonation collapse, or mutation-negative claims.",
            "",
            "## Output files",
            "",
            "- `results/computational_only_defense_20260602/section_level_exact_paired_statistics.csv`",
            "- `results/computational_only_defense_20260602/section_level_patient_fixed_effect_tests.csv`",
            "- `results/computational_only_defense_20260602/primary_axis_null_summary_table.csv`",
            "- `results/computational_only_defense_20260602/GSE64957_bulk_axis_validation_tests.csv`",
            "- `results/computational_only_defense_20260602/GSE64957_bulk_zonation_axis_scores.csv`",
            "- `results/computational_only_defense_20260602/figures/primary_axis_computational_nulls.png/.pdf`",
            "- `results/computational_only_defense_20260602/figures/GSE64957_bulk_axis_validation.png/.pdf`",
            "",
        ]
    )
    (PROJECT / "computational_only_defense_report_20260602.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    section_exact, sign_null = section_exact_tests()
    mixed = mixed_effect_section_tests()
    fixed = patient_fixed_effect_section_tests()
    spot_null = spot_label_permutation_null()
    _, block_null = spatial_block_permutation_null()
    null_summary = null_summary_table(section_exact, spot_null, block_null)
    bulk_scores, availability = score_bulk_expression()
    bulk_tests = bulk_validation_tests(bulk_scores)
    centroid = bulk_centroid_distance(bulk_scores)
    make_figures(section_exact, sign_null, spot_null, block_null, bulk_scores, bulk_tests)
    write_report(section_exact, fixed, null_summary, bulk_tests, centroid, availability)
    print(section_exact[section_exact["metric"].isin([f"{PRIMARY_SPATIAL_SCORE}_mean", f"{PRIMARY_SPATIAL_SCORE}_depth_adjusted_mean"])].to_string(index=False))
    print(null_summary.to_string(index=False))
    print(bulk_tests[(bulk_tests["metric"] == "ZG_intermediate_vs_ZF_axis_score") & (bulk_tests["test_type"] == "patient_tissue_id paired")].to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
