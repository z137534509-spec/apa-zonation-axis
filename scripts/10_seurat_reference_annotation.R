suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grepl("^--file=", args)]
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
PROJECT <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
RAW <- file.path(PROJECT, "data", "raw")
RESULTS <- file.path(PROJECT, "results")
OUT_DIR <- file.path(RESULTS, "seurat_reference")
FIG_DIR <- file.path(RESULTS, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
set.seed(20260601)

PANELS <- list(
  cortical_steroidogenic = c("CYP11A1", "STAR", "HSD3B2", "CYP21A2", "CYP11B1", "CYP17A1", "CYP11B2"),
  ZG_like = c("CYP11B2", "AGTR1", "KCNJ5", "DACH1", "LGR5", "VSNL1", "ANO4", "PDE2A"),
  ZF_ZR_like = c("CYP11B1", "CYP17A1", "MC2R", "SULT2A1", "CYB5A", "PAPSS2"),
  endothelial = c("PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"),
  fibroblast_stromal = c("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"),
  immune = c("PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1"),
  smooth_muscle_pericyte = c("ACTA2", "MYH11", "RGS5", "MCAM", "PDGFRB", "CSPG4"),
  medulla_chromaffin = c("TH", "DBH", "PNMT", "CHGA", "CHGB", "SLC18A1")
)

DECONV_LABELS <- c(
  "ZG_like",
  "ZF_ZR_like",
  "intermediate_steroidogenic",
  "endothelial",
  "fibroblast_stromal",
  "immune",
  "smooth_muscle_pericyte",
  "medulla_chromaffin"
)

DECONV_GENES <- sort(unique(c(
  unlist(PANELS),
  "CYP11A1", "CYP11B2", "CYP11B1", "CYP17A1", "HSD3B2", "STAR", "CYP21A2",
  "SULT2A1", "CYB5A", "PAPSS2", "COL4A1", "COL4A2", "TAGLN", "ADGRL4", "LYVE1"
)))

read_tsv_maybe_gz <- function(path, header = FALSE) {
  con <- if (grepl("\\.gz$", path)) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(con), add = TRUE)
  read.delim(con, header = header, stringsAsFactors = FALSE, sep = "\t")
}

read_lines_maybe_gz <- function(path) {
  con <- if (grepl("\\.gz$", path)) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(con), add = TRUE)
  readLines(con, warn = FALSE)
}

read_mtx_maybe_gz <- function(path) {
  con <- if (grepl("\\.gz$", path)) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(con), add = TRUE)
  as(readMM(con), "dgCMatrix")
}

sample_specs <- function() {
  sn_dir <- file.path(RAW, "Zenodo_18128100_normal_snRNA")
  apcc_dir <- file.path(RAW, "E-MTAB-11837", "processed")
  specs <- list()
  for (sample in c("NAd1", "NAd2", "NAd3", "NAd4")) {
    specs[[length(specs) + 1]] <- list(
      dataset = "normal_snRNA",
      sample_id = sample,
      matrix = file.path(sn_dir, paste0(sample, "_matrix.mtx.gz")),
      features = file.path(sn_dir, paste0(sample, "_features.tsv.gz")),
      barcodes = file.path(sn_dir, paste0(sample, "_barcodes.tsv.gz"))
    )
  }
  for (sample in c("Sample1", "Sample2")) {
    specs[[length(specs) + 1]] <- list(
      dataset = "APCC_adrenal_scRNA",
      sample_id = sample,
      matrix = file.path(apcc_dir, paste0(sample, "_matrix.mtx")),
      features = file.path(apcc_dir, paste0(sample, "_features.tsv")),
      barcodes = file.path(apcc_dir, paste0(sample, "_barcodes.tsv"))
    )
  }
  specs
}

load_one_sample <- function(spec) {
  mat <- read_mtx_maybe_gz(spec$matrix)
  features <- read_tsv_maybe_gz(spec$features, header = FALSE)
  barcodes <- read_lines_maybe_gz(spec$barcodes)
  gene_symbol <- as.character(features[[min(2, ncol(features))]])
  keep <- !duplicated(gene_symbol)
  mat <- mat[keep, , drop = FALSE]
  gene_symbol <- gene_symbol[keep]
  rownames(mat) <- gene_symbol
  colnames(mat) <- paste(spec$dataset, spec$sample_id, barcodes, sep = ":")
  meta <- data.frame(
    cell_id = colnames(mat),
    dataset = spec$dataset,
    sample_id = spec$sample_id,
    barcode = barcodes,
    stringsAsFactors = FALSE,
    row.names = colnames(mat)
  )
  list(mat = mat, meta = meta, genes = gene_symbol)
}

cat("Loading single-cell matrices...\n")
loaded <- lapply(sample_specs(), load_one_sample)
common_genes <- Reduce(intersect, lapply(loaded, function(x) x$genes))
common_genes <- sort(common_genes)
blocks <- lapply(loaded, function(x) x$mat[common_genes, , drop = FALSE])
counts <- do.call(cbind, blocks)
meta <- do.call(rbind, lapply(loaded, function(x) x$meta))

cat("Creating Seurat object with", ncol(counts), "cells and", nrow(counts), "common genes...\n")
obj <- CreateSeuratObject(counts = counts, meta.data = meta, project = "PA_adrenal_reference", min.cells = 0, min.features = 0)
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

qc_before <- obj@meta.data
qc_before$cell_id <- rownames(qc_before)
write.csv(qc_before, file.path(OUT_DIR, "seurat_qc_before_filter.csv"), row.names = FALSE)

obj <- subset(obj, subset = nCount_RNA >= 500 & nFeature_RNA >= 300 & percent.mt <= 20)
qc_after <- obj@meta.data
qc_after$cell_id <- rownames(qc_after)
write.csv(qc_after, file.path(OUT_DIR, "seurat_qc_pass_metadata.csv"), row.names = FALSE)

qc_summary <- aggregate(
  cbind(nCount_RNA, nFeature_RNA, percent.mt) ~ dataset + sample_id,
  data = obj@meta.data,
  FUN = median
)
qc_summary$n_cells <- as.integer(table(paste(obj$dataset, obj$sample_id, sep = "|"))[paste(qc_summary$dataset, qc_summary$sample_id, sep = "|")])
write.csv(qc_summary, file.path(OUT_DIR, "seurat_qc_summary.csv"), row.names = FALSE)

cat("Running Seurat normalization, PCA, neighbors, clustering...\n")
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = 2500, verbose = FALSE)
marker_genes <- sort(unique(c(unlist(PANELS), "CYP11B2", "CYP11B1", "CYP17A1", "HSD3B2", "STAR", "CYP21A2")))
features_use <- sort(unique(c(VariableFeatures(obj), intersect(marker_genes, rownames(obj)))))
obj <- ScaleData(obj, features = features_use, verbose = FALSE)
obj <- RunPCA(obj, features = features_use, npcs = 30, verbose = FALSE)
obj <- FindNeighbors(obj, dims = 1:20, verbose = FALSE)
obj <- FindClusters(obj, resolution = 0.5, algorithm = 1, verbose = FALSE)

data_mat <- GetAssayData(obj, assay = "RNA", layer = "data")
score_module <- function(genes) {
  present <- intersect(genes, rownames(data_mat))
  if (length(present) == 0) return(rep(NA_real_, ncol(data_mat)))
  Matrix::colMeans(data_mat[present, , drop = FALSE])
}

for (nm in names(PANELS)) {
  obj[[paste0(nm, "_score")]] <- score_module(PANELS[[nm]])
}

gene_expr <- function(gene) {
  if (!gene %in% rownames(data_mat)) return(rep(NA_real_, ncol(data_mat)))
  as.numeric(data_mat[gene, ])
}
for (gene in c("CYP11B2", "CYP11B1", "CYP17A1", "HSD3B2", "STAR", "CYP21A2", "AGTR1", "KCNJ5", "PTPRC", "PECAM1", "COL1A1", "TH")) {
  obj[[paste0(gene, "_expr")]] <- gene_expr(gene)
}

meta2 <- obj@meta.data
meta2$cell_id <- rownames(meta2)
meta2$seurat_cluster <- as.character(Idents(obj))

score_cols <- paste0(names(PANELS), "_score")
expr_cols <- paste0(c("CYP11B2", "CYP11B1", "CYP17A1", "HSD3B2", "STAR", "CYP21A2", "AGTR1", "KCNJ5", "PTPRC", "PECAM1", "COL1A1", "TH"), "_expr")
cluster_ids <- sort(unique(meta2$seurat_cluster))

assign_label <- function(row) {
  non_steroid <- c("endothelial", "fibroblast_stromal", "immune", "smooth_muscle_pericyte", "medulla_chromaffin")
  non_scores <- as.numeric(row[paste0(non_steroid, "_score")])
  names(non_scores) <- non_steroid
  best_non <- names(non_scores)[which.max(non_scores)]
  best_non_score <- max(non_scores, na.rm = TRUE)
  cortical <- as.numeric(row["cortical_steroidogenic_score"])
  zg <- as.numeric(row["ZG_like_score"])
  zf <- as.numeric(row["ZF_ZR_like_score"])
  if (is.finite(best_non_score) && is.finite(cortical) && best_non_score > cortical + 0.2) return(best_non)
  if (is.finite(zg) && is.finite(zf) && zg > zf + 0.3 && zg > 0.2) return("ZG_like")
  if (is.finite(zg) && is.finite(zf) && zf > zg + 0.3 && zf > 0.2) return("ZF_ZR_like")
  if (is.finite(cortical) && cortical > 0.2) return("intermediate_steroidogenic")
  best_non
}

cluster_summary <- do.call(rbind, lapply(cluster_ids, function(cl) {
  idx <- which(meta2$seurat_cluster == cl)
  row <- data.frame(seurat_cluster = cl, n_cells = length(idx), stringsAsFactors = FALSE)
  for (col in c(score_cols, expr_cols)) {
    row[[col]] <- mean(meta2[[col]][idx], na.rm = TRUE)
  }
  sample_tab <- sort(table(meta2$sample_id[idx]), decreasing = TRUE)
  row$top_sample <- names(sample_tab)[1]
  row$top_sample_fraction <- as.numeric(sample_tab[1]) / length(idx)
  row$deconv_label <- assign_label(row[1, ])
  row
}))
write.csv(cluster_summary, file.path(OUT_DIR, "seurat_cluster_marker_annotation_summary.csv"), row.names = FALSE)

label_map <- setNames(cluster_summary$deconv_label, cluster_summary$seurat_cluster)
meta2$deconv_label <- unname(label_map[meta2$seurat_cluster])
obj$seurat_cluster <- meta2$seurat_cluster
obj$deconv_label <- meta2$deconv_label
write.csv(meta2, file.path(OUT_DIR, "seurat_cell_cluster_annotations.csv"), row.names = FALSE)

label_counts <- as.data.frame(table(meta2$deconv_label), stringsAsFactors = FALSE)
colnames(label_counts) <- c("deconv_label", "n_cells")
write.csv(label_counts, file.path(OUT_DIR, "seurat_reference_label_counts.csv"), row.names = FALSE)

cluster_sample_contribution <- do.call(rbind, lapply(cluster_ids, function(cl) {
  idx <- which(meta2$seurat_cluster == cl)
  tab <- as.data.frame(table(meta2$sample_id[idx]), stringsAsFactors = FALSE)
  colnames(tab) <- c("sample_id", "n")
  tab$seurat_cluster <- cl
  tab$fraction <- tab$n / sum(tab$n)
  tab[, c("seurat_cluster", "sample_id", "n", "fraction")]
}))
write.csv(cluster_sample_contribution, file.path(OUT_DIR, "seurat_cluster_sample_contribution.csv"), row.names = FALSE)

present_deconv_genes <- intersect(DECONV_GENES, rownames(data_mat))
profiles <- data.frame(gene = present_deconv_genes, stringsAsFactors = FALSE)
for (label in DECONV_LABELS) {
  cells <- rownames(meta2)[meta2$deconv_label == label]
  if (length(cells) == 0) {
    profiles[[label]] <- NA_real_
  } else {
    profiles[[label]] <- Matrix::rowMeans(data_mat[present_deconv_genes, cells, drop = FALSE])
  }
}
write.csv(profiles, file.path(OUT_DIR, "seurat_reference_deconv_profiles.csv"), row.names = FALSE)

pca <- Embeddings(obj, reduction = "pca")[, 1:2, drop = FALSE]
pca_df <- data.frame(
  PC_1 = pca[, 1],
  PC_2 = pca[, 2],
  sample_id = obj$sample_id,
  dataset = obj$dataset,
  seurat_cluster = as.character(Idents(obj)),
  deconv_label = meta2$deconv_label,
  CYP11B2_expr = meta2$CYP11B2_expr,
  CYP11B1_expr = meta2$CYP11B1_expr,
  stringsAsFactors = FALSE
)
write.csv(pca_df, file.path(OUT_DIR, "seurat_pca_cell_embeddings.csv"), row.names = FALSE)
saveRDS(obj, file.path(OUT_DIR, "seurat_reference_object.rds"))

png(file.path(FIG_DIR, "seurat_reference_pca_overview.png"), width = 1800, height = 520, res = 150)
p1 <- ggplot(pca_df, aes(PC_1, PC_2, color = deconv_label)) +
  geom_point(size = 0.25, alpha = 0.6) +
  theme_classic(base_size = 10) +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  ggtitle("Seurat reference labels")
p2 <- ggplot(pca_df, aes(PC_1, PC_2, color = CYP11B2_expr)) +
  geom_point(size = 0.25, alpha = 0.6) +
  scale_color_viridis_c(option = "magma") +
  theme_classic(base_size = 10) +
  ggtitle("CYP11B2")
p3 <- ggplot(pca_df, aes(PC_1, PC_2, color = CYP11B1_expr)) +
  geom_point(size = 0.25, alpha = 0.6) +
  scale_color_viridis_c(option = "viridis") +
  theme_classic(base_size = 10) +
  ggtitle("CYP11B1")
print(p1 + p2 + p3)
dev.off()

score_space <- cluster_summary[cluster_summary$deconv_label %in% c("ZG_like", "ZF_ZR_like", "intermediate_steroidogenic"), ]
png(file.path(FIG_DIR, "seurat_steroidogenic_cluster_score_space.png"), width = 900, height = 650, res = 150)
ggplot(score_space, aes(ZG_like_score, ZF_ZR_like_score, color = deconv_label, size = n_cells, label = seurat_cluster)) +
  geom_point(alpha = 0.75) +
  geom_text(vjust = -0.7, size = 3, color = "black") +
  theme_classic(base_size = 11) +
  labs(title = "Seurat steroidogenic cluster score space", x = "Cluster mean ZG-like score", y = "Cluster mean ZF/ZR-like score")
dev.off()

cat("Done. Outputs written to", OUT_DIR, "\n")
