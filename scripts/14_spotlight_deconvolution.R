suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SingleCellExperiment)
  library(SpatialExperiment)
  library(scater)
  library(scran)
  library(SPOTlight)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grepl("^--file=", args)]
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
PROJECT <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
RAW <- file.path(PROJECT, "data", "raw")
PROCESSED <- file.path(PROJECT, "data", "processed")
RESULTS <- file.path(PROJECT, "results")
OUT_DIR <- file.path(RESULTS, "spotlight")
FIG_DIR <- file.path(RESULTS, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(20260601)

max_spots_per_section <- as.integer(Sys.getenv("SPOTLIGHT_MAX_SPOTS_PER_SECTION", "0"))
if (is.na(max_spots_per_section)) max_spots_per_section <- 0L
max_ref_per_label <- as.integer(Sys.getenv("SPOTLIGHT_MAX_REF_PER_LABEL", "700"))
if (is.na(max_ref_per_label) || max_ref_per_label < 20) max_ref_per_label <- 700L
marker_mode <- Sys.getenv("SPOTLIGHT_MARKER_MODE", "scran")
if (!marker_mode %in% c("scran", "panel")) marker_mode <- "scran"
run_base <- if (max_spots_per_section > 0) paste0("smoke_", max_spots_per_section) else "full"
run_tag <- paste(run_base, marker_mode, sep = "_")

PANELS <- list(
  ZG_like = c("CYP11B2", "AGTR1", "KCNJ5", "DACH1", "LGR5", "VSNL1", "ANO4", "PDE2A"),
  ZF_ZR_like = c("CYP11B1", "CYP17A1", "MC2R", "SULT2A1", "CYB5A", "PAPSS2"),
  intermediate_steroidogenic = c("CYP11A1", "STAR", "HSD3B2", "CYP21A2", "CYP11B1", "CYP17A1", "CYP11B2"),
  endothelial = c("PECAM1", "VWF", "KDR", "ENG", "CLDN5", "RAMP2"),
  fibroblast_stromal = c("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"),
  immune = c("PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1"),
  smooth_muscle_pericyte = c("ACTA2", "MYH11", "RGS5", "MCAM", "PDGFRB", "CSPG4"),
  medulla_chromaffin = c("TH", "DBH", "PNMT", "CHGA", "CHGB", "SLC18A1")
)

collapse_sparse_by_gene <- function(mat, gene_symbols) {
  gene_symbols <- as.character(gene_symbols)
  valid <- !is.na(gene_symbols) & gene_symbols != ""
  mat <- mat[valid, , drop = FALSE]
  gene_symbols <- gene_symbols[valid]
  unique_genes <- unique(gene_symbols)
  idx <- match(gene_symbols, unique_genes)
  sm <- summary(mat)
  collapsed <- sparseMatrix(
    i = idx[sm$i],
    j = sm$j,
    x = sm$x,
    dims = c(length(unique_genes), ncol(mat)),
    dimnames = list(unique_genes, colnames(mat))
  )
  as(collapsed, "dgCMatrix")
}

read_positions <- function(path) {
  x <- read.csv(gzfile(path), header = FALSE, stringsAsFactors = FALSE)
  if (tolower(as.character(x[1, 1])) == "barcode") x <- x[-1, , drop = FALSE]
  colnames(x) <- c("barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
  x$in_tissue <- as.integer(x$in_tissue)
  x$array_row <- as.numeric(x$array_row)
  x$array_col <- as.numeric(x$array_col)
  x$pxl_row_in_fullres <- as.numeric(x$pxl_row_in_fullres)
  x$pxl_col_in_fullres <- as.numeric(x$pxl_col_in_fullres)
  x
}

build_reference_sce <- function() {
  ref <- readRDS(file.path(RESULTS, "seurat_reference", "seurat_reference_object.rds"))
  labels <- ref$deconv_label
  names(labels) <- colnames(ref)
  keep <- !is.na(labels) & labels != ""
  ref <- subset(ref, cells = names(labels)[keep])
  labels <- labels[keep]
  cells_by_label <- split(names(labels), labels)
  cells_keep <- unlist(lapply(cells_by_label, function(cells) {
    if (length(cells) <= max_ref_per_label) cells else sample(cells, max_ref_per_label)
  }), use.names = FALSE)
  counts <- GetAssayData(ref, assay = "RNA", layer = "counts")[, cells_keep, drop = FALSE]
  labels <- labels[cells_keep]
  sce <- SingleCellExperiment(assays = list(counts = counts))
  sce$deconv_label <- labels
  sce <- logNormCounts(sce)
  hvg <- unique(c(VariableFeatures(ref), unlist(PANELS)))
  hvg <- intersect(hvg, rownames(sce))
  write.csv(
    data.frame(cell_type = names(table(labels)), n_reference_cells = as.integer(table(labels))),
    file.path(OUT_DIR, paste0("spotlight_reference_label_counts_", run_tag, ".csv")),
    row.names = FALSE
  )
  list(sce = sce, hvg = hvg)
}

build_spatial_spe <- function() {
  meta <- read.csv(file.path(PROCESSED, "GSE274314_sample_metadata.csv"), stringsAsFactors = FALSE)
  meta_by_gsm <- split(meta, meta$geo_accession)
  tar_path <- file.path(RAW, "GSE274314", "GSE274314_RAW.tar")
  tmp <- file.path(tempdir(), paste0("GSE274314_spotlight_", Sys.getpid()))
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  members <- utils::untar(tar_path, list = TRUE)
  needed <- members[grepl("(_matrix.mtx.gz|_features.tsv.gz|_barcodes.tsv.gz|_tissue_positions_list.csv.gz)$", members)]
  utils::untar(tar_path, files = needed, exdir = tmp)
  prefixes <- sort(sub("_matrix.mtx.gz$", "", needed[grepl("_matrix.mtx.gz$", needed)]))

  mats <- list()
  metas <- list()
  coords <- list()
  for (section_idx in seq_along(prefixes)) {
    prefix <- prefixes[[section_idx]]
    gsm <- sub("_.*$", "", prefix)
    sample_meta <- meta_by_gsm[[gsm]][1, , drop = FALSE]
    mat <- readMM(gzfile(file.path(tmp, paste0(prefix, "_matrix.mtx.gz"))))
    mat <- as(mat, "dgCMatrix")
    features <- read.delim(gzfile(file.path(tmp, paste0(prefix, "_features.tsv.gz"))), header = FALSE, stringsAsFactors = FALSE)
    barcodes <- readLines(gzfile(file.path(tmp, paste0(prefix, "_barcodes.tsv.gz"))), warn = FALSE)
    positions <- read_positions(file.path(tmp, paste0(prefix, "_tissue_positions_list.csv.gz")))
    colnames(mat) <- barcodes
    mat <- collapse_sparse_by_gene(mat, features[[min(2, ncol(features))]])
    tissue_barcodes <- positions$barcode[positions$in_tissue == 1]
    tissue_barcodes <- intersect(tissue_barcodes, colnames(mat))
    if (max_spots_per_section > 0 && length(tissue_barcodes) > max_spots_per_section) {
      tissue_barcodes <- sample(tissue_barcodes, max_spots_per_section)
    }
    mat <- mat[, tissue_barcodes, drop = FALSE]
    spot_ids <- paste(gsm, tissue_barcodes, sep = ":")
    colnames(mat) <- spot_ids
    pos <- positions[match(tissue_barcodes, positions$barcode), , drop = FALSE]
    spot_meta <- data.frame(
      spot_id = spot_ids,
      barcode = tissue_barcodes,
      geo_accession = gsm,
      title = sample_meta$title,
      patient = sample_meta$patient,
      tissue = sample_meta$tissue,
      genotype = sample_meta$genotype,
      sample_type = sample_meta$sample_type,
      stringsAsFactors = FALSE,
      row.names = spot_ids
    )
    coord <- data.frame(
      x = pos$pxl_col_in_fullres + (section_idx - 1) * 100000,
      y = pos$pxl_row_in_fullres,
      row.names = spot_ids
    )
    mats[[prefix]] <- mat
    metas[[prefix]] <- spot_meta
    coords[[prefix]] <- coord
  }
  common_genes <- sort(Reduce(intersect, lapply(mats, rownames)))
  counts <- do.call(cbind, lapply(mats, function(m) m[common_genes, , drop = FALSE]))
  spot_meta <- do.call(rbind, metas)
  coord_df <- do.call(rbind, coords)
  spot_meta <- spot_meta[colnames(counts), , drop = FALSE]
  coord_df <- coord_df[colnames(counts), , drop = FALSE]
  rownames(spot_meta) <- colnames(counts)
  rownames(coord_df) <- colnames(counts)
  coord_mat <- as.matrix(coord_df[, c("x", "y")])
  rownames(coord_mat) <- colnames(counts)
  SpatialExperiment(
    assays = list(counts = counts),
    colData = DataFrame(spot_meta, row.names = colnames(counts)),
    spatialCoords = coord_mat
  )
}

marker_table_panel <- function(genes_available) {
  rows <- list()
  for (label in names(PANELS)) {
    genes <- intersect(PANELS[[label]], genes_available)
    if (!length(genes)) next
    rows[[label]] <- data.frame(gene = genes, cluster = label, weight = 1, stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

marker_table_scran <- function(sce, genes_available) {
  keep_genes <- genes_available[!grepl("^MT-|^RPL|^RPS", genes_available)]
  colLabels(sce) <- sce$deconv_label
  markers <- scoreMarkers(sce, subset.row = keep_genes)
  rows <- list()
  for (label in names(markers)) {
    tab <- as.data.frame(markers[[label]])
    tab$gene <- rownames(tab)
    tab$cluster <- label
    tab <- tab[is.finite(tab$mean.AUC), , drop = FALSE]
    tab <- tab[tab$mean.AUC >= 0.65, , drop = FALSE]
    tab <- tab[order(tab$mean.AUC, decreasing = TRUE), , drop = FALSE]
    tab <- head(tab, 120)
    if (nrow(tab)) rows[[label]] <- tab
  }
  mgs <- do.call(rbind, rows)
  if (is.null(mgs) || !nrow(mgs)) {
    warning("scran marker table was empty; falling back to panel markers")
    mgs <- marker_table_panel(genes_available)
    mgs$mean.AUC <- mgs$weight
  }
  missing <- setdiff(unique(sce$deconv_label), unique(mgs$cluster))
  if (length(missing)) {
    panel <- marker_table_panel(genes_available)
    panel <- panel[panel$cluster %in% missing, , drop = FALSE]
    if (nrow(panel)) {
      panel$mean.AUC <- 0.65
      keep_cols <- union(colnames(mgs), colnames(panel))
      for (col in setdiff(keep_cols, colnames(mgs))) mgs[[col]] <- NA
      for (col in setdiff(keep_cols, colnames(panel))) panel[[col]] <- NA
      mgs <- rbind(mgs[, keep_cols], panel[, keep_cols])
    }
  }
  mgs
}

paired_tests <- function(mat, spot_meta) {
  sample_df <- cbind(spot_meta[, c("patient", "tissue", "genotype")], mat)
  sample_means <- aggregate(. ~ patient + tissue + genotype, data = sample_df, FUN = mean)
  write.csv(sample_means, file.path(OUT_DIR, paste0("GSE274314_SPOTlight_sample_mean_props_", run_tag, ".csv")), row.names = FALSE)
  labels <- setdiff(colnames(sample_means), c("patient", "tissue", "genotype"))
  paired_rows <- list()
  test_rows <- list()
  for (label in labels) {
    wide <- reshape(sample_means[, c("patient", "tissue", label)], idvar = "patient", timevar = "tissue", direction = "wide")
    apa_col <- paste0(label, ".APA")
    adj_col <- paste0(label, ".Adjacent")
    if (!all(c(apa_col, adj_col) %in% colnames(wide))) next
    wide <- wide[complete.cases(wide[, c(apa_col, adj_col)]), ]
    delta <- wide[[apa_col]] - wide[[adj_col]]
    p <- if (length(delta) >= 2) t.test(wide[[apa_col]], wide[[adj_col]], paired = TRUE)$p.value else NA_real_
    npos <- sum(delta > 0)
    sign_p <- if (length(delta) > 0) binom.test(npos, length(delta), p = 0.5, alternative = "greater")$p.value else NA_real_
    test_rows[[label]] <- data.frame(
      label = label,
      n_pairs = length(delta),
      mean_delta_APA_minus_adjacent = mean(delta),
      median_delta_APA_minus_adjacent = median(delta),
      n_positive = npos,
      n_negative = sum(delta < 0),
      paired_t_p = p,
      one_sided_sign_p_APA_gt_adjacent = sign_p
    )
    paired_rows[[label]] <- data.frame(
      patient = wide$patient,
      label = label,
      APA_minus_adjacent = delta,
      APA_mean = wide[[apa_col]],
      Adjacent_mean = wide[[adj_col]]
    )
  }
  tests <- do.call(rbind, test_rows)
  tests <- tests[order(tests$paired_t_p), ]
  paired <- do.call(rbind, paired_rows)
  write.csv(tests, file.path(OUT_DIR, paste0("GSE274314_SPOTlight_paired_tests_", run_tag, ".csv")), row.names = FALSE)
  write.csv(paired, file.path(OUT_DIR, paste0("GSE274314_SPOTlight_paired_delta_", run_tag, ".csv")), row.names = FALSE)
  list(tests = tests, paired = paired)
}

plot_paired <- function(paired) {
  wanted <- intersect(c("ZG_like", "ZF_ZR_like", "intermediate_steroidogenic", "endothelial", "fibroblast_stromal", "immune"), unique(paired$label))
  sub <- paired[paired$label %in% wanted, ]
  plot_df <- rbind(
    data.frame(patient = sub$patient, label = sub$label, tissue = "Adjacent", prop = sub$Adjacent_mean),
    data.frame(patient = sub$patient, label = sub$label, tissue = "APA", prop = sub$APA_mean)
  )
  plot_df$tissue <- factor(plot_df$tissue, levels = c("Adjacent", "APA"))
  p <- ggplot(plot_df, aes(tissue, prop, group = patient)) +
    facet_wrap(~ label, scales = "free_y", nrow = 1) +
    geom_line(color = "#777777", alpha = 0.7) +
    geom_point(size = 1.8, color = "#2ca02c") +
    theme_classic(base_size = 9) +
    labs(title = paste("GSE274314 SPOTlight paired proportions", run_tag), x = NULL, y = "SPOTlight proportion")
  ggsave(file.path(FIG_DIR, paste0("GSE274314_SPOTlight_paired_props_", run_tag, ".png")), p, width = 11, height = 3.8, dpi = 220)
}

write_report <- function(tests, sce, spe, mgs) {
  ref_counts <- as.data.frame(table(sce$deconv_label), stringsAsFactors = FALSE)
  colnames(ref_counts) <- c("cell_type", "n_reference_cells")
  lines <- c(
    "# SPOTlight deconvolution",
    "",
    "Date: 2026-06-01",
    "",
    paste0("Run tag: `", run_tag, "`"),
    "",
    paste0("Spatial spots: ", ncol(spe)),
    paste0("Reference cells: ", ncol(sce)),
    paste0("Common genes: ", length(intersect(rownames(sce), rownames(spe)))),
    paste0("Marker rows used: ", nrow(mgs)),
    "",
    "## Reference label counts",
    "",
    "|cell_type|n_reference_cells|",
    "|---|---:|",
    paste0("|", ref_counts$cell_type, "|", ref_counts$n_reference_cells, "|"),
    "",
    "## Paired APA-vs-adjacent SPOTlight tests",
    "",
    "|label|n_pairs|mean_delta_APA_minus_adjacent|n_positive|n_negative|paired_t_p|one_sided_sign_p_APA_gt_adjacent|",
    "|---|---:|---:|---:|---:|---:|---:|",
    apply(tests, 1, function(row) {
      paste0("|", row[["label"]], "|", row[["n_pairs"]], "|", signif(as.numeric(row[["mean_delta_APA_minus_adjacent"]]), 4), "|", row[["n_positive"]], "|", row[["n_negative"]], "|", signif(as.numeric(row[["paired_t_p"]]), 4), "|", signif(as.numeric(row[["one_sided_sign_p_APA_gt_adjacent"]]), 4), "|")
    }),
    "",
    "## Key outputs",
    "",
    paste0("- `results/spotlight/GSE274314_SPOTlight_props_", run_tag, ".csv`"),
    paste0("- `results/spotlight/GSE274314_SPOTlight_paired_tests_", run_tag, ".csv`"),
    paste0("- `results/figures/GSE274314_SPOTlight_paired_props_", run_tag, ".png`"),
    "",
    "## Claim boundary",
    "",
    "This is an established package benchmark, but marker genes were supplied from curated adrenal panels rather than discovered de novo with a full marker workflow. Interpret as a deconvolution benchmark, not histologic cell counting.",
    ""
  )
  writeLines(lines, file.path(PROJECT, paste0("spotlight_deconvolution_report_20260601_", run_tag, ".md")), useBytes = TRUE)
}

cat("Building SPOTlight reference...\n")
ref <- build_reference_sce()
sce <- ref$sce
cat("Building spatial object...\n")
spe <- build_spatial_spe()
common <- sort(intersect(rownames(sce), rownames(spe)))
sce <- sce[common, ]
spe <- spe[common, ]
mgs <- if (marker_mode == "scran") marker_table_scran(sce, common) else marker_table_panel(common)
hvg <- intersect(unique(c(ref$hvg, mgs$gene)), common)
write.csv(mgs, file.path(OUT_DIR, paste0("spotlight_marker_table_", run_tag, ".csv")), row.names = FALSE)
weight_col <- if (marker_mode == "scran" && "mean.AUC" %in% colnames(mgs)) "mean.AUC" else "weight"

cat("Running SPOTlight on", ncol(spe), "spots and", ncol(sce), "reference cells...\n")
res <- SPOTlight(
  x = sce,
  y = spe,
  groups = as.character(sce$deconv_label),
  mgs = mgs,
  hvg = hvg,
  weight_id = weight_col,
  group_id = "cluster",
  gene_id = "gene",
  slot_sc = "counts",
  slot_sp = "counts",
  min_prop = 0.001,
  verbose = TRUE
)
saveRDS(res, file.path(OUT_DIR, paste0("GSE274314_SPOTlight_result_", run_tag, ".rds")))
mat <- res$mat
write.csv(data.frame(spot_id = rownames(mat), mat, check.names = FALSE), file.path(OUT_DIR, paste0("GSE274314_SPOTlight_props_", run_tag, ".csv")), row.names = FALSE)
sample_meta <- read.csv(file.path(PROCESSED, "GSE274314_sample_metadata.csv"), stringsAsFactors = FALSE)
gsm <- sub(":.*$", "", rownames(mat))
spot_meta <- data.frame(spot_id = rownames(mat), geo_accession = gsm, stringsAsFactors = FALSE)
spot_meta <- merge(spot_meta, sample_meta, by = "geo_accession", all.x = TRUE, sort = FALSE)
rownames(spot_meta) <- spot_meta$spot_id
spot_meta <- spot_meta[rownames(mat), , drop = FALSE]
pt <- paired_tests(mat, spot_meta)
plot_paired(pt$paired)
write_report(pt$tests, sce, spe, mgs)
cat("DONE\n")
print(pt$tests)
