suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SummarizedExperiment)
  library(SpatialExperiment)
  library(spacexr)
  library(ggplot2)
})

project_override <- Sys.getenv("PROJECT_OVERRIDE", "")
if (nzchar(project_override)) {
  PROJECT <- gsub("\\\\", "/", project_override)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grepl("^--file=", args)]
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
  PROJECT <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
}
RAW <- file.path(PROJECT, "data", "raw")
PROCESSED <- file.path(PROJECT, "data", "processed")
RESULTS <- file.path(PROJECT, "results")
OUT_DIR <- file.path(RESULTS, "rctd_spacexr")
FIG_DIR <- file.path(RESULTS, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(20260601)

max_spots_per_section <- as.integer(Sys.getenv("RCTD_MAX_SPOTS_PER_SECTION", "0"))
if (is.na(max_spots_per_section)) max_spots_per_section <- 0L
max_ref_per_label <- as.integer(Sys.getenv("RCTD_MAX_REF_PER_LABEL", "1000"))
if (is.na(max_ref_per_label) || max_ref_per_label < 25) max_ref_per_label <- 1000L
max_genes <- as.integer(Sys.getenv("RCTD_MAX_GENES", "6000"))
if (is.na(max_genes) || max_genes < 200) max_genes <- 6000L
rctd_cores <- as.integer(Sys.getenv("RCTD_CORES", "2"))
if (is.na(rctd_cores) || rctd_cores < 1) rctd_cores <- 2L
run_tag <- if (max_spots_per_section > 0) paste0("smoke_", max_spots_per_section) else "full"

MARKER_GENES <- unique(c(
  "CYP11A1", "STAR", "HSD3B2", "CYP21A2", "CYP11B1", "CYP17A1", "CYP11B2",
  "AGTR1", "KCNJ5", "DACH1", "LGR5", "VSNL1", "ANO4", "PDE2A",
  "MC2R", "SULT2A1", "CYB5A", "PAPSS2", "PECAM1", "VWF", "KDR", "ENG",
  "CLDN5", "RAMP2", "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA",
  "PTPRC", "LST1", "HLA-DRA", "CD74", "CD3D", "NKG7", "MS4A1", "ACTA2",
  "MYH11", "RGS5", "MCAM", "PDGFRB", "CSPG4", "TH", "DBH", "PNMT",
  "CHGA", "CHGB", "SLC18A1"
))

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
  if (tolower(as.character(x[1, 1])) == "barcode") {
    x <- x[-1, , drop = FALSE]
  }
  colnames(x) <- c("barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
  x$in_tissue <- as.integer(x$in_tissue)
  x$array_row <- as.numeric(x$array_row)
  x$array_col <- as.numeric(x$array_col)
  x$pxl_row_in_fullres <- as.numeric(x$pxl_row_in_fullres)
  x$pxl_col_in_fullres <- as.numeric(x$pxl_col_in_fullres)
  x
}

build_reference <- function() {
  ref_path <- file.path(RESULTS, "seurat_reference", "seurat_reference_object.rds")
  ref <- readRDS(ref_path)
  if (!"deconv_label" %in% colnames(ref@meta.data)) {
    stop("The Seurat reference object lacks deconv_label metadata.")
  }
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
  umi <- Matrix::colSums(counts)
  reference_se <- SummarizedExperiment(
    assays = list(counts = counts),
    colData = DataFrame(cell_type = labels, nUMI = umi)
  )
  write.csv(
    data.frame(cell_type = names(table(labels)), n_reference_cells = as.integer(table(labels))),
    file.path(OUT_DIR, paste0("rctd_reference_label_counts_", run_tag, ".csv")),
    row.names = FALSE
  )
  reference_se
}

select_rctd_genes <- function(reference_se, spatial_spe) {
  ref_counts <- assay(reference_se, "counts")
  sp_counts <- assay(spatial_spe, "counts")
  ref_detect <- Matrix::rowSums(ref_counts > 0)
  sp_detect <- Matrix::rowSums(sp_counts > 0)
  ref_mean <- Matrix::rowMeans(ref_counts)
  sp_mean <- Matrix::rowMeans(sp_counts)
  valid <- ref_detect >= 10 & sp_detect >= 3
  score <- log1p(ref_mean) + log1p(sp_mean)
  score[!valid] <- -Inf
  ordered <- names(sort(score, decreasing = TRUE))
  top <- ordered[is.finite(score[ordered])]
  top <- head(top, max_genes)
  selected <- sort(unique(c(top, intersect(MARKER_GENES, rownames(reference_se)))))
  selected <- intersect(selected, rownames(reference_se))
  selected
}

build_spatial <- function() {
  meta <- read.csv(file.path(PROCESSED, "GSE274314_sample_metadata.csv"), stringsAsFactors = FALSE)
  meta_by_gsm <- split(meta, meta$geo_accession)
  tar_path <- file.path(RAW, "GSE274314", "GSE274314_RAW.tar")
  tmp <- file.path(tempdir(), paste0("GSE274314_rctd_", Sys.getpid()))
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
    gene_symbols <- features[[min(2, ncol(features))]]
    mat <- collapse_sparse_by_gene(mat, gene_symbols)
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
  common_genes <- Reduce(intersect, lapply(mats, rownames))
  common_genes <- sort(common_genes)
  counts <- do.call(cbind, lapply(mats, function(m) m[common_genes, , drop = FALSE]))
  spot_meta <- do.call(rbind, metas)
  coord_df <- do.call(rbind, coords)
  spot_meta <- spot_meta[colnames(counts), , drop = FALSE]
  coord_df <- coord_df[colnames(counts), , drop = FALSE]
  rownames(spot_meta) <- colnames(counts)
  rownames(coord_df) <- colnames(counts)
  coord_mat <- as.matrix(coord_df[, c("x", "y")])
  rownames(coord_mat) <- colnames(counts)
  spatial_spe <- SpatialExperiment(
    assays = list(counts = counts),
    colData = DataFrame(spot_meta, row.names = colnames(counts)),
    spatialCoords = coord_mat
  )
  spatial_spe
}

paired_tests <- function(weights, spot_meta) {
  sample_rows <- list()
  for (label in rownames(weights)) {
    df <- data.frame(
      label = label,
      weight = as.numeric(weights[label, ]),
      patient = spot_meta$patient,
      tissue = spot_meta$tissue,
      genotype = spot_meta$genotype,
      stringsAsFactors = FALSE
    )
    sample_mean <- aggregate(weight ~ patient + tissue + genotype, data = df, FUN = mean)
    sample_rows[[label]] <- sample_mean
  }
  sample_df <- do.call(rbind, Map(function(label, df) transform(df, label = label), names(sample_rows), sample_rows))
  write.csv(sample_df, file.path(OUT_DIR, paste0("GSE274314_RCTD_sample_mean_weights_", run_tag, ".csv")), row.names = FALSE)

  paired_rows <- list()
  test_rows <- list()
  for (label in unique(sample_df$label)) {
    sub <- sample_df[sample_df$label == label, ]
    wide <- reshape(sub[, c("patient", "tissue", "weight")], idvar = "patient", timevar = "tissue", direction = "wide")
    if (!all(c("weight.APA", "weight.Adjacent") %in% colnames(wide))) next
    wide <- wide[complete.cases(wide[, c("weight.APA", "weight.Adjacent")]), ]
    delta <- wide$weight.APA - wide$weight.Adjacent
    p <- if (length(delta) >= 2) t.test(wide$weight.APA, wide$weight.Adjacent, paired = TRUE)$p.value else NA_real_
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
      APA_mean = wide$weight.APA,
      Adjacent_mean = wide$weight.Adjacent
    )
  }
  tests <- do.call(rbind, test_rows)
  tests <- tests[order(tests$paired_t_p), ]
  paired <- do.call(rbind, paired_rows)
  write.csv(tests, file.path(OUT_DIR, paste0("GSE274314_RCTD_paired_tests_", run_tag, ".csv")), row.names = FALSE)
  write.csv(paired, file.path(OUT_DIR, paste0("GSE274314_RCTD_paired_delta_", run_tag, ".csv")), row.names = FALSE)
  list(tests = tests, paired = paired)
}

plot_paired <- function(paired) {
  wanted <- intersect(c("ZG_like", "ZF_ZR_like", "intermediate_steroidogenic", "endothelial", "fibroblast_stromal", "immune"), unique(paired$label))
  sub <- paired[paired$label %in% wanted, ]
  plot_df <- rbind(
    data.frame(patient = sub$patient, label = sub$label, tissue = "Adjacent", weight = sub$Adjacent_mean),
    data.frame(patient = sub$patient, label = sub$label, tissue = "APA", weight = sub$APA_mean)
  )
  plot_df$tissue <- factor(plot_df$tissue, levels = c("Adjacent", "APA"))
  p <- ggplot(plot_df, aes(x = tissue, y = weight, group = patient)) +
    facet_wrap(~ label, scales = "free_y", nrow = 1) +
    geom_line(aes(group = patient), color = "#777777", alpha = 0.7) +
    geom_point(size = 1.8, color = "#1f77b4") +
    theme_classic(base_size = 9) +
    labs(title = paste("GSE274314 RCTD paired weights", run_tag), x = NULL, y = "RCTD weight")
  ggsave(file.path(FIG_DIR, paste0("GSE274314_RCTD_paired_weights_", run_tag, ".png")), p, width = 11, height = 3.8, dpi = 220)
}

write_report <- function(tests, reference_se, spatial_spe, assay_name) {
  ref_counts <- as.data.frame(table(colData(reference_se)$cell_type), stringsAsFactors = FALSE)
  colnames(ref_counts) <- c("cell_type", "n_reference_cells")
  lines <- c(
    "# RCTD spacexr deconvolution",
    "",
    "Date: 2026-06-01",
    "",
    paste0("Run tag: `", run_tag, "`"),
    "",
    paste0("Spatial spots: ", ncol(spatial_spe)),
    paste0("Reference cells: ", ncol(reference_se)),
    paste0("Common genes used: ", nrow(spatial_spe)),
    paste0("RCTD assay used for paired tests: `", assay_name, "`"),
    "",
    "## Reference label counts",
    "",
    "|cell_type|n_reference_cells|",
    "|---|---:|",
    paste0("|", ref_counts$cell_type, "|", ref_counts$n_reference_cells, "|"),
    "",
    "## Paired APA-vs-adjacent RCTD tests",
    "",
    "|label|n_pairs|mean_delta_APA_minus_adjacent|n_positive|n_negative|paired_t_p|one_sided_sign_p_APA_gt_adjacent|",
    "|---|---:|---:|---:|---:|---:|---:|",
    apply(tests, 1, function(row) {
      paste0("|", row[["label"]], "|", row[["n_pairs"]], "|", signif(as.numeric(row[["mean_delta_APA_minus_adjacent"]]), 4), "|", row[["n_positive"]], "|", row[["n_negative"]], "|", signif(as.numeric(row[["paired_t_p"]]), 4), "|", signif(as.numeric(row[["one_sided_sign_p_APA_gt_adjacent"]]), 4), "|")
    }),
    "",
    "## Key outputs",
    "",
    paste0("- `results/rctd_spacexr/GSE274314_RCTD_weights_", run_tag, ".csv`"),
    paste0("- `results/rctd_spacexr/GSE274314_RCTD_paired_tests_", run_tag, ".csv`"),
    paste0("- `results/rctd_spacexr/GSE274314_RCTD_paired_delta_", run_tag, ".csv`"),
    paste0("- `results/figures/GSE274314_RCTD_paired_weights_", run_tag, ".png`"),
    "",
    "## Claim boundary",
    "",
    "RCTD is an established spatial deconvolution method and is stronger than the earlier NNLS checks. The result still estimates spot-level mixture weights rather than directly counting histologic ZG cells.",
    ""
  )
  writeLines(lines, file.path(PROJECT, paste0("rctd_spacexr_deconvolution_report_20260601_", run_tag, ".md")), useBytes = TRUE)
}

cat("Building reference...\n")
reference_se <- build_reference()
cat("Building spatial object...\n")
spatial_spe <- build_spatial()
common <- intersect(rownames(reference_se), rownames(spatial_spe))
common <- sort(common)
reference_se <- reference_se[common, ]
spatial_spe <- spatial_spe[common, ]
selected_genes <- select_rctd_genes(reference_se, spatial_spe)
reference_se <- reference_se[selected_genes, ]
spatial_spe <- spatial_spe[selected_genes, ]

cat("Creating RCTD object with", ncol(spatial_spe), "spots,", ncol(reference_se), "reference cells, and", length(selected_genes), "genes...\n")
rctd_data <- createRctd(
  spatial_spe,
  reference_se,
  cell_type_col = "cell_type",
  UMI_min = 50,
  ref_UMI_min = 100,
  ref_n_cells_min = 25,
  ref_n_cells_max = max_ref_per_label
)

cat("Running RCTD...\n")
results_spe <- runRctd(rctd_data, rctd_mode = "multi", max_cores = rctd_cores, max_multi_types = 4)
saveRDS(results_spe, file.path(OUT_DIR, paste0("GSE274314_RCTD_results_", run_tag, ".rds")))

assay_names <- assayNames(results_spe)
assay_name <- if ("weights" %in% assay_names) "weights" else if ("weights_full" %in% assay_names) "weights_full" else assay_names[[1]]
weights <- assay(results_spe, assay_name)
weights <- as.matrix(weights)
col_sums <- colSums(weights)
if (any(col_sums > 0, na.rm = TRUE)) {
  weights[, col_sums > 0] <- sweep(weights[, col_sums > 0, drop = FALSE], 2, col_sums[col_sums > 0], "/")
}
write.csv(
  data.frame(spot_id = colnames(weights), t(weights), check.names = FALSE),
  file.path(OUT_DIR, paste0("GSE274314_RCTD_weights_", run_tag, ".csv")),
  row.names = FALSE
)
sample_meta <- read.csv(file.path(PROCESSED, "GSE274314_sample_metadata.csv"), stringsAsFactors = FALSE)
gsm <- sub(":.*$", "", colnames(weights))
spot_meta <- data.frame(spot_id = colnames(weights), geo_accession = gsm, stringsAsFactors = FALSE)
spot_meta <- merge(spot_meta, sample_meta, by = "geo_accession", all.x = TRUE, sort = FALSE)
rownames(spot_meta) <- spot_meta$spot_id
spot_meta <- spot_meta[colnames(weights), , drop = FALSE]
pt <- paired_tests(weights, spot_meta)
plot_paired(pt$paired)
write_report(pt$tests, reference_se, spatial_spe, assay_name)

cat("DONE\n")
print(pt$tests)
