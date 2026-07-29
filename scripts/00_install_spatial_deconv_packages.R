suppressPackageStartupMessages({
  library(BiocManager)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grepl("^--file=", args)]
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
PROJECT <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
LOG_DIR <- file.path(PROJECT, "logs")
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(LOG_DIR, "install_spatial_deconv_packages_20260601.log")
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")
on.exit({
  sink(type = "message")
  sink()
  close(log_con)
}, add = TRUE)

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  timeout = 1200,
  Ncpus = max(1L, parallel::detectCores(logical = TRUE) - 1L)
)

cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("R:", R.version.string, "\n")
cat("Bioconductor:", as.character(BiocManager::version()), "\n")
cat("Library paths:\n")
cat(paste(.libPaths(), collapse = "\n"), "\n\n")

cran_pkgs <- c(
  "nnls",
  "NMF",
  "reshape2",
  "scatterpie",
  "RANN",
  "fields",
  "remotes",
  "devtools"
)

bioc_pkgs <- c(
  "SingleCellExperiment",
  "SummarizedExperiment",
  "SpatialExperiment",
  "scater",
  "scran",
  "spacexr",
  "SPOTlight",
  "CARDspa"
)

install_one_cran <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("[OK] CRAN", pkg, "already installed\n")
    return(TRUE)
  }
  cat("[INSTALL] CRAN", pkg, "\n")
  ok <- tryCatch({
    install.packages(pkg, dependencies = TRUE)
    requireNamespace(pkg, quietly = TRUE)
  }, error = function(e) {
    cat("[ERROR] CRAN", pkg, conditionMessage(e), "\n")
    FALSE
  })
  cat(if (ok) "[DONE]" else "[FAILED]", "CRAN", pkg, "\n")
  ok
}

install_one_bioc <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("[OK] Bioc", pkg, "already installed\n")
    return(TRUE)
  }
  cat("[INSTALL] Bioc", pkg, "\n")
  ok <- tryCatch({
    BiocManager::install(pkg, ask = FALSE, update = FALSE, dependencies = TRUE)
    requireNamespace(pkg, quietly = TRUE)
  }, error = function(e) {
    cat("[ERROR] Bioc", pkg, conditionMessage(e), "\n")
    FALSE
  })
  cat(if (ok) "[DONE]" else "[FAILED]", "Bioc", pkg, "\n")
  ok
}

results <- data.frame(
  package = character(),
  source = character(),
  installed = logical(),
  version = character(),
  stringsAsFactors = FALSE
)

for (pkg in cran_pkgs) {
  ok <- install_one_cran(pkg)
  ver <- if (requireNamespace(pkg, quietly = TRUE)) as.character(utils::packageVersion(pkg)) else ""
  results <- rbind(results, data.frame(package = pkg, source = "CRAN", installed = ok, version = ver))
}

for (pkg in bioc_pkgs) {
  ok <- install_one_bioc(pkg)
  ver <- if (requireNamespace(pkg, quietly = TRUE)) as.character(utils::packageVersion(pkg)) else ""
  results <- rbind(results, data.frame(package = pkg, source = "Bioconductor", installed = ok, version = ver))
}

out_csv <- file.path(LOG_DIR, "install_spatial_deconv_packages_20260601_summary.csv")
write.csv(results, out_csv, row.names = FALSE)

cat("\nSummary:\n")
print(results)
cat("\nSummary CSV:", out_csv, "\n")
cat("Log file:", log_file, "\n")

if (!all(results$installed)) {
  quit(status = 2)
}
