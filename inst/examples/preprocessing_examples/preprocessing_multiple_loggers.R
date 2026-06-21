# preprocessing_multiple_loggers.R
#
# Step-by-step data preparation for MULTIPLE loggers (pooled workflow).
# Corresponds to the right panel of Figure 1 in the manuscript.
#
# Input files (in data/):
#   data/example_loggers/             — folder with one CSV per field logger
#   data/example_nichemapr_multiple.csv — NicheMapR predictions for all sites (one file)
#
# Output:
#   data/aligned_pooled.csv           — stacked merged file ready for microclCorr

library(microclCorr)

# ── 1. Load NicheMapR predictions (one file for all loggers) ───────────────────

script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
nm_path    <- file.path(script_dir, "data", "example_nichemapr_multiple.csv")
nm <- read.csv(nm_path)
nm$time <- as.POSIXct(nm$time, tz = "UTC")
cat("NicheMapR rows:", nrow(nm), "| sites:", length(unique(nm$site_id)), "\n")

# ── 2. Load and process each logger CSV ────────────────────────────────────────

logger_dir   <- file.path(script_dir, "data", "example_loggers")
logger_files <- list.files(logger_dir, pattern = "\\.csv$", full.names = TRUE)
cat("Found", length(logger_files), "logger files\n")

aligned_list <- lapply(logger_files, function(f) {

  logger <- read.csv(f)
  logger$time <- as.POSIXct(logger$time, tz = "UTC")

  # Step 1: align this logger to its NicheMapR predictions
  site <- unique(logger$site_id)
  nm_site <- nm[nm$site_id == site, ]
  merged <- merge(logger, nm_site, by = c("time", "site_id"), all = FALSE)

  # Step 2: compute residual = measured - predicted
  merged$residual <- merged$shade_temp - merged$predicted

  # Step 3: microhabitat column is already present ('microhabitat')
  #         Add it explicitly if your data does not have it:
  # merged$microhabitat <- "shade"   # or "sun", "rock", etc.

  # Step 4: site_id column already present (used as-is)

  cat("  Site:", site, "| rows:", nrow(merged), "\n")
  merged
})

# ── 3. Step 5: stack all loggers into one file ─────────────────────────────────

aligned_pooled <- do.call(rbind, aligned_list)
cat("Total pooled rows:", nrow(aligned_pooled),
    "| sites:", length(unique(aligned_pooled$site_id)), "\n")

# ── 4. Save pooled file ─────────────────────────────────────────────────────────

out_path <- file.path(script_dir, "data", "aligned_pooled.csv")
write.csv(aligned_pooled, out_path, row.names = FALSE)
cat("Pooled CSV saved to:", out_path, "\n")

# ── 5. Verify with microclCorr loader ──────────────────────────────────────────

data <- load_prepared_csv_data(out_path)
cat("Loaded by microclCorr — rows:", nrow(data), "| columns:", ncol(data), "\n")
cat("Sites in loaded data:", paste(unique(data$site_id), collapse = ", "), "\n")
