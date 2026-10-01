# preprocessing_single_logger.R
#
# Step-by-step data preparation for a SINGLE logger.
# Corresponds to the left panel of Figure 1 in the manuscript.
#
# Input files (in data/):
#   data/example_logger_single.csv    — measured temperatures from one field logger
#   data/example_nichemapr_single.csv — NicheMapR model predictions for the same site
#
# Output:
#   data/aligned_single.csv           — merged file ready for microclCorr

library(microclCorr)

# ── 1. Load the two raw files ──────────────────────────────────────────────────

data_dir <- system.file("examples", "preprocessing_examples", "data", package = "microclCorr")
if (!nzchar(data_dir) || !dir.exists(data_dir)) {
  data_dir <- if (dir.exists("data")) "data" else file.path("inst", "examples", "preprocessing_examples", "data")
}

logger_path <- file.path(data_dir, "example_logger_single.csv")
nm_path     <- file.path(data_dir, "example_nichemapr_single.csv")

logger <- read.csv(logger_path)
nm     <- read.csv(nm_path)

cat("Logger rows:", nrow(logger), "| columns:", paste(names(logger), collapse = ", "), "\n")
cat("NicheMapR rows:", nrow(nm),  "| columns:", paste(names(nm),     collapse = ", "), "\n")

# ── 2. Parse datetime in both files ────────────────────────────────────────────

logger$time <- as.POSIXct(logger$time, tz = "UTC")
nm$time     <- as.POSIXct(nm$time,     tz = "UTC")

# ── 3. Join on timestamp ────────────────────────────────────────────────────────

aligned <- merge(logger, nm, by = c("time"), all = FALSE)
cat("Aligned rows after join:", nrow(aligned), "\n")

# ── 4. Compute residual = measured - predicted ──────────────────────────────────
# The measured temperature is in the 'shade_temp' column for this logger.

aligned$residual <- aligned$shade_temp - aligned$predicted

cat("Residual summary:\n")
print(summary(aligned$residual))

# ── 5. Save aligned file ────────────────────────────────────────────────────────

out_dir  <- if (dir.exists(data_dir) && file.access(data_dir, 2) == 0) data_dir else getwd()
out_path <- file.path(out_dir, "aligned_single.csv")
write.csv(aligned, out_path, row.names = FALSE)
cat("Aligned CSV saved to:", out_path, "\n")

# ── 6. Verify with microclCorr loader ──────────────────────────────────────────
# Option A: Prepare directly in-memory using prepare_dataframe() (recommended)
prep_df <- prepare_dataframe(aligned)
cat("Prepared in-memory — rows:", nrow(prep_df), "| columns:", ncol(prep_df), "\n")

# Option B: Or load from saved CSV using load_prepared_csv_data()
loaded_df <- load_prepared_csv_data(out_path, includes_index = FALSE)
cat("Loaded from CSV    — rows:", nrow(loaded_df), "| columns:", ncol(loaded_df), "\n")

