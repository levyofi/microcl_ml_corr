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

script_dir  <- dirname(rstudioapi::getSourceEditorContext()$path)
logger_path <- file.path(script_dir, "data", "example_logger_single.csv")
nm_path     <- file.path(script_dir, "data", "example_nichemapr_single.csv")

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

out_path <- file.path(script_dir, "data", "aligned_single.csv")
write.csv(aligned, out_path, row.names = FALSE)
cat("Aligned CSV saved to:", out_path, "\n")

# ── 6. Verify with microclCorr loader ──────────────────────────────────────────

data <- load_prepared_csv_data(out_path)
cat("Loaded by microclCorr — rows:", nrow(data), "| columns:", ncol(data), "\n")
