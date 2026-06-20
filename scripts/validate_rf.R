#!/usr/bin/env Rscript
# ===========================================================
# Quick RF Validation: Run RF on Harod loggers and compare
# with Python pipeline_v2 results
# ===========================================================

# --- Load package ---
suppressPackageStartupMessages({
  library(dplyr)
  library(ranger)
})

args <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("^--file=", args, value = TRUE)
if (length(file_flag) > 0) {
  pkg_root <- normalizePath(dirname(dirname(sub("^--file=", "", file_flag))))
} else {
  pkg_root <- normalizePath("..")
}
suppressMessages(devtools::load_all(pkg_root, quiet = TRUE))

PROJECT_ROOT <- normalizePath(file.path(pkg_root, ".."))
HAROD_PATH   <- file.path(PROJECT_ROOT, "data/experiments_data/Harod_dataset.csv")
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "microcl_ml_corr/outputs/phase_1")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Config matching Python pipeline
SEED <- 123
N_RUNS <- 5
TRAINING_DAYS <- c(1, 2, 3, 7, 14, 21, 28, 35, 42)

cat("Loading Harod Valley data...\n")
harod_data <- load_prepared_csv_data(
  HAROD_PATH,
  is_continuous_microhabitat = FALSE,
  datetime_format = "%d/%m/%Y %H:%M",
  includes_index = TRUE
)
cat(sprintf("Total rows: %d\n", nrow(harod_data)))

harod_sites <- unique(harod_data$time_series_doc)
cat(sprintf("Loggers: %s\n\n", paste(harod_sites, collapse = ", ")))

all_results <- list()

for (site in harod_sites) {
  cat(sprintf("\n=== Processing: %s ===\n", site))
  site_data <- harod_data[harod_data$time_series_doc == site, , drop = FALSE]
  cat(sprintf("  Rows: %d\n", nrow(site_data)))

  # Split using Python-matched block assignments
  splits <- split_train_val_test(
    site_data, train_pct = 0.75, val_pct = 0.125,
    block_days = 7, use_blocks = TRUE, seed = SEED,
    train_blocks = c(0, 1, 2, 3, 4, 7),
    val_blocks = c(5),
    test_blocks = c(6)
  )
  cat(sprintf("  Train: %d, Val: %d, Test: %d\n",
              nrow(splits$train), nrow(splits$val), nrow(splits$test)))

  feature_cols <- get_feature_columns(splits$train)
  cat(sprintf("  Features (%d): %s\n", length(feature_cols),
              paste(feature_cols, collapse = ", ")))

  # RF HPO on full data
  rf_train_X <- splits$train[, feature_cols, drop = FALSE]
  rf_train_y <- splits$train$residual
  rf_val_X   <- splits$val[, feature_cols, drop = FALSE]
  rf_val_y   <- splits$val$residual

  rf_full <- train_rf(rf_train_X, rf_train_y,
                      tune = TRUE, n_combinations = 5,
                      val_X = rf_val_X, val_y = rf_val_y,
                      seed = SEED)

  rf_params <- list(
    max.depth = rf_full$max.depth,
    min.node.size = rf_full$min.node.size,
    mtry = rf_full$mtry
  )
  cat(sprintf("  RF HPO: max.depth=%s, min.node.size=%d, mtry=%d\n",
              ifelse(is.null(rf_params$max.depth), "NULL",
                     as.character(rf_params$max.depth)),
              rf_params$min.node.size, rf_params$mtry))

  # Test data
  rf_test_X <- splits$test[, feature_cols, drop = FALSE]
  rf_test_y <- splits$test$residual
  rf_test_base <- splits$test$predicted

  # Train at each data size
  for (n_days in TRAINING_DAYS) {
    n_hours <- n_days * 24
    ts_sites <- unique(splits$train$time_series_doc)
    partial_rows <- list()
    for (ts in ts_sites) {
      ts_data <- splits$train[splits$train$time_series_doc == ts, , drop = FALSE]
      ts_data <- ts_data[order(ts_data$time), , drop = FALSE]
      k <- min(nrow(ts_data), n_hours)
      partial_rows[[length(partial_rows) + 1]] <- ts_data[seq_len(k), , drop = FALSE]
    }
    rf_partial_train <- do.call(rbind, partial_rows)
    train_size <- nrow(rf_partial_train)
    if (train_size == 0) next

    rf_partial_X <- rf_partial_train[, feature_cols, drop = FALSE]
    rf_partial_y <- rf_partial_train$residual

    for (run_id in 0:(N_RUNS - 1)) {
      rf_model <- ranger::ranger(
        x = rf_partial_X, y = rf_partial_y,
        num.trees = 500,
        max.depth = rf_params$max.depth,
        min.node.size = rf_params$min.node.size,
        mtry = rf_params$mtry,
        seed = run_id
      )

      metrics <- evaluate_correction(rf_model, rf_test_X, rf_test_y,
                                     rf_test_base, model_type = "rf")

      all_results[[length(all_results) + 1]] <- data.frame(
        model = "RF",
        window_size = NA,
        perc = n_days,
        train_size = train_size,
        ts_name = "ALL",
        rmse_corr = metrics$rmse_corr,
        rmse_base = metrics$rmse_base,
        run = run_id,
        site = gsub("\\.csv$", "", site),
        pipeline = "R",
        stringsAsFactors = FALSE
      )
    }

    mean_rmse <- mean(sapply(tail(all_results, N_RUNS), function(x) x$rmse_corr))
    cat(sprintf("  RF %2d days (%4d pts): RMSE_corr=%.3f (base=%.3f)\n",
                n_days, train_size, mean_rmse, metrics$rmse_base))
  }

  # Save per-site results
  safe_name <- gsub("[/ .]+", "_", gsub("\\.csv$", "", site))
  site_dir <- file.path(OUTPUT_DIR, safe_name)
  dir.create(site_dir, recursive = TRUE, showWarnings = FALSE)
  site_results <- do.call(rbind, Filter(function(x) x$site == gsub("\\.csv$", "", site),
                                         all_results))
  write.csv(site_results, file.path(site_dir, "model_performance_results.csv"),
            row.names = FALSE)
}

# ===========================================================
# COMPARISON WITH PYTHON
# ===========================================================
cat("\n\n============================================\n")
cat("R vs PYTHON COMPARISON (RF only)\n")
cat("============================================\n\n")

r_df <- do.call(rbind, all_results)

# Load Python results for matching sites
py_results_dir <- file.path(PROJECT_ROOT, "pipeline_v2/outputs/phase_1/replicates")
py_all <- list()
for (site_name in unique(r_df$site)) {
  py_path <- file.path(py_results_dir, site_name, "model_performance_results.csv")
  if (file.exists(py_path)) {
    py_df <- read.csv(py_path, stringsAsFactors = FALSE)
    py_df <- py_df[py_df$model == "RF" & py_df$ts_name == "ALL", ]
    py_df$site <- site_name
    py_df$pipeline <- "Python"
    py_all[[length(py_all) + 1]] <- py_df
  }
}

if (length(py_all) > 0) {
  py_combined <- do.call(rbind, py_all)

  # Align columns
  common_cols <- c("model", "perc", "train_size", "ts_name", "rmse_corr",
                   "rmse_base", "run", "site", "pipeline")
  r_compare <- r_df[, common_cols]
  py_compare <- py_combined[, common_cols]
  combined <- rbind(r_compare, py_compare)

  # Summary by site, days
  cat("Per-site RF RMSE comparison (mean ± sd across runs):\n")
  cat("─────────────────────────────────────────────────────\n")

  summary_table <- combined %>%
    group_by(site, perc, pipeline) %>%
    summarize(
      mean_rmse = mean(rmse_corr),
      sd_rmse = sd(rmse_corr),
      base_rmse = mean(rmse_base),
      .groups = "drop"
    ) %>%
    arrange(site, perc, pipeline)

  # Print nicely
  for (s in unique(summary_table$site)) {
    cat(sprintf("\n--- %s ---\n", s))
    cat(sprintf("%-10s %-8s  %-20s %-20s  %-10s\n",
                "Days", "Pipeline", "RMSE_corr", "RMSE_base", "Δ"))
    site_tbl <- summary_table[summary_table$site == s, ]

    for (d in unique(site_tbl$perc)) {
      day_rows <- site_tbl[site_tbl$perc == d, ]
      for (j in seq_len(nrow(day_rows))) {
        row <- day_rows[j, ]
        cat(sprintf("%-10d %-8s  %.3f ± %.3f         %.3f       %.1f%%\n",
                    row$perc, row$pipeline,
                    row$mean_rmse, row$sd_rmse, row$base_rmse,
                    (1 - row$mean_rmse / row$base_rmse) * 100))
      }
    }
  }

  # Save comparison
  write.csv(summary_table, file.path(OUTPUT_DIR, "rf_r_vs_python_comparison.csv"),
            row.names = FALSE)
  cat(sprintf("\nComparison saved to: %s\n",
              file.path(OUTPUT_DIR, "rf_r_vs_python_comparison.csv")))
} else {
  cat("No matching Python results found.\n")
}

cat("\n=== Phase 1 RF Validation Complete ===\n")
