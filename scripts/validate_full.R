#!/usr/bin/env Rscript
# ===========================================================
# Full Phase 1 Validation: RF + LSTM 2h on harod2_sun.csv
# Uses Python-aligned block splits and 24h test alignment.
# Compares results side-by-side with Python pipeline_v2.
# ===========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ranger)
  library(keras3)
})

# Source package functions
pkg_dir <- "/home/ofir/Dropbox/Antigravity/NichMapR_ml_corr/microcl_ml_corr/R"
for (f in list.files(pkg_dir, pattern = "\\.R$", full.names = TRUE)) {
  source(f, local = FALSE)
}

PROJECT_ROOT <- "/home/ofir/Dropbox/Antigravity/NichMapR_ml_corr"
HAROD_PATH   <- file.path(PROJECT_ROOT, "data/experiments_data/Harod_dataset.csv")
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "microcl_ml_corr/outputs/phase_1_full")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

SEED <- 123
N_RUNS <- 5  # 5 runs to match Python replicates exactly
TRAINING_DAYS <- c(7, 21, 42)

cat("=== microclCorr Phase 1: RF + LSTM 2h Aligned Validation ===\n\n")

# Load data
harod_data <- load_prepared_csv_data(
  HAROD_PATH, is_continuous_microhabitat = FALSE,
  datetime_format = "%d/%m/%Y %H:%M", includes_index = TRUE
)

site <- "harod2_sun.csv"
cat(sprintf("Processing: %s\n", site))
site_data <- harod_data[harod_data$time_series_doc == site, , drop = FALSE]
site_name_col <- "time_series_doc"

# Split using exact Python blocks
splits <- split_train_val_test(
  site_data, train_pct = 0.75, val_pct = 0.125,
  block_days = 7, use_blocks = TRUE, seed = SEED,
  train_blocks = c(0, 1, 2, 3, 4, 7),
  val_blocks = c(5),
  test_blocks = c(6)
)
cat(sprintf("Split sizes: train=%d, val=%d, test=%d\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# Scale
scaled <- lstm_scaling(splits$train, splits$val, splits$test)
feature_cols <- get_feature_columns(splits$train)

# Pre-generate 24h window test set for test alignment
lstm_data_24h <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size = 24, ts_names_col = site_name_col
)

all_results <- list()

# ===========================================================
# RF PIPELINE
# ===========================================================
cat("\n--- Random Forest ---\n")

# Align RF test set to 24h window endpoints (first 23 hours dropped)
rf_test_dataset_aligned <- align_test_sets(
  splits$test, lstm_data_24h$test_dict,
  lstm_data_24h$index_info, site_name_col
)

rf_train_X <- splits$train[, feature_cols, drop = FALSE]
rf_train_y <- splits$train$residual
rf_val_X   <- splits$val[, feature_cols, drop = FALSE]
rf_val_y   <- splits$val$residual

rf_test_X  <- rf_test_dataset_aligned[, feature_cols, drop = FALSE]
rf_test_y  <- rf_test_dataset_aligned$residual
rf_test_base <- rf_test_dataset_aligned$predicted

# HPO
rf_full <- train_rf(rf_train_X, rf_train_y, tune = TRUE,
                    n_combinations = 5, val_X = rf_val_X, val_y = rf_val_y,
                    seed = SEED)

rf_params <- list(max.depth = rf_full$max.depth,
                  min.node.size = rf_full$min.node.size,
                  mtry = rf_full$mtry)

for (n_days in TRAINING_DAYS) {
  n_hours <- n_days * 24
  ts_sites <- unique(splits$train[[site_name_col]])
  partial_rows <- list()
  for (ts in ts_sites) {
    ts_d <- splits$train[splits$train[[site_name_col]] == ts, , drop = FALSE]
    ts_d <- ts_d[order(ts_d$time), , drop = FALSE]
    k <- min(nrow(ts_d), n_hours)
    partial_rows[[length(partial_rows) + 1]] <- ts_d[seq_len(k), , drop = FALSE]
  }
  rf_partial <- do.call(rbind, partial_rows)
  train_size <- nrow(rf_partial)

  for (run_id in 0:(N_RUNS - 1)) {
    rf_m <- ranger::ranger(
      x = rf_partial[, feature_cols, drop = FALSE],
      y = rf_partial$residual,
      num.trees = 500, max.depth = rf_params$max.depth,
      min.node.size = rf_params$min.node.size,
      mtry = rf_params$mtry, seed = run_id
    )
    metrics <- evaluate_correction(rf_m, rf_test_X, rf_test_y,
                                   rf_test_base, model_type = "rf")
    all_results[[length(all_results) + 1]] <- data.frame(
      model = "RF", perc = n_days, train_size = train_size,
      rmse_corr = metrics$rmse_corr, rmse_base = metrics$rmse_base,
      run = run_id, stringsAsFactors = FALSE
    )
  }
  mr <- mean(sapply(tail(all_results, N_RUNS), function(x) x$rmse_corr))
  cat(sprintf("  RF %2d days: RMSE_corr=%.3f (base=%.3f)\n",
              n_days, mr, metrics$rmse_base))
}

# ===========================================================
# LSTM 2h PIPELINE
# ===========================================================
cat("\n--- LSTM 2h ---\n")

ws <- 2  # 2-hour window

# Generate 2h windowed datasets for training
lstm_data_2h <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size = ws, ts_names_col = site_name_col
)

# Align LSTM 2h test inputs to the 24h window endpoints (taking last 2 steps of 24h sequences)
X_test_aligned <- lstm_data_24h$test_dict$X[, (24 - ws + 1):24, , drop = FALSE]
y_test_aligned <- lstm_data_24h$test_dict$y
base_test_aligned <- lstm_data_24h$test_dict$base_pred

# Run HPO (3 trials for speed)
cat("  Running LSTM HPO (3 trials)...\n")
hpo <- lstm_hypertuning(
  lstm_data_2h$train_dict$X, lstm_data_2h$train_dict$y,
  lstm_data_2h$val_dict$X, lstm_data_2h$val_dict$y,
  n_trials = 3, epochs = 100,
  batch_size = 32, patience = 10,
  seed = SEED
)

bp <- hpo$params
cat(sprintf("  Best: units=%d, layers=%d, dropout=%.2f, lr=%.6f\n",
            bp$n_units, bp$n_layers, bp$dropout, bp$lr))

for (n_days in TRAINING_DAYS) {
  n_hours <- n_days * 24
  total_train <- length(lstm_data_2h$train_dict$y)
  k <- min(total_train, n_hours)
  if (k == 0) next

  train_idx <- seq_len(k)
  X_partial <- lstm_data_2h$train_dict$X[train_idx, , , drop = FALSE]
  y_partial <- lstm_data_2h$train_dict$y[train_idx]

  for (run_id in 0:(N_RUNS - 1)) {
    model <- tryCatch({
      train_lstm(
        X_partial, y_partial,
        lstm_data_2h$val_dict$X, lstm_data_2h$val_dict$y,
        n_units = bp$n_units, n_layers = bp$n_layers,
        dropout = bp$dropout, lr = bp$lr,
        epochs = 100, batch_size = 32, patience = 10,
        seed = run_id
      )
    }, error = function(e) {
      cat(sprintf("    [ERROR]: %s\n", e$message))
      NULL
    })
    if (is.null(model)) next

    metrics <- evaluate_correction(
      model, X_test_aligned, y_test_aligned,
      base_test_aligned, model_type = "lstm"
    )
    all_results[[length(all_results) + 1]] <- data.frame(
      model = "LSTM_2h", perc = n_days, train_size = k,
      rmse_corr = metrics$rmse_corr, rmse_base = metrics$rmse_base,
      run = run_id, stringsAsFactors = FALSE
    )
  }
  mr <- mean(sapply(tail(all_results, N_RUNS), function(x) x$rmse_corr))
  cat(sprintf("  LSTM_2h %2d days: RMSE_corr=%.3f (base=%.3f)\n", n_days, mr, metrics$rmse_base))
}

# ===========================================================
# SAVE & COMPARE
# ===========================================================
results_df <- do.call(rbind, all_results)
write.csv(results_df, file.path(OUTPUT_DIR, "harod2_sun_results.csv"), row.names = FALSE)

cat("\n============================================\n")
cat("RESULTS SUMMARY (harod2_sun - Aligned splits)\n")
cat("============================================\n\n")

summary_df <- results_df %>%
  group_by(model, perc) %>%
  summarize(
    mean_rmse_corr = mean(rmse_corr),
    sd_rmse_corr = sd(rmse_corr),
    base_rmse = mean(rmse_base),
    .groups = "drop"
  ) %>%
  arrange(model, perc)

cat(sprintf("%-12s %-8s %-20s %-10s %-10s\n",
            "Model", "Days", "RMSE_corr", "Base", "Improv%"))
cat(paste(rep("─", 65), collapse = ""), "\n")
for (i in seq_len(nrow(summary_df))) {
  r <- summary_df[i, ]
  cat(sprintf("%-12s %-8d %.3f ± %.3f        %.3f     %.1f%%\n",
              r$model, r$perc, r$mean_rmse_corr, r$sd_rmse_corr,
              r$base_rmse, (1 - r$mean_rmse_corr / r$base_rmse) * 100))
}

# Load Python results for comparison
py_path <- file.path(PROJECT_ROOT,
  "pipeline_v2/outputs/phase_1/replicates/harod2_sun/model_performance_results.csv")
if (file.exists(py_path)) {
  py_df <- read.csv(py_path, stringsAsFactors = FALSE)
  py_df <- py_df[py_df$ts_name == "ALL", ]
  py_rf <- py_df[py_df$model == "RF", ]
  py_lstm <- py_df[py_df$model == "LSTM_2h", ]

  cat("\n\nPYTHON PIPELINE RESULTS (for comparison):\n")
  cat(paste(rep("─", 65), collapse = ""), "\n")

  py_summary <- py_df %>%
    filter(model %in% c("RF", "LSTM_2h"), perc %in% TRAINING_DAYS) %>%
    group_by(model, perc) %>%
    summarize(
      mean_rmse_corr = mean(rmse_corr),
      sd_rmse_corr = sd(rmse_corr),
      base_rmse = mean(rmse_base),
      .groups = "drop"
    ) %>%
    arrange(model, perc)

  for (i in seq_len(nrow(py_summary))) {
    r <- py_summary[i, ]
    cat(sprintf("%-12s %-8d %.3f ± %.3f        %.3f     %.1f%%\n",
                r$model, r$perc, r$mean_rmse_corr, r$sd_rmse_corr,
                r$base_rmse, (1 - r$mean_rmse_corr / r$base_rmse) * 100))
  }
}

cat(sprintf("\nResults saved to: %s\n", file.path(OUTPUT_DIR, "harod2_sun_results.csv")))
cat("\n=== Phase 1 Full Validation Complete ===\n")
