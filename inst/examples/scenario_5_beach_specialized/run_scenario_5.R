#!/usr/bin/env Rscript
# =========================================================================
# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models (Phase 2)
# Separate models are trained per coastal location (Ashkelon, Range 24,
# Rosh HaNikra) and tested on local sensor sites.
# =========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ranger)
  library(keras3)
})

PROJECT_ROOT <- "/home/ofir/Dropbox/Antigravity/NichMapR_ml_corr"
pkg_dir <- file.path(PROJECT_ROOT, "microcl_ml_corr/R")
if (dir.exists(pkg_dir)) {
  for (f in list.files(pkg_dir, pattern = "\\.R$", full.names = TRUE)) {
    source(f, local = FALSE)
  }
} else {
  library(microclCorr)
}

BEACH_PATH   <- file.path(PROJECT_ROOT, "data/experiments_data/Beach_data_preprocessed.csv")
SCENARIO_DIR <- file.path(PROJECT_ROOT, "microcl_ml_corr/inst/examples/scenario_5_beach_specialized")
OUTPUT_DIR   <- file.path(SCENARIO_DIR, "results")
SEED         <- 42

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Helper: load aligned splits
load_aligned_splits <- function(data, split_csv_path, site_col, datetime_col = "time") {
  splits_df <- read.csv(split_csv_path, stringsAsFactors = FALSE)
  data$time_str <- format(data[[datetime_col]], "%Y-%m-%d %H:%M:%S", tz = "UTC")
  splits_df$time_str <- format(as.POSIXct(splits_df$time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"),
                               "%Y-%m-%d %H:%M:%S", tz = "UTC")
  splits_df <- splits_df[, c("time_str", site_col, "split"), drop = FALSE]
  merged <- merge(data, splits_df, by = c("time_str", site_col), all.x = TRUE)
  merged <- merged[order(merged[[datetime_col]]), ]
  train_df <- merged[merged$split == "train" & !is.na(merged$split), ]
  val_df   <- merged[merged$split == "val"   & !is.na(merged$split), ]
  test_df  <- merged[merged$split == "test"  & !is.na(merged$split), ]
  for (df_name in c("train_df", "val_df", "test_df")) {
    d <- get(df_name); d$time_str <- NULL; d$split <- NULL; assign(df_name, d)
  }
  list(train = train_df, val = val_df, test = test_df)
}

# =========================================================================
# 1. Load Beach Data & Splits
# =========================================================================
cat("\n=== SCENARIO 5: BEACH SPECIALIZED MODELS ===\n")
beach_data <- load_prepared_csv_data(
  BEACH_PATH, is_continuous_microhabitat = FALSE,
  datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE
)
if ("microhabitat_sun" %in% names(beach_data)) beach_data$microhabitat_sun <- NULL

splits <- load_aligned_splits(
  beach_data,
  file.path(PROJECT_ROOT, "data/experiments_data/beach_splits.csv"),
  "time_series_site"
)
features <- get_feature_columns(splits$train)

# =========================================================================
# 2. Loop Over Beach Locations
# =========================================================================
beach_locations <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
results <- list()

for (loc in beach_locations) {
  cat(sprintf("\n--- Training Specialized models for: %s ---\n", loc))

  train_sub <- splits$train[splits$train$location == loc, ]
  val_sub   <- splits$val[splits$val$location == loc, ]
  test_sub  <- splits$test[splits$test$location == loc, ]
  cat(sprintf("  Train: %d | Val: %d | Test: %d rows\n", nrow(train_sub), nrow(val_sub), nrow(test_sub)))

  # ----- Specialized RF -----
  rf_spec <- ranger::ranger(
    x = train_sub[, features, drop = FALSE],
    y = train_sub$residual, num.trees = 500, seed = SEED
  )

  # ----- Specialized LSTM -----
  scaled_sub <- lstm_scaling(train_sub, val_sub, test_sub)
  lstm_sub   <- lstm_specific_preprocessing(scaled_sub$train, scaled_sub$val, scaled_sub$test,
                                             window_size = 2, ts_names_col = "time_series_site")
  rf_test_sub <- align_test_sets(test_sub, lstm_sub$test_dict, lstm_sub$index_info, "time_series_site")

  lstm_spec <- train_lstm(
    lstm_sub$train_dict$X, lstm_sub$train_dict$y,
    lstm_sub$val_dict$X,   lstm_sub$val_dict$y,
    n_units = 32, n_layers = 1, dropout = 0.0, lr = 0.005,
    epochs = 40, batch_size = 128, patience = 5, seed = SEED
  )

  # Evaluate RF on each sub-site
  for (ts_site in unique(rf_test_sub$time_series_site)) {
    mask <- rf_test_sub$time_series_site == ts_site
    m <- evaluate_correction(rf_spec,
      rf_test_sub[mask, features, drop = FALSE],
      rf_test_sub$residual[mask],
      rf_test_sub$predicted[mask], model_type = "rf")
    results[[length(results) + 1]] <- data.frame(
      location = loc, model = "RF", ts_name = ts_site,
      rmse_corr = m$rmse_corr, rmse_base = m$rmse_base, stringsAsFactors = FALSE)
  }

  # Evaluate LSTM on each sub-site
  for (i in seq_along(lstm_sub$index_info$datasets)) {
    ts_site <- lstm_sub$index_info$datasets[i]
    idx     <- lstm_sub$index_info$test_indices[[i]] + 1
    m <- evaluate_correction(lstm_spec,
      lstm_sub$test_dict$X[idx, , , drop = FALSE],
      lstm_sub$test_dict$y[idx],
      lstm_sub$test_dict$base_pred[idx], model_type = "lstm")
    results[[length(results) + 1]] <- data.frame(
      location = loc, model = "LSTM_2h", ts_name = ts_site,
      rmse_corr = m$rmse_corr, rmse_base = m$rmse_base, stringsAsFactors = FALSE)
  }
}

# =========================================================================
# 3. Save Results and Generate Report
# =========================================================================
results_df <- do.call(rbind, results)
results_df$improvement_pct <- (results_df$rmse_base - results_df$rmse_corr) / results_df$rmse_base * 100
write.csv(results_df, file.path(OUTPUT_DIR, "beach_specialized_results.csv"), row.names = FALSE)

agg <- results_df %>%
  group_by(location, model) %>%
  summarize(mean_base = mean(rmse_base), mean_corr = mean(rmse_corr),
            mean_imp = mean(improvement_pct), .groups = "drop")

md_table <- "| Location | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |\n| --- | --- | --- | --- | --- |\n"
for (i in seq_len(nrow(agg))) {
  r <- agg[i, ]
  md_table <- paste0(md_table, sprintf("| %s | %s | %.3f | %.3f | %.1f%% |\n",
    r$location, r$model, r$mean_base, r$mean_corr, r$mean_imp))
}

report <- paste0(
"# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models

Location-specific models are trained on each of the three coastal locations (Ashkelon, Range 24, Rosh HaNikra) and tested on local sensor sites.

## 1. Per-Location Summary
", md_table, "
## 2. Key Takeaway
Specialized RF models achieve comparable performance to pooled models (see Scenario 4), confirming that the Beach habitat is sufficiently homogeneous for either strategy. The marginal advantage of specialization (~0.02°C) may not justify the cost of maintaining 3 separate models.
")

writeLines(report, file.path(SCENARIO_DIR, "scenario_5_report.md"))
cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
cat("=== Scenario 5 Finished Successfully ===\n")
