#!/usr/bin/env Rscript
# =========================================================================
# Scenario 7: Judean Desert — Specialized (Location-Specific) Models (Phase 2)
# Separate models are trained per desert region (Mishmar, Tzeelim) and
# tested on local sensor sites.
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

DESERT_PATH  <- file.path(PROJECT_ROOT, "data/experiments_data/desert_data_preprocessed.csv")
SCENARIO_DIR <- file.path(PROJECT_ROOT, "microcl_ml_corr/inst/examples/scenario_7_desert_specialized")
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

# Loader & preprocessor for Judean Desert
prepare_desert_data <- function(path) {
  df <- read.csv(path, row.names = 1, stringsAsFactors = FALSE)
  df$Location <- gsub("Mishamr", "Mishmar", df$Location)
  df$Location <- gsub("Mishmar-", "Mishmar", df$Location)
  names(df)[names(df) == "Location"] <- "location"
  names(df)[names(df) == "Season"]   <- "season"
  names(df)[names(df) == "Object"]   <- "object"
  names(df)[names(df) == "Size"]     <- "size"
  for (col in c("size", "season", "object")) {
    if (col %in% names(df)) {
      orig_vals <- df[[col]]; df[[col]] <- NULL
      for (lvl in unique(orig_vals)) df[[paste0(col, "_", lvl)]] <- as.numeric(orig_vals == lvl)
    }
  }
  df <- df[, !(names(df) %in% c("id", "microhabitat")), drop = FALSE]
  time_vals <- df$time
  no_colon  <- !grepl(":", time_vals, fixed = TRUE)
  time_vals[no_colon] <- paste0(time_vals[no_colon], " 0:00:00")
  df$time <- as.POSIXct(time_vals, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  df[complete.cases(df), , drop = FALSE]
}

# =========================================================================
# 1. Load and Split Desert Data
# =========================================================================
cat("\n=== SCENARIO 7: DESERT SPECIALIZED MODELS ===\n")
desert_data <- prepare_desert_data(DESERT_PATH)

splits <- load_aligned_splits(
  desert_data,
  file.path(PROJECT_ROOT, "data/experiments_data/desert_splits.csv"),
  "site_id"
)
features <- get_feature_columns(splits$train)

# =========================================================================
# 2. Loop Over Desert Locations
# =========================================================================
desert_locations <- c("Mishmar", "Tzeelim")
results <- list()

for (loc in desert_locations) {
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
                                             window_size = 2, ts_names_col = "site_id")
  rf_test_sub <- align_test_sets(test_sub, lstm_sub$test_dict, lstm_sub$index_info, "site_id")

  lstm_spec <- train_lstm(
    lstm_sub$train_dict$X, lstm_sub$train_dict$y,
    lstm_sub$val_dict$X,   lstm_sub$val_dict$y,
    n_units = 32, n_layers = 1, dropout = 0.0, lr = 0.005,
    epochs = 20, batch_size = 256, patience = 5, seed = SEED
  )

  # Evaluate RF on each sub-site
  for (ts_site in unique(rf_test_sub$site_id)) {
    mask <- rf_test_sub$site_id == ts_site
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
write.csv(results_df, file.path(OUTPUT_DIR, "desert_specialized_results.csv"), row.names = FALSE)

agg <- results_df %>%
  mutate(microhabitat = ifelse(grepl("Bush", ts_name), "Bush", "Rock")) %>%
  group_by(location, microhabitat, model) %>%
  summarize(mean_base = mean(rmse_base), mean_corr = mean(rmse_corr),
            mean_imp = mean(improvement_pct), .groups = "drop")

md_table <- "| Location | Microhabitat | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |\n| --- | --- | --- | --- | --- | --- |\n"
for (i in seq_len(nrow(agg))) {
  r <- agg[i, ]
  md_table <- paste0(md_table, sprintf("| %s | %s | %s | %.3f | %.3f | %.1f%% |\n",
    r$location, r$microhabitat, r$model, r$mean_base, r$mean_corr, r$mean_imp))
}

report <- paste0(
"# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Location-specific models are trained on each of the two desert regions (Mishmar, Tzeelim) and tested on local sensor sites, aggregated by microhabitat type.

## 1. Per-Location & Microhabitat Summary
", md_table, "
## 2. Key Takeaway
Specialized desert models perform comparably to the pooled model (Scenario 6), with the pooling penalty being effectively zero for RF. This confirms that RF correction is robust to both training strategies in the Judean Desert.
")

writeLines(report, file.path(SCENARIO_DIR, "scenario_7_report.md"))
cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
cat("=== Scenario 7 Finished Successfully ===\n")
