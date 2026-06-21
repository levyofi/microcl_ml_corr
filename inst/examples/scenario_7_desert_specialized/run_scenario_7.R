# =============================================================================
# Scenario 7: Judean Desert — Specialized (Location-Specific) Models
# =============================================================================
# Goal: Train a SEPARATE model for each desert region (Mishmar, Tzeelim)
#       and test whether this beats the single pooled model from Scenario 6.
#
# Compare with: Scenario 3 (single logger), Scenario 6 (all sites pooled)
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED        <- 42
SITE_COL    <- "site_id"

DATA_PATH   <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH <- system.file("extdata", "desert_splits.csv",            package = "microclCorr")
RESULTS_DIR <- file.path("inst", "examples", "scenario_7_desert_specialized", "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 7: Desert Specialized ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
data <- load_prepared_csv_data(DATA_PATH,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)

# ── Step 2: Split into train / validation / test ───────────────────────────────
# Use the SAME pre-defined blocks as Scenario 6 for a fair comparison.
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Steps 4–7: Run the full pipeline once per region ──────────────────────────
results <- list()

for (region in c("Mishmar", "Tzeelim")) {
  cat(sprintf("\n── Region: %s ──\n", region))

  # Subset to this region only
  train_reg <- splits$train[splits$train$location == region, ]
  val_reg   <- splits$val  [splits$val$location   == region, ]
  test_reg  <- splits$test [splits$test$location  == region, ]
  cat(sprintf("Train: %d | Validation: %d | Test: %d rows\n",
              nrow(train_reg), nrow(val_reg), nrow(test_reg)))

  # Step 4: Train a Random Forest on this region's data only
  rf_model <- train_rf(train_reg[, feature_cols], train_reg$residual, seed = SEED)

  # Step 5: Train an LSTM on this region's data only
  scaled    <- lstm_scaling(train_reg, val_reg, test_reg)
  lstm_data <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                            window_size  = 2,
                                            ts_names_col = SITE_COL)
  lstm_model <- train_lstm(lstm_data$train_dict$X, lstm_data$train_dict$y,
                            lstm_data$val_dict$X,   lstm_data$val_dict$y,
                            n_units = 32, n_layers = 1, dropout = 0.0, lr = 0.005,
                            epochs = 20, batch_size = 256, patience = 5, seed = SEED)

  # Step 6: Align test sets
  rf_test <- align_test_sets(test_reg, lstm_data$test_dict, lstm_data$index_info, SITE_COL)

  # Step 7: Evaluate, recording results per logger site within this region
  for (site in unique(rf_test[[SITE_COL]])) {
    mask <- rf_test[[SITE_COL]] == site
    m    <- evaluate_correction(rf_model, rf_test[mask, feature_cols],
                                 rf_test$residual[mask], rf_test$predicted[mask],
                                 model_type = "rf")
    results[[length(results) + 1]] <- c(results_row("RF", site, m), list(region = region))
  }

  for (i in seq_along(lstm_data$index_info$datasets)) {
    site <- lstm_data$index_info$datasets[i]
    idx  <- lstm_data$index_info$test_indices[[i]] + 1
    m    <- evaluate_correction(lstm_model,
                                 lstm_data$test_dict$X[idx, , , drop = FALSE],
                                 lstm_data$test_dict$y[idx],
                                 lstm_data$test_dict$base_pred[idx],
                                 model_type = "lstm")
    results[[length(results) + 1]] <- c(results_row("LSTM_2h", site, m), list(region = region))
  }

  # Step 8: Save this region's model
  save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR, paste0("rf_", region, "_model.rds")))
}

results_df <- do.call(rbind, lapply(results, as.data.frame))
results_df$microhabitat <- ifelse(grepl("Bush", results_df$site), "Bush", "Rock")

# ── Save results ──────────────────────────────────────────────────────────────
write.csv(results_df,
          file.path(RESULTS_DIR, "desert_specialized_results.csv"),
          row.names = FALSE)

cat("\nAverage performance by region and microhabitat:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model + region + microhabitat,
                results_df, mean))
cat("=== Scenario 7 complete ===\n")
