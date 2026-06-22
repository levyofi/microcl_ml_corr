# =============================================================================
# Scenario 6: Judean Desert — Pooled Spatial Generalization
# =============================================================================
# Goal: Train ONE shared model on data from all 48 desert loggers combined,
#       then test how well it corrects predictions at each individual site.
#       Results are aggregated by region (Mishmar / Tzeelim) and microhabitat
#       type (Bush / Rock).
#
# Compare with: Scenario 3 (single logger), Scenario 7 (per-region models)
# Note: the pooled model uses far more training data than Scenario 3, so
#       the performance difference is not purely due to spatial diversity.
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED        <- 42
SITE_COL    <- "site_id"

DATA_PATH   <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH <- system.file("extdata", "desert_splits.csv",            package = "microclCorr")
RESULTS_DIR <- file.path("inst", "examples", "scenario_6_desert_pooled", "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 6: Desert Pooled ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
# All 48 desert loggers are stored in one pre-aligned CSV file.
data <- load_prepared_csv_data(DATA_PATH,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)

# ── Step 2: Split into train / validation / test ───────────────────────────────
# Use pre-defined time blocks shared with Scenario 7, ensuring that both
# scenarios are evaluated on the same test rows for a fair comparison.
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)
cat(sprintf("Train: %d | Validation: %d | Test: %d rows\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Step 4: Train a Random Forest ─────────────────────────────────────────────
# The RF is trained on all 48 sites at once — it will learn correction patterns
# that generalise across both regions and all microhabitat types.
rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual, seed = SEED)

# ── Step 5: Train an LSTM neural network ──────────────────────────────────────
# Normalise, then reshape the data into 2-hour windows per logger.
# ts_names_col ensures that windows do not bridge the gap between different loggers.
scaled    <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_data <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                          window_size  = 2,
                                          ts_names_col = SITE_COL)
lstm_model <- train_lstm(lstm_data$train_dict$X, lstm_data$train_dict$y,
                          lstm_data$val_dict$X,   lstm_data$val_dict$y,
                          n_units    = 32,
                          n_layers   = 1,
                          dropout    = 0.0,
                          lr         = 0.005,
                          epochs     = 20,     # fewer epochs than beach — desert patterns are simpler
                          batch_size = 256,    # larger batch to handle the bigger dataset efficiently
                          patience   = 10,
                          seed       = SEED)

# ── Step 6: Align test sets ───────────────────────────────────────────────────
# Filter the RF test set to only the time points where the LSTM also predicted,
# so that RMSE values from both models are computed on identical rows.
rf_test <- align_test_sets(splits$test, lstm_data$test_dict, lstm_data$index_info, SITE_COL)

# ── Step 7: Evaluate both models, site by site ────────────────────────────────
results <- list()

for (site in unique(rf_test[[SITE_COL]])) {
  mask <- rf_test[[SITE_COL]] == site
  m    <- evaluate_correction(rf_model, rf_test[mask, feature_cols],
                               rf_test$residual[mask], rf_test$predicted[mask],
                               model_type = "rf")
  results[[length(results) + 1]] <- results_row("RF", site, m)
}

for (i in seq_along(lstm_data$index_info$datasets)) {
  site <- lstm_data$index_info$datasets[i]
  idx  <- lstm_data$index_info$test_indices[[i]] + 1
  m    <- evaluate_correction(lstm_model,
                               lstm_data$test_dict$X[idx, , , drop = FALSE],
                               lstm_data$test_dict$y[idx],
                               lstm_data$test_dict$base_pred[idx],
                               model_type = "lstm")
  results[[length(results) + 1]] <- results_row("LSTM_2h", site, m)
}

results_df <- do.call(rbind, results)

# Derive region and microhabitat from the site ID naming convention (e.g. "Bush_M_T_1_W")
results_df$region       <- ifelse(grepl("_T_", results_df$site), "Tzeelim", "Mishmar")
results_df$microhabitat <- ifelse(grepl("Bush",  results_df$site), "Bush",    "Rock")

# ── Step 8: Save ──────────────────────────────────────────────────────────────
write.csv(results_df,
          file.path(RESULTS_DIR, "desert_pooled_results.csv"),
          row.names = FALSE)
save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "rf_pooled_model.rds"))

cat("\nAverage performance by region and microhabitat:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model + region + microhabitat,
                results_df, mean))
cat("=== Scenario 6 complete ===\n")
