# =============================================================================
# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models
# =============================================================================
# Goal: Train a SEPARATE model for each coastal location (Ashkelon, Range_24,
#       Rosh_HaNikra) and test whether this beats the single pooled model
#       from Scenario 4.
#
# The same pipeline is run three times — once per location. Each model only
# sees data from its own location during training, so it can specialise to
# local conditions (wind patterns, sea proximity, terrain).
#
# Compare with: Scenario 2 (single logger), Scenario 4 (all sites pooled)
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED        <- 42
SITE_COL    <- "time_series_site"

DATA_PATH   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH <- system.file("extdata", "beach_splits.csv",            package = "microclCorr")
RESULTS_DIR <- file.path("inst", "examples", "scenario_5_beach_specialized", "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 5: Beach Specialized ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
# Same loading step as Scenario 4 — all beach loggers in one file.
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL

# ── Step 2: Split into train / validation / test ───────────────────────────────
# Use the SAME pre-defined time blocks as Scenario 4 so that both scenarios
# are evaluated on identical test rows — enabling a fair comparison.
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Steps 4–7: Run the full pipeline once per location ────────────────────────
# For each coastal location we subset the data, train both models, and evaluate.
results <- list()

for (loc in c("Ashkelon", "Range_24", "Rosh_HaNikra")) {
  cat(sprintf("\n── Location: %s ──\n", loc))

  # Subset to this location only
  train_loc <- splits$train[splits$train$location == loc, ]
  val_loc   <- splits$val  [splits$val$location   == loc, ]
  test_loc  <- splits$test [splits$test$location  == loc, ]
  cat(sprintf("Train: %d | Validation: %d | Test: %d rows\n",
              nrow(train_loc), nrow(val_loc), nrow(test_loc)))

  # Step 4: Train a Random Forest on this location's data only
  rf_model <- train_rf(train_loc[, feature_cols], train_loc$residual, seed = SEED)

  # Step 5: Train an LSTM on this location's data only
  scaled    <- lstm_scaling(train_loc, val_loc, test_loc)
  lstm_data <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                            window_size  = 2,
                                            ts_names_col = SITE_COL)
  lstm_model <- train_lstm(lstm_data$train_dict$X, lstm_data$train_dict$y,
                            lstm_data$val_dict$X,   lstm_data$val_dict$y,
                            n_units = 32, n_layers = 1, dropout = 0.0, lr = 0.005,
                            epochs = 40, batch_size = 128, patience = 5, seed = SEED)

  # Step 6: Align test sets (keep only rows where the LSTM made a prediction)
  rf_test <- align_test_sets(test_loc, lstm_data$test_dict, lstm_data$index_info, SITE_COL)

  # Step 7: Evaluate, recording results per logger site within this location
  for (site in unique(rf_test[[SITE_COL]])) {
    mask <- rf_test[[SITE_COL]] == site
    m    <- evaluate_correction(rf_model, rf_test[mask, feature_cols],
                                 rf_test$residual[mask], rf_test$predicted[mask],
                                 model_type = "rf")
    results[[length(results) + 1]] <- c(results_row("RF", site, m), list(location = loc))
  }

  for (i in seq_along(lstm_data$index_info$datasets)) {
    site <- lstm_data$index_info$datasets[i]
    idx  <- lstm_data$index_info$test_indices[[i]] + 1
    m    <- evaluate_correction(lstm_model,
                                 lstm_data$test_dict$X[idx, , , drop = FALSE],
                                 lstm_data$test_dict$y[idx],
                                 lstm_data$test_dict$base_pred[idx],
                                 model_type = "lstm")
    results[[length(results) + 1]] <- c(results_row("LSTM_2h", site, m), list(location = loc))
  }

  # Step 8: Save this location's model so it can be applied to new data later
  save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR, paste0("rf_", loc, "_model.rds")))
}

results_df <- do.call(rbind, lapply(results, as.data.frame))

# ── Save results ──────────────────────────────────────────────────────────────
write.csv(results_df,
          file.path(RESULTS_DIR, "beach_specialized_results.csv"),
          row.names = FALSE)

cat("\nAverage performance by location:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ location + model,
                results_df, mean))
cat("=== Scenario 5 complete ===\n")
