# =============================================================================
# Scenario 4: Beach Habitat — Pooled Spatial Generalization
# =============================================================================
# Goal: Train ONE shared model on data from ALL 7 beach loggers combined,
#       then test how well it corrects predictions at each individual site.
#
# Background — what this pipeline does:
#   NicheMapR is a physical model that predicts local microclimate temperatures.
#   It is not perfect: there is always a gap between its prediction and what a
#   temperature logger actually measures. This gap is called the "residual":
#
#       residual = measured temperature − NicheMapR prediction
#
#   The microclCorr package trains a machine learning model to predict that
#   residual. At any new time point, the corrected temperature is then:
#
#       corrected temperature = NicheMapR prediction + predicted residual
#
#   Two model types are compared:
#     • Random Forest (RF) — an ensemble of decision trees, fast and robust.
#     • LSTM — a neural network designed for time-series data.
#
#   Accuracy is measured by RMSE (Root Mean Squared Error, in °C).
#   Lower RMSE = smaller average error. Improvement % = reduction relative
#   to the uncorrected NicheMapR baseline.
#
# Compare with: Scenario 2 (single logger), Scenario 5 (per-location models)
# Note: the pooled model uses ~10× more training data than Scenario 2, so
#       the performance difference is not purely due to spatial diversity.
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED        <- 42    # fixing the random seed makes results reproducible
SITE_COL    <- "time_series_site"   # column that identifies each logger

# Input data (installed with the package)
DATA_PATH   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH <- system.file("extdata", "beach_splits.csv",            package = "microclCorr")

# Where to write results
RESULTS_DIR <- file.path("inst", "examples", "scenario_4_beach_pooled", "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 4: Beach Pooled ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
# Read the pre-aligned CSV. The function parses the datetime column and
# creates one binary (0/1) column per habitat category (e.g. shade, rock).
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
# The beach data has no "sun" microhabitat — remove that column if present
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL

# ── Step 2: Split into train / validation / test ───────────────────────────────
# We use pre-defined time blocks shared with Scenario 5 so that results are
# directly comparable. Each row is labelled "train", "val", or "test".
#   • train  — rows the model learns from
#   • val    — rows used to tune the model during training (not seen at test time)
#   • test   — rows held out to measure final accuracy
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)
cat(sprintf("Train: %d | Validation: %d | Test: %d rows\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# ── Step 3: Select predictor columns ──────────────────────────────────────────
# Identify which columns to use as model inputs (environmental variables,
# cyclical time features, habitat indicators). The function automatically
# excludes target columns (residual), identifiers (site ID), and timestamps.
feature_cols <- get_feature_columns(splits$train)

# ── Step 4: Train a Random Forest ─────────────────────────────────────────────
# A Random Forest builds many decision trees on random subsets of the data
# and averages their predictions. It is fast and handles non-linear patterns
# without requiring data normalisation.
rf_model <- train_rf(
  splits$train[, feature_cols],  # predictor columns for training rows
  splits$train$residual,         # target: the gap between measured and NicheMapR
  seed = SEED
)

# ── Step 5: Train an LSTM neural network ──────────────────────────────────────
# An LSTM (Long Short-Term Memory) network is designed for sequential data.
# It looks at a short window of consecutive hours and learns temporal patterns
# (e.g. how temperature from the previous 2 hours affects the current error).

# 5a. Normalise all values to the 0–1 range.
#     Neural networks train more stably when inputs are on a common scale.
#     Crucially, the scaling parameters are computed on TRAINING data only
#     so that the test set remains unseen.
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# 5b. Reshape the time series into overlapping windows of fixed length.
#     Each window contains 2 consecutive hours; the model predicts the residual
#     at the final hour of the window.
lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size  = 2,         # look back 2 hours
  ts_names_col = SITE_COL   # column identifying each logger (prevents bridging gaps between sites)
)

# 5c. Fit the LSTM.
lstm_model <- train_lstm(
  lstm_data$train_dict$X, lstm_data$train_dict$y,   # training windows
  lstm_data$val_dict$X,   lstm_data$val_dict$y,     # validation windows (for early stopping)
  n_units    = 32,    # number of memory cells in the LSTM layer
  n_layers   = 1,     # a single LSTM layer is sufficient for hourly microclimate data
  dropout    = 0.0,   # no dropout — the dataset is large enough without regularisation
  lr         = 0.005, # learning rate: how quickly the network adjusts its weights
  epochs     = 40,    # maximum number of passes over the training data
  batch_size = 128,   # number of windows processed per weight update
  patience   = 5,     # stop early if validation error has not improved for 5 epochs
  seed       = SEED
)

# ── Step 6: Align the test sets ───────────────────────────────────────────────
# The LSTM predicts only at the END of each time window (not for every row).
# This step filters the RF test set to those same time points so that both
# models are evaluated on an identical set of rows — a fair comparison.
rf_test <- align_test_sets(
  splits$test,
  lstm_data$test_dict,
  lstm_data$index_info,
  SITE_COL
)

# ── Step 7: Evaluate both models, site by site ────────────────────────────────
# evaluate_correction() computes RMSE before correction (NicheMapR baseline)
# and after correction (model output), for the held-out test rows.
results <- list()

# RF evaluation — one row per site
for (site in unique(rf_test[[SITE_COL]])) {
  mask <- rf_test[[SITE_COL]] == site
  m    <- evaluate_correction(
            rf_model,
            rf_test[mask, feature_cols],
            rf_test$residual[mask],
            rf_test$predicted[mask],
            model_type = "rf")
  results[[length(results) + 1]] <- results_row("RF", site, m)
}

# LSTM evaluation — one row per site (index_info maps windows back to sites)
for (i in seq_along(lstm_data$index_info$datasets)) {
  site <- lstm_data$index_info$datasets[i]
  idx  <- lstm_data$index_info$test_indices[[i]] + 1   # convert 0-based to 1-based index
  m    <- evaluate_correction(
            lstm_model,
            lstm_data$test_dict$X[idx, , , drop = FALSE],
            lstm_data$test_dict$y[idx],
            lstm_data$test_dict$base_pred[idx],
            model_type = "lstm")
  results[[length(results) + 1]] <- results_row("LSTM_2h", site, m)
}

results_df <- do.call(rbind, results)

# ── Step 8: Save results and model ────────────────────────────────────────────
write.csv(results_df,
          file.path(RESULTS_DIR, "beach_pooled_results.csv"),
          row.names = FALSE)

# Save the RF model so it can be loaded and applied to new data later
save_correction_model(rf_model,
                       scaler       = NULL,         # RF does not need a scaler
                       feature_cols = feature_cols,
                       path         = file.path(RESULTS_DIR, "rf_pooled_model.rds"))

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\nAverage performance across all sites:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model, results_df, mean))
cat("=== Scenario 4 complete ===\n")
