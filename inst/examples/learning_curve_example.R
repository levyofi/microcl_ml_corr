# =============================================================================
# Learning Curve Example — How much logger data do you actually need?
# =============================================================================
#
# Background
# ----------
# The main scenario scripts (run_scenario_1.R, run_scenario_2.R, etc.) always
# train on the full available dataset (~42 days). But in practice you might
# want to know: can I get good results with only 1 week of data? Or 2 weeks?
#
# This script answers that question by training the same models on progressively
# smaller amounts of data and measuring how accuracy changes. The result is a
# "learning curve" — a plot of prediction error vs. training data size.
#
# How it works
# ------------
# find_min_training_days() trains both models at 9 different training sizes
# (1, 2, 3, 7, 14, 21, 28, 35, and 42 days). At each size, training is
# repeated 5 times with different random seeds; the average and spread across
# those 5 runs becomes the curve and ribbon on the plot.
#
# The `tolerance` parameter defines how close to the full-data accuracy you
# are willing to accept. tolerance = 0.10 means: find the fewest days where
# the prediction error (RMSE) is at most 10% worse than using all 42 days.
#
# This example uses the Valley - Shade microhabitat from Scenario 1.
# To adapt it to your own site, change SITE, SITE_COL, and DATA_PATH below.
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED      <- 123
SITE      <- "harod2_shd.csv"   # which logger to analyse
SITE_COL  <- "time_series_doc"  # column that identifies each logger
DATA_PATH <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")

# ── Step 1: Load and prepare the data ─────────────────────────────────────────
# These are the same steps as in run_scenario_1.R — repeated here so this
# script can be run on its own without sourcing the main scenario first.

data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%d/%m/%Y %H:%M",
                               includes_index  = TRUE)
# Keep only the logger we want to analyse
data <- data[data[[SITE_COL]] == SITE, ]

# ── Step 2: Split into train / validation / test ───────────────────────────────
# 75% of 7-day blocks go to training, 12.5% to validation, 12.5% to testing.
# Using blocks (not random rows) prevents the model from "seeing the future".
splits <- split_train_val_test(data,
                                train_pct  = 0.75,
                                val_pct    = 0.125,
                                block_days = 7,
                                use_blocks = TRUE,
                                seed       = SEED)
cat(sprintf("Data split — Train: %d rows | Val: %d rows | Test: %d rows\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# ── Step 3: Prepare features and LSTM windows ─────────────────────────────────
# Identify which columns to use as model inputs (environmental variables,
# cyclical time features, habitat indicators).
feature_cols <- get_feature_columns(splits$train)

# Normalise all values to the 0–1 range using training data statistics only.
# Then reshape the time series into 2-hour sliding windows for the LSTM.
# Each window contains 2 consecutive hours; the model predicts the residual
# at the last hour of the window.
scaled  <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                        window_size = 2, ts_names_col = SITE_COL)

# ── Step 4: Align test sets ───────────────────────────────────────────────────
# The LSTM only predicts at the end of each 2-hour window, not every row.
# Trim the RF test set to those same rows so both models are evaluated fairly.
rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                   lstm_2h$index_info, SITE_COL)
X_test_lstm    <- lstm_2h$test_dict$X          # LSTM input windows (test set)
y_test_lstm    <- lstm_2h$test_dict$y          # actual residuals (test set)
base_test_lstm <- lstm_2h$test_dict$base_pred  # NicheMapR predictions (unscaled)

# ── Step 5: Train the full-data reference models ───────────────────────────────
# These are the "gold standard" — trained on all available data.
# find_min_training_days() will compare smaller-data results against these.

cat("Training reference RF on full dataset...\n")
rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                      tune = TRUE, n_combinations = 5,  # search 5 HP combinations
                      val_X = splits$val[, feature_cols],
                      val_y = splits$val$residual,
                      seed  = SEED)

cat("Finding best LSTM architecture on full dataset...\n")
# lstm_hypertuning() tries 5 different network configurations and picks the best.
hpo         <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                                 lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                                 n_trials = 5, epochs = 40,
                                 batch_size = 32, patience = 10, seed = SEED)
lstm_params <- hpo$params   # best architecture found (units, layers, dropout, lr)

cat("Training reference LSTM on full dataset...\n")
lstm_model <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                          lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                          n_units    = lstm_params$n_units,
                          n_layers   = lstm_params$n_layers,
                          dropout    = lstm_params$dropout,
                          lr         = lstm_params$lr,
                          epochs = 40, batch_size = 32, patience = 10,
                          seed   = SEED)

# ── Step 6: Run the learning curve search ─────────────────────────────────────
# For each training size (1 to 42 days), train both models 5 times and record
# the prediction error. Then find the fewest days where error stays within
# 10% of the full-data result.
#
# This step takes several minutes because it trains 9 sizes × 5 runs × 2 models
# = 90 models in total.
cat("\nRunning learning curve (this may take several minutes)...\n")

lc <- find_min_training_days(
  splits         = splits,
  lstm_2h        = lstm_2h,
  feature_cols   = feature_cols,
  rf_model       = rf_model,       # full-data reference model
  lstm_model     = lstm_model,     # full-data reference model
  lstm_params    = lstm_params,    # architecture to reuse at each partial size
  rf_test        = rf_test,
  X_test_lstm    = X_test_lstm,
  y_test_lstm    = y_test_lstm,
  base_test_lstm = base_test_lstm,
  site_col       = SITE_COL,
  tolerance     = 0.10,            # accept RMSE up to 10% worse than full-data
  training_days = c(1, 2, 3, 7, 14, 21, 28, 35, 42),  # sizes to test
  n_runs        = 5,               # repetitions per size (for variance estimate)
  seed          = SEED
)

# ── Step 7: Inspect results ───────────────────────────────────────────────────

# Minimum days needed at 10% tolerance
cat("\nMinimum training days needed (within 10% of full-data accuracy):\n")
print(lc$min_days)
# Example output:
#      RF  LSTM_2h
#       3       14
# Interpretation: RF reaches near-full accuracy after just 3 days of data;
# the LSTM needs 14 days to stabilise.

# Average RMSE at each training size
cat("\nMean prediction error (RMSE) by model and training size:\n")
print(lc$summary[, c("model", "n_days", "rmse_corr", "sd_corr")])
# rmse_corr = average RMSE across 5 runs (lower = better)
# sd_corr   = standard deviation across runs (smaller = more stable)

# ── Step 8: Save the learning curve plot ──────────────────────────────────────
# The plot shows RMSE vs. training size for both models.
# The ribbon around each line shows ±1 SD across the 5 repeated runs.
# The dashed horizontal lines mark the 10% tolerance threshold for each model.
ggsave(
  file.path("inst", "examples", "scenario_1_valley", "learning_curve_shade.png"),
  lc$plot,
  width = 8, height = 5, dpi = 300
)
cat("Learning curve plot saved.\n")
