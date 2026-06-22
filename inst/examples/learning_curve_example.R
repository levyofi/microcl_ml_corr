# =============================================================================
# Learning Curve Example — How much logger data do you actually need?
# =============================================================================
# This script answers a practical question: how many days of temperature
# logger measurements do you need before the correction model plateaus?
#
# It uses find_min_training_days() from utils.R, which:
#   1. Trains both RF and LSTM at 9 different training sizes (1–42 days),
#      repeating each size 5 times to estimate variance.
#   2. Finds the minimum number of days where RMSE stays within `tolerance`
#      of the full-data RMSE (default: 10% worse at most).
#   3. Returns a learning curve plot and a results table.
#
# IMPORTANT: Run this AFTER running a main scenario script, since it reuses
# the already-prepared splits, LSTM windows, and trained models from that run.
# This example uses Scenario 1 (Valley, Shade microhabitat) as a template.
# =============================================================================

library(microclCorr)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123
SITE     <- "harod2_shd.csv"
SITE_COL <- "time_series_doc"
DATA_PATH <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")

# ── Reproduce the pipeline from Scenario 1 (shade microhabitat) ───────────────
# (These are the same steps as run_scenario_1.R — included here so this script
#  is self-contained and can be run independently.)

data     <- load_prepared_csv_data(DATA_PATH,
                                   is_continuous_microhabitat = FALSE,
                                   datetime_format = "%d/%m/%Y %H:%M",
                                   includes_index  = TRUE)
data     <- data[data[[SITE_COL]] == SITE, ]

splits <- split_train_val_test(data,
                                train_pct  = 0.75,
                                val_pct    = 0.125,
                                block_days = 7,
                                use_blocks = TRUE,
                                seed       = SEED)

feature_cols <- get_feature_columns(splits$train)

scaled  <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                        window_size = 2, ts_names_col = SITE_COL)

rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                   lstm_2h$index_info, SITE_COL)
X_test_lstm    <- lstm_2h$test_dict$X
y_test_lstm    <- lstm_2h$test_dict$y
base_test_lstm <- lstm_2h$test_dict$base_pred

# Train full-data RF and LSTM (these are the reference models)
cat("Training full-data RF...\n")
rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                      tune = TRUE, n_combinations = 5,
                      val_X = splits$val[, feature_cols],
                      val_y = splits$val$residual,
                      seed  = SEED)

cat("Tuning and training full-data LSTM...\n")
hpo <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                         lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                         n_trials = 5, epochs = 40,
                         batch_size = 32, patience = 10, seed = SEED)
lstm_params <- hpo$params
lstm_model  <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                           lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                           n_units    = lstm_params$n_units,
                           n_layers   = lstm_params$n_layers,
                           dropout    = lstm_params$dropout,
                           lr         = lstm_params$lr,
                           epochs = 40, batch_size = 32, patience = 10,
                           seed   = SEED)

# ── Run the learning curve search ─────────────────────────────────────────────
# tolerance = 0.10 means: find the fewest days where RMSE is at most 10% worse
#             than training on all 42 days of available data.
# n_runs = 5 means: repeat each training size 5 times with different random seeds
#          so the ribbon on the plot shows how stable the result is.
cat("\nSearching for minimum training days (tolerance = 10%)...\n")

lc <- find_min_training_days(
  splits        = splits,
  lstm_2h       = lstm_2h,
  feature_cols  = feature_cols,
  rf_model      = rf_model,
  lstm_model    = lstm_model,
  lstm_params   = lstm_params,
  rf_test       = rf_test,
  X_test_lstm   = X_test_lstm,
  y_test_lstm   = y_test_lstm,
  base_test_lstm = base_test_lstm,
  site_col      = SITE_COL,
  tolerance     = 0.10,            # accept RMSE within 10% of full-data RMSE
  training_days = c(1, 2, 3, 7, 14, 21, 28, 35, 42),
  n_runs        = 5,
  seed          = SEED
)

# ── Results ───────────────────────────────────────────────────────────────────
cat("\nMinimum days needed (10% tolerance):\n")
print(lc$min_days)
#   RF:      X days
#   LSTM_2h: Y days

cat("\nAverage RMSE by model and training size:\n")
print(lc$summary[, c("model", "n_days", "rmse_corr", "sd_corr")])

# ── Save the learning curve plot ──────────────────────────────────────────────
ggsave(
  file.path("inst", "examples", "scenario_1_valley",
            "learning_curve_shade.png"),
  lc$plot,
  width = 8, height = 5, dpi = 300
)
cat("\nPlot saved to inst/examples/scenario_1_valley/learning_curve_shade.png\n")
