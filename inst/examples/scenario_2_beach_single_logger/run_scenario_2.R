# =============================================================================
# Scenario 2: Coastal Beach Habitat (Ashkelon 15 m logger)
# =============================================================================
# Goal: Train a local correction model for a single beach logger using the
#       full available training data.
#
# What the pipeline does:
#   NicheMapR predicts microclimate temperatures but is not perfect.
#   The gap between its prediction and what the logger actually measured
#   is called the "residual":
#
#       residual = measured temperature − NicheMapR prediction
#
#   We train a model to predict that residual. The corrected temperature is:
#
#       corrected temperature = NicheMapR prediction + predicted residual
#
#   Two model types are compared:
#     • Random Forest (RF) — an ensemble of decision trees, fast and robust.
#     • LSTM (2 h window) — a neural network that uses the past 2 hours of
#       measurements to predict the current residual.
#
#   Accuracy is measured by RMSE (°C); lower = better.
#   Improvement % = how much the model reduced the original NicheMapR error.
#
# Note: Coastal microclimate is strongly influenced by marine winds and sea
# surface temperature, making NicheMapR errors larger and harder to correct
# than in inland habitats. More training data is needed here than in the desert.
#
# Want to know how many days of data you need? See learning_curve_example.R.
# Compare with: Scenario 4 (pooled, all beach loggers combined)
# =============================================================================

source(system.file("examples", "utils.R", package = "microclCorr"))
setup_tensorflow()
library(reticulate)
py_require("tensorflow")
library(microclCorr)
library(ggplot2)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123            # fixing the random seed makes results reproducible
SITE     <- "Ashkelon 15 m"
SITE_COL <- "time_series_site"   # column that identifies each logger

DATA_PATH    <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_2_beach_single_logger")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 2: Beach Habitat (Ashkelon 15m) ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
# Read the pre-aligned CSV, parse the datetime column, and create one binary
# (0/1) column per habitat category (e.g. shade = 1, rock = 0).
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
# The beach dataset has no "sun" microhabitat — remove that column if present
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL
# Keep only the target logger and the relevant summer season
data <- data[data[[SITE_COL]] == SITE &
             data$time > as.POSIXct("2025-05-25", tz = "UTC") &
             data$time < as.POSIXct("2025-08-26", tz = "UTC"), ]

# Compute temperature stats on the full (filtered) dataset before splitting
temp_stats <- logger_temp_stats(data, paste0("Beach - ", SITE))

# ── Step 2: Split into train / validation / test ───────────────────────────────
# Rows are divided into 7-day blocks, which are then randomly assigned:
#   75% to training (what the model learns from)
#   12.5% to validation (used to monitor the model during training)
#   12.5% to test (held out to measure final accuracy — never seen during training)
# Using whole blocks rather than random rows prevents the model from
# "seeing the future" — each block is either fully in or fully out.
splits <- split_train_val_test(data,
                                train_pct  = 0.75,
                                val_pct    = 0.125,
                                block_days = 7,
                                use_blocks = TRUE,
                                seed       = SEED)
cat(sprintf("Train: %d | Val: %d | Test: %d rows\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Step 4 (LSTM): Normalise and create 2-hour windows ────────────────────────
scaled <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                        window_size = 2, ts_names_col = SITE_COL)

# ── Step 5: Align test sets ───────────────────────────────────────────────────
rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                   lstm_2h$index_info, SITE_COL)
X_test_lstm    <- lstm_2h$test_dict$X
y_test_lstm    <- lstm_2h$test_dict$y
base_test_lstm <- lstm_2h$test_dict$base_pred

# ── Step 6: Tune and train Random Forest ──────────────────────────────────────
cat("  Tuning and training RF...\n")
rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                      tune = TRUE, n_combinations = 5,
                      val_X = splits$val[, feature_cols],
                      val_y = splits$val$residual,
                      seed  = SEED)

# ── Step 7: Tune and train LSTM ───────────────────────────────────────────────
cat("  Tuning LSTM hyperparameters...\n")
hpo <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                         lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                         n_trials   = 5,
                         epochs     = 40,
                         batch_size = 32,
                         patience   = 10,
                         seed       = SEED)
lstm_params <- hpo$params

cat("  Training LSTM with best architecture...\n")
lstm_model <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                          lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                          n_units    = lstm_params$n_units,
                          n_layers   = lstm_params$n_layers,
                          dropout    = lstm_params$dropout,
                          lr         = lstm_params$lr,
                          epochs = 40, batch_size = 32, patience = 10,
                          seed   = SEED)

# ── Step 8: Evaluate both models ──────────────────────────────────────────────
m_rf   <- evaluate_correction(rf_model, rf_test[, feature_cols],
                                rf_test$residual, rf_test$predicted,
                                model_type = "rf")
m_lstm <- evaluate_correction(lstm_model, X_test_lstm, y_test_lstm,
                                base_test_lstm, model_type = "lstm")

result <- rbind(
  data.frame(model = "RF",      rmse_base = m_rf$rmse_base,
             rmse_corr = m_rf$rmse_corr),
  data.frame(model = "LSTM_2h", rmse_base = m_lstm$rmse_base,
             rmse_corr = m_lstm$rmse_corr)
)
result$improvement_pct <- (result$rmse_base - result$rmse_corr) /
                           result$rmse_base * 100

cat("\nResults:\n"); print(result)
write.csv(result, file.path(RESULTS_DIR, "Ashkelon_15_m_results.csv"),
          row.names = FALSE)

# ── Step 9: Save models ───────────────────────────────────────────────────────
save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "rf_model.rds"))
save_correction_model(lstm_model, scaler = scaled$scaler,
                       feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "lstm_model.rds"))

# ── Save temperature statistics table ─────────────────────────────────────────
write.csv(temp_stats,
          file.path(RESULTS_DIR, "Ashkelon_15_m_temp_stats.csv"),
          row.names = FALSE)
cat("\nLogger temperature statistics (full dataset):\n")
print(temp_stats)

# ── Build prediction data frame and plots ─────────────────────────────────────
full_df <- build_pred_df(rf_test, feature_cols, rf_model,
                          base_test_lstm, lstm_model, X_test_lstm)
full_df <- full_df[order(full_df$time), ]

# 120-hour excerpt
ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach.png"),
       make_pred_plot(head(full_df, 120),
                      "Coastal Beach (Ashkelon 15 m) — First 120 Hours of Test Set"),
       width = 8, height = 4.5, dpi = 300)

# Full test-set temporal plot
ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach.png"),
       make_pred_plot(full_df,
                      "Coastal Beach (Ashkelon 15 m) — Full Test Set"),
       width = 12, height = 5, dpi = 300)

# ── Daily min / mean / max statistics ─────────────────────────────────────────
cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
print_daily_stats(compute_daily_stats(full_df), paste0("Beach - ", SITE))

cat("=== Scenario 2 complete ===\n")
