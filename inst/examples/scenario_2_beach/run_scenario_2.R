# =============================================================================
# Scenario 2: Coastal Beach Habitat (Ashkelon 15 m logger)
# =============================================================================
# Goal: Train a local correction model for a single beach logger using the
#       full available training data.
#
# Coastal microclimate is strongly influenced by marine winds and sea surface
# temperature, making NicheMapR errors larger and harder to correct than in
# inland habitats.
#
# See Scenario 1 for a detailed explanation of the pipeline steps.
# Want to know how many days of data you need? See learning_curve_example.R.
# Compare with: Scenario 4 (pooled, all beach loggers combined)
# =============================================================================

library(microclCorr)
library(ggplot2)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123
SITE     <- "Ashkelon 15 m"
SITE_COL <- "time_series_site"

DATA_PATH    <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_2_beach")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 2: Beach Habitat (Ashkelon 15m) ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL
data <- data[data[[SITE_COL]] == SITE &
             data$time > as.POSIXct("2025-05-25", tz = "UTC") &
             data$time < as.POSIXct("2025-08-26", tz = "UTC"), ]

# ── Step 2: Split — 75% train, 12.5% validation, 12.5% test ──────────────────
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
scaled  <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                        window_size = 2, ts_names_col = SITE_COL)

# ── Step 5: Align test sets ───────────────────────────────────────────────────
# Trim the RF test set to the rows where the LSTM produced a prediction.
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
hpo         <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                                 lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                                 n_trials = 5, epochs = 40,
                                 batch_size = 32, patience = 10, seed = SEED)
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

# ── Prediction plot ───────────────────────────────────────────────────────────
rf_preds   <- rf_test$predicted +
              predict(rf_model, data = rf_test[, feature_cols])$predictions
lstm_preds <- base_test_lstm + predict(lstm_model, X_test_lstm, verbose = 0)[, 1]

plot_df <- head(data.frame(
  time     = rf_test$time,
  measured = rf_test$predicted + rf_test$residual,
  base     = rf_test$predicted,
  rf       = rf_preds, lstm = lstm_preds
)[order(rf_test$time), ], 120)

p <- ggplot(plot_df, aes(x = time)) +
  geom_line(aes(y = measured, color = "Observed"),       linewidth = 1.0) +
  geom_line(aes(y = base,     color = "NicheMapR"),      linetype = "dashed",  linewidth = 0.8) +
  geom_line(aes(y = lstm,     color = "LSTM Corrected"), linewidth = 0.9) +
  geom_line(aes(y = rf,       color = "RF Corrected"),   linetype = "dotted",  linewidth = 0.9) +
  scale_color_manual(values = c("Observed" = "#111111", "NicheMapR" = "#ef4444",
                                 "LSTM Corrected" = "#3b82f6", "RF Corrected" = "#10b981")) +
  labs(title = "Coastal Beach (Ashkelon 15m) — First 120 Hours of Test Set",
       x = NULL, y = "Temperature (°C)", color = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top", panel.grid.minor = element_blank())

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach.png"),
       p, width = 8, height = 4.5, dpi = 300)

cat("=== Scenario 2 complete ===\n")
