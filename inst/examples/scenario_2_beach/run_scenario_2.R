# =============================================================================
# Scenario 2: Coastal Beach Habitat (Ashkelon 15 m logger)
# =============================================================================
# Goal: Train local correction models for a single beach logger and measure
#       how performance changes with the amount of training data available.
#
# Coastal microclimate is strongly influenced by marine winds and sea surface
# temperature, making NicheMapR errors larger and harder to correct than in
# inland habitats. This scenario shows how much data is needed before the
# model captures those coastal dynamics.
#
# See Scenario 1 for a detailed explanation of the pipeline steps.
# Compare with: Scenario 4 (pooled, ~10× more training data)
# =============================================================================

library(microclCorr)
library(ggplot2)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED          <- 123
N_RUNS        <- 5
TRAINING_DAYS <- c(1, 2, 3, 7, 14, 21, 28, 35, 42)

DATA_PATH    <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_2_beach")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

SITE     <- "Ashkelon 15 m"   # the single logger used in this scenario
SITE_COL <- "time_series_site"

# Pre-defined 7-day block assignments for this logger
TRAIN_BLOCKS <- c(0, 1, 3, 4, 5, 6, 7, 8, 9, 10)
VAL_BLOCKS   <- c(11)
TEST_BLOCKS  <- c(2)

cat("=== Scenario 2: Beach Habitat (Ashkelon 15m) ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL

# Keep only the target logger and the relevant time period
data <- data[data[[SITE_COL]] == SITE &
             data$time > as.POSIXct("2025-05-25", tz = "UTC") &
             data$time < as.POSIXct("2025-08-26", tz = "UTC"), ]

# ── Step 2: Split into train / validation / test ───────────────────────────────
splits <- split_train_val_test(data,
                                train_pct    = 0.75,
                                val_pct      = 0.125,
                                block_days   = 7,
                                use_blocks   = TRUE,
                                seed         = SEED,
                                train_blocks = TRAIN_BLOCKS,
                                val_blocks   = VAL_BLOCKS,
                                test_blocks  = TEST_BLOCKS)
cat(sprintf("Train: %d | Validation: %d | Test: %d rows\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test)))

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Step 4 (LSTM prep): Normalise and create time windows ─────────────────────
scaled   <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h  <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                         window_size = 2,  ts_names_col = SITE_COL)
lstm_24h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                         window_size = 24, ts_names_col = SITE_COL)

# ── Step 6: Align test sets ───────────────────────────────────────────────────
rf_test        <- align_test_sets(splits$test, lstm_24h$test_dict,
                                   lstm_24h$index_info, SITE_COL)
X_test_2h      <- lstm_24h$test_dict$X[, 23:24, , drop = FALSE]
y_test_lstm    <- lstm_24h$test_dict$y
base_test_lstm <- lstm_24h$test_dict$base_pred

# ── Tune hyperparameters once using the full training set ─────────────────────
cat("  Tuning RF hyperparameters...\n")
rf_tuned  <- train_rf(splits$train[, feature_cols], splits$train$residual,
                       tune = TRUE, n_combinations = 5,
                       val_X = splits$val[, feature_cols],
                       val_y = splits$val$residual,
                       seed  = SEED)
rf_params <- list(max.depth     = rf_tuned$max.depth,
                   min.node.size = rf_tuned$min.node.size,
                   mtry          = rf_tuned$mtry)

cat("  Tuning LSTM hyperparameters...\n")
hpo         <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                                 lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                                 n_trials = 3, epochs = 40, batch_size = 32,
                                 patience = 5, seed = SEED)
lstm_params <- hpo$params

# ── Learning curve ────────────────────────────────────────────────────────────
train_sorted <- splits$train[order(splits$train$time), ]
lc_results   <- list()

for (n_days in TRAINING_DAYS) {
  n_hours    <- n_days * 24
  rf_partial <- train_sorted[seq_len(min(nrow(train_sorted), n_hours)), ]
  lstm_n     <- min(length(lstm_2h$train_dict$y), n_hours)
  X_partial  <- lstm_2h$train_dict$X[seq_len(lstm_n), , , drop = FALSE]
  y_partial  <- lstm_2h$train_dict$y[seq_len(lstm_n)]

  for (run in 0:(N_RUNS - 1)) {
    rf_m  <- train_rf(rf_partial[, feature_cols], rf_partial$residual,
                       max.depth = rf_params$max.depth,
                       min.node.size = rf_params$min.node.size,
                       mtry = rf_params$mtry, seed = run)
    m_rf  <- evaluate_correction(rf_m, rf_test[, feature_cols],
                                  rf_test$residual, rf_test$predicted,
                                  model_type = "rf")
    lc_results[[length(lc_results) + 1]] <- data.frame(
      model = "RF", n_days = n_days, run = run,
      rmse_base = m_rf$rmse_base, rmse_corr = m_rf$rmse_corr)

    lstm_m <- train_lstm(X_partial, y_partial,
                          lstm_2h$val_dict$X, lstm_2h$val_dict$y,
                          n_units = lstm_params$n_units, n_layers = lstm_params$n_layers,
                          dropout = lstm_params$dropout, lr = lstm_params$lr,
                          epochs = 40, batch_size = 32, patience = 5, seed = run)
    m_lstm <- evaluate_correction(lstm_m, X_test_2h, y_test_lstm, base_test_lstm,
                                   model_type = "lstm")
    lc_results[[length(lc_results) + 1]] <- data.frame(
      model = "LSTM_2h", n_days = n_days, run = run,
      rmse_base = m_lstm$rmse_base, rmse_corr = m_lstm$rmse_corr)
  }
}

lc_df <- do.call(rbind, lc_results)
lc_df$improvement_pct <- (lc_df$rmse_base - lc_df$rmse_corr) / lc_df$rmse_base * 100
write.csv(lc_df, file.path(RESULTS_DIR, "Ashkelon_15_m_results.csv"), row.names = FALSE)

# ── Step 8: Save the full (42-day) model ─────────────────────────────────────
save_correction_model(rf_tuned, scaler = NULL, feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "rf_model.rds"))

# ── Prediction plot ───────────────────────────────────────────────────────────
rf_preds   <- rf_test$predicted + predict(rf_tuned,
                data = rf_test[, feature_cols])$predictions

lstm_full  <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                          lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                          n_units = lstm_params$n_units, n_layers = lstm_params$n_layers,
                          dropout = lstm_params$dropout, lr = lstm_params$lr,
                          epochs = 40, batch_size = 32, patience = 5, seed = SEED)
lstm_preds <- base_test_lstm + predict(lstm_full, X_test_2h, verbose = 0)[, 1]

plot_df <- head(data.frame(
  time     = rf_test$time,
  measured = rf_test$predicted + rf_test$residual,
  base     = rf_test$predicted,
  rf       = rf_preds,
  lstm     = lstm_preds
)[order(rf_test$time), ], 120)

p_pred <- ggplot(plot_df, aes(x = time)) +
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

# ── Learning curve plot ────────────────────────────────────────────────────────
lc_summ    <- aggregate(cbind(rmse_corr, rmse_base) ~ model + n_days, lc_df, mean)
lc_sd      <- aggregate(rmse_corr ~ model + n_days, lc_df, sd)
lc_summ$sd <- lc_sd$rmse_corr

p_lc <- ggplot(lc_summ, aes(x = n_days, y = rmse_corr, color = model, fill = model)) +
  geom_ribbon(aes(ymin = rmse_corr - sd, ymax = rmse_corr + sd),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 1.5) +
  geom_hline(yintercept = mean(lc_summ$rmse_base), color = "#ef4444",
             linetype = "dashed", linewidth = 0.8) +
  scale_color_manual(values = c("LSTM_2h" = "#3b82f6", "RF" = "#10b981"),
                     labels = c("LSTM (2h)", "Random Forest")) +
  scale_fill_manual(values  = c("LSTM_2h" = "#3b82f6", "RF" = "#10b981"),
                    labels  = c("LSTM (2h)", "Random Forest")) +
  scale_x_continuous(breaks = TRAINING_DAYS) +
  labs(title = "Learning Curves: Coastal Beach (Ashkelon)",
       x     = "Training Data Size (Days)",
       y     = "Test RMSE (°C)  — lower is better",
       color = "Model", fill = "Model") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom", panel.grid.minor = element_blank())

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach.png"),
       p_pred, width = 8, height = 4.5, dpi = 300)
ggsave(file.path(SCENARIO_DIR, "learning_curves_beach.png"),
       p_lc,   width = 8, height = 4.5, dpi = 300)

cat("\nPerformance at 42 days of training data:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model,
                lc_df[lc_df$n_days == 42, ], mean))
cat("=== Scenario 2 complete ===\n")
