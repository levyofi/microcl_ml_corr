# =============================================================================
# Scenario 1: Mediterranean Valley Habitat (Harod)
# =============================================================================
# Goal: Train local correction models for three microhabitats (Sun, Shade, Air)
#       at the Harod valley site using the full available training data (42 days).
#
# What the pipeline does:
#   NicheMapR predicts microclimate temperatures but is not perfect.
#   The gap between the prediction and what a logger measured is the "residual":
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
#
# Want to know how many days of data you actually need?
#   See inst/examples/learning_curve_example.R for find_min_training_days().
# =============================================================================

library(microclCorr)
library(ggplot2)
library(gridExtra)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED <- 123

DATA_PATH    <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_1_valley_single_logger")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Each "task" is one microhabitat logger at the Harod site.
tasks <- list(
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Valley - Sun"),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Valley - Shade"),
  list(name = "harod2_air", site = "harod2_air.csv", title = "Valley - Air")
)

cat("=== Scenario 1: Valley Habitat ===\n")

all_results <- list()
plot_list   <- list()

for (task in tasks) {
  cat(sprintf("\n── Microhabitat: %s ──\n", task$name))

  # ── Step 1: Load data ───────────────────────────────────────────────────────
  data     <- load_prepared_csv_data(DATA_PATH,
                                     is_continuous_microhabitat = FALSE,
                                     datetime_format = "%d/%m/%Y %H:%M",
                                     includes_index  = TRUE)
  data     <- data[data$time_series_doc == task$site, ]
  site_col <- "time_series_doc"

  # ── Step 2: Split — 75% train, 12.5% validation, 12.5% test ────────────────
  # Blocks of 7 consecutive days are randomly assigned to each role.
  # This prevents the model from "seeing the future" during training.
  splits <- split_train_val_test(data,
                                  train_pct  = 0.75,
                                  val_pct    = 0.125,
                                  block_days = 7,
                                  use_blocks = TRUE,
                                  seed       = SEED)
  cat(sprintf("Train: %d | Val: %d | Test: %d rows\n",
              nrow(splits$train), nrow(splits$val), nrow(splits$test)))

  # ── Step 3: Select predictor columns ────────────────────────────────────────
  feature_cols <- get_feature_columns(splits$train)

  # ── Step 4 (LSTM): Normalise and create 2-hour windows ──────────────────────
  # Scale all values to 0–1 (using training data only — no data leakage).
  # Then reshape the time series into overlapping 2-hour windows:
  # each window covers 2 consecutive hours; the model predicts the residual
  # at the last hour of the window.
  scaled  <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                          window_size = 2, ts_names_col = site_col)

  # ── Step 5: Align test sets ─────────────────────────────────────────────────
  # The LSTM predicts only at the end of each 2-hour window (not every row).
  # This trims the RF test set to those same rows for a fair comparison.
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                    lstm_2h$index_info, site_col)
  X_test_lstm    <- lstm_2h$test_dict$X
  y_test_lstm    <- lstm_2h$test_dict$y
  base_test_lstm <- lstm_2h$test_dict$base_pred

  # ── Step 6: Tune and train Random Forest ────────────────────────────────────
  cat("  Tuning and training RF...\n")
  rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                        tune = TRUE, n_combinations = 5,
                        val_X = splits$val[, feature_cols],
                        val_y = splits$val$residual,
                        seed  = SEED)

  # ── Step 7: Tune and train LSTM ─────────────────────────────────────────────
  # First find the best network architecture (units, layers, dropout, lr)
  # using 5 random trials on the validation set.
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

  # ── Step 8: Evaluate both models ────────────────────────────────────────────
  m_rf   <- evaluate_correction(rf_model, rf_test[, feature_cols],
                                  rf_test$residual, rf_test$predicted,
                                  model_type = "rf")
  m_lstm <- evaluate_correction(lstm_model, X_test_lstm, y_test_lstm,
                                  base_test_lstm, model_type = "lstm")

  result <- rbind(
    data.frame(model = "RF",      rmse_base = m_rf$rmse_base,
               rmse_corr = m_rf$rmse_corr,   task = task$name),
    data.frame(model = "LSTM_2h", rmse_base = m_lstm$rmse_base,
               rmse_corr = m_lstm$rmse_corr, task = task$name)
  )
  result$improvement_pct <- (result$rmse_base - result$rmse_corr) /
                             result$rmse_base * 100
  all_results[[task$name]] <- result
  write.csv(result, file.path(RESULTS_DIR, paste0(task$name, "_results.csv")),
            row.names = FALSE)

  # ── Step 9: Save models ──────────────────────────────────────────────────────
  save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR,
                                          paste0(task$name, "_rf_model.rds")))
  save_correction_model(lstm_model, scaler = scaled$scaler,
                         feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR,
                                          paste0(task$name, "_lstm_model.rds")))

  # ── Prediction plot ──────────────────────────────────────────────────────────
  rf_preds   <- rf_test$predicted +
                predict(rf_model, data = rf_test[, feature_cols])$predictions
  lstm_preds <- base_test_lstm +
                predict(lstm_model, X_test_lstm, verbose = 0)[, 1]

  plot_df <- head(data.frame(
    time     = rf_test$time,
    measured = rf_test$predicted + rf_test$residual,
    base     = rf_test$predicted,
    rf       = rf_preds,
    lstm     = lstm_preds
  )[order(rf_test$time), ], 120)

  plot_list[[task$name]] <- ggplot(plot_df, aes(x = time)) +
    geom_line(aes(y = measured, color = "Observed"),       linewidth = 1.0) +
    geom_line(aes(y = base,     color = "NicheMapR"),      linetype = "dashed",  linewidth = 0.8) +
    geom_line(aes(y = lstm,     color = "LSTM Corrected"), linewidth = 0.9) +
    geom_line(aes(y = rf,       color = "RF Corrected"),   linetype = "dotted",  linewidth = 0.9) +
    scale_color_manual(values = c("Observed" = "#111111", "NicheMapR" = "#ef4444",
                                   "LSTM Corrected" = "#3b82f6", "RF Corrected" = "#10b981")) +
    labs(title = task$title, x = NULL, y = "Temperature (°C)", color = NULL) +
    theme_minimal(base_size = 10) +
    theme(plot.title      = element_text(face = "bold", hjust = 0.5),
          legend.position = if (task$name == tasks[[1]]$name) "top" else "none",
          panel.grid.minor = element_blank())
}

# ── Save prediction plot and summary ──────────────────────────────────────────
ggsave(file.path(SCENARIO_DIR, "prediction_examples_valley.png"),
       grid.arrange(grobs = plot_list, ncol = 3), width = 15, height = 5, dpi = 300)

all_df <- do.call(rbind, all_results)
cat("\nPerformance summary:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model, all_df, mean))
cat("=== Scenario 1 complete ===\n")
