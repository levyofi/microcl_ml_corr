# =============================================================================
# Scenario 3: Judean Desert Habitat (Tzeelim)
# =============================================================================
# Goal: Train local correction models for two desert microhabitats (Rock, Bush)
#       at the Tzeelim site using the full available training data.
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
# Note: Desert temperature patterns are very regular (hot days, cool nights,
# repeated daily). Both models learn the correction pattern quickly — often
# from just 1–2 days of data. See learning_curve_example.R to verify this.
#
# Want to know how many days of data you need? See learning_curve_example.R.
# Compare with: Scenario 6 (pooled, all 48 desert loggers combined)
# =============================================================================

library(microclCorr)
library(ggplot2)
library(gridExtra)   # for arranging multiple plots side by side

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123            # fixing the random seed makes results reproducible
SITE_COL <- "site_id"     # column that identifies each logger

DATA_PATH    <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_3_desert")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Two loggers at the Tzeelim site — one on a rock surface, one under a bush.
# The pipeline is run independently for each microhabitat.
tasks <- list(
  list(name = "Rock_S_T_2_W", site = "Rock_S_T_2_W", title = "Desert - Rock"),
  list(name = "Bush_S_T_2_W", site = "Bush_S_T_2_W", title = "Desert - Bush")
)

cat("=== Scenario 3: Desert Habitat ===\n")

all_results <- list()
plot_list   <- list()

for (task in tasks) {
  cat(sprintf("\n── Microhabitat: %s ──\n", task$name))

  # ── Step 1: Load data ───────────────────────────────────────────────────────
  # Read the pre-aligned CSV, parse the datetime column, and create one binary
  # (0/1) column per habitat category. Keep only the current logger's rows.
  data <- load_prepared_csv_data(DATA_PATH,
                                  datetime_format = "%Y-%m-%d %H:%M:%S",
                                  includes_index  = TRUE)
  data <- data[data[[SITE_COL]] == task$site, ]

  # ── Step 2: Split into train / validation / test ────────────────────────────
  # Rows are divided into 7-day blocks, then randomly assigned:
  #   75% to training   — what the model learns from
  #   12.5% to validation — used to monitor the model during training
  #   12.5% to test      — held out to measure final accuracy
  # Using whole blocks prevents the model from "seeing the future".
  splits <- split_train_val_test(data,
                                  train_pct  = 0.75,
                                  val_pct    = 0.125,
                                  block_days = 7,
                                  use_blocks = TRUE,
                                  seed       = SEED)
  cat(sprintf("Train: %d | Val: %d | Test: %d rows\n",
              nrow(splits$train), nrow(splits$val), nrow(splits$test)))

  # ── Step 3: Select predictor columns ────────────────────────────────────────
  # Identify which columns to use as model inputs — environmental variables
  # (radiation, humidity, wind speed), cyclical time features (hour, month),
  # and habitat indicators. Identifiers, timestamps, and the target column
  # (residual) are automatically excluded.
  feature_cols <- get_feature_columns(splits$train)

  # ── Step 4 (LSTM): Normalise and create 2-hour windows ──────────────────────
  # Scale all values to the 0–1 range using training data statistics only
  # (no information from the test set leaks into the model this way).
  scaled <- lstm_scaling(splits$train, splits$val, splits$test)

  # Reshape the time series into overlapping 2-hour windows. Each window
  # contains 2 consecutive hours; the model predicts the residual at the
  # last hour of the window. All LSTM results in this script use 2h windows.
  lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                          window_size = 2, ts_names_col = SITE_COL)

  # ── Step 5: Align test sets ─────────────────────────────────────────────────
  # The LSTM only produces a prediction at the END of each 2-hour window,
  # not for every row. Trim the RF test set to those same time points so
  # both models are evaluated on exactly the same rows — a fair comparison.
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                    lstm_2h$index_info, SITE_COL)
  X_test_lstm    <- lstm_2h$test_dict$X          # 2-hour input windows (test)
  y_test_lstm    <- lstm_2h$test_dict$y          # actual residuals (test)
  base_test_lstm <- lstm_2h$test_dict$base_pred  # NicheMapR raw predictions (test)

  # ── Step 6: Tune and train Random Forest ────────────────────────────────────
  # A Random Forest builds many decision trees and combines their predictions.
  # "Tuning" searches over 5 combinations of hyperparameters and picks the
  # best one using the validation set.
  cat("  Tuning and training RF...\n")
  rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                        tune = TRUE, n_combinations = 5,
                        val_X = splits$val[, feature_cols],
                        val_y = splits$val$residual,
                        seed  = SEED)

  # ── Step 7: Tune and train LSTM ─────────────────────────────────────────────
  # First, find the best network architecture by testing 5 random configurations:
  # how many memory units, layers, how much dropout, and what learning rate.
  # The best configuration is chosen based on validation error.
  cat("  Tuning LSTM hyperparameters...\n")
  hpo <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                           lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                           n_trials   = 5,    # number of architectures to try
                           epochs     = 40,   # maximum training passes per trial
                           batch_size = 32,   # rows processed per weight update
                           patience   = 10,   # stop early if no improvement for 10 epochs
                           seed       = SEED)
  lstm_params <- hpo$params   # best architecture found

  # Now train the LSTM with the best architecture on the full training set.
  cat("  Training LSTM with best architecture...\n")
  lstm_model <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                            lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                            n_units    = lstm_params$n_units,   # memory cells
                            n_layers   = lstm_params$n_layers,  # stacked layers
                            dropout    = lstm_params$dropout,   # regularisation
                            lr         = lstm_params$lr,        # learning rate
                            epochs = 40, batch_size = 32, patience = 10,
                            seed   = SEED)

  # ── Step 8: Evaluate both models ────────────────────────────────────────────
  # Measure how well each model corrected NicheMapR on the held-out test rows.
  #   rmse_base — RMSE of the raw NicheMapR prediction (before correction)
  #   rmse_corr — RMSE after applying the correction model (lower = better)
  #   improvement_pct — percentage reduction in error
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
  # Saving the model bundles everything needed to correct new NicheMapR output:
  # the trained model, the scaling parameters, and the list of input columns.
  save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR,
                                          paste0(task$name, "_rf_model.rds")))
  save_correction_model(lstm_model, scaler = scaled$scaler,
                         feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR,
                                          paste0(task$name, "_lstm_model.rds")))

  # ── Prediction plot ──────────────────────────────────────────────────────────
  # Plot the first 120 hours (5 days) of the test set showing:
  #   Observed       — actual temperature from the logger
  #   NicheMapR      — original prediction before correction
  #   RF Corrected   — NicheMapR + RF-predicted residual
  #   LSTM Corrected — NicheMapR + LSTM-predicted residual
  rf_preds   <- rf_test$predicted +
                predict(rf_model, data = rf_test[, feature_cols])$predictions
  lstm_preds <- base_test_lstm +
                predict(lstm_model, X_test_lstm, verbose = 0)[, 1]

  plot_df <- head(data.frame(
    time     = rf_test$time,
    measured = rf_test$predicted + rf_test$residual,
    base     = rf_test$predicted,
    rf       = rf_preds, lstm = lstm_preds
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

# ── Save prediction plots and summary ─────────────────────────────────────────
ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert.png"),
       grid.arrange(grobs = plot_list, ncol = 2), width = 12, height = 5, dpi = 300)

all_df <- do.call(rbind, all_results)
cat("\nPerformance summary:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model, all_df, mean))
cat("=== Scenario 3 complete ===\n")
