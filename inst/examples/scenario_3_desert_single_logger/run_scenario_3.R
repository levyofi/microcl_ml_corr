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

source(system.file("examples", "utils.R", package = "microclCorr"))
setup_tensorflow()
library(reticulate)
py_require("tensorflow")
library(microclCorr)
library(ggplot2)
library(gridExtra)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123            # fixing the random seed makes results reproducible
SITE_COL <- "site_id"     # column that identifies each logger

DATA_PATH    <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_3_desert_single_logger")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Two loggers at the Tzeelim site — one on a rock surface, one under a bush.
# The pipeline is run independently for each microhabitat.
tasks <- list(
  list(name = "Rock_S_T_2_W", site = "Rock_S_T_2_W", title = "Desert - Rock"),
  list(name = "Bush_S_T_2_W", site = "Bush_S_T_2_W", title = "Desert - Bush")
)

cat("=== Scenario 3: Desert Habitat ===\n")

all_results      <- list()
plot_list        <- list()
temp_plot_list   <- list()
stats_list       <- list()
daily_stats_list <- list()

for (task in tasks) {
  cat(sprintf("\n── Microhabitat: %s ──\n", task$name))

  # ── Step 1: Load data ───────────────────────────────────────────────────────
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
  feature_cols <- get_feature_columns(splits$train)

  # ── Step 4 (LSTM): Normalise and create 2-hour windows ──────────────────────
  scaled <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                          window_size = 2, ts_names_col = SITE_COL)

  # ── Step 5: Align test sets ─────────────────────────────────────────────────
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                    lstm_2h$index_info, SITE_COL)
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

  # ── Temperature statistics for this logger ───────────────────────────────────
  stats_list[[task$name]] <- logger_temp_stats(data, task$title)

  # ── Build prediction data frame ───────────────────────────────────────────────
  full_df <- build_pred_df(rf_test, feature_cols, rf_model,
                            base_test_lstm, lstm_model, X_test_lstm)
  full_df <- full_df[order(full_df$time), ]

  # ── Daily min / mean / max RMSE, ME, and SD ──────────────────────────────────
  daily_stats_list[[task$name]] <- compute_daily_stats(full_df)

  # ── Prediction plots ──────────────────────────────────────────────────────────
  is_first <- task$name == tasks[[1]]$name

  plot_list[[task$name]] <- make_pred_plot(
    head(full_df, 120), task$title, show_legend = is_first)

  temp_plot_list[[task$name]] <- make_pred_plot(
    full_df, task$title, show_legend = is_first)
}

# ── Save prediction plot (120-hour excerpt) ───────────────────────────────────
ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert.png"),
       grid.arrange(grobs = plot_list, ncol = 2), width = 12, height = 5, dpi = 300)

# ── Save full test-set temporal plots ─────────────────────────────────────────
ggsave(file.path(SCENARIO_DIR, "temporal_predictions_desert.png"),
       grid.arrange(grobs = temp_plot_list, ncol = 1),
       width = 12, height = 9, dpi = 300)

# ── Save temperature statistics table ─────────────────────────────────────────
stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("\nLogger temperature statistics (full dataset):\n")
print(stats_df)

# ── Performance summary ────────────────────────────────────────────────────────
all_df <- do.call(rbind, all_results)
cat("\nPerformance summary:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model, all_df, mean))

# ── Daily min / mean / max statistics ─────────────────────────────────────────
cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
for (task in tasks) {
  print_daily_stats(daily_stats_list[[task$name]], task$title)
}

cat("=== Scenario 3 complete ===\n")
