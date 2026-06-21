# =============================================================================
# Scenario 1: Mediterranean Valley Habitat (Harod)
# =============================================================================
# Goal: Train local correction models for three microhabitats (Sun, Shade, Air)
#       at the Harod valley site and measure how performance changes with the
#       amount of training data available.
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
# Learning curves:
#   We repeat training for 9 different training-set sizes (1 to 42 days) and
#   5 independent random runs each to show how much data is needed before
#   accuracy plateaus. Accuracy is measured by RMSE (°C); lower = better.
# =============================================================================

library(microclCorr)
library(ggplot2)
library(gridExtra)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED          <- 123
N_RUNS        <- 5     # repeat each training size 5 times with different random seeds
TRAINING_DAYS <- c(1, 2, 3, 7, 14, 21, 28, 35, 42)   # training set sizes to test

DATA_PATH    <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_1_valley")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Each "task" is one microhabitat logger at the Harod site.
# train/val/test_blocks are pre-defined 7-day time blocks assigned to each role.
tasks <- list(
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Valley - Sun",
       train_blocks = c(0,1,2,3,4,7), val_blocks = c(5), test_blocks = c(6)),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Valley - Shade",
       train_blocks = c(0,1,2,3,4,7), val_blocks = c(5), test_blocks = c(6)),
  list(name = "harod2_air", site = "harod2_air.csv", title = "Valley - Air",
       train_blocks = c(0,1,2,3,4,7), val_blocks = c(5), test_blocks = c(6))
)

cat("=== Scenario 1: Valley Habitat ===\n")

all_results <- list()
plot_list   <- list()

for (task in tasks) {
  cat(sprintf("\n── Microhabitat: %s ──\n", task$name))

  # ── Step 1: Load data ───────────────────────────────────────────────────────
  # Read the logger data and parse the datetime column.
  # Rows from other loggers are removed so this task trains only on one site.
  data     <- load_prepared_csv_data(DATA_PATH,
                                     is_continuous_microhabitat = FALSE,
                                     datetime_format = "%d/%m/%Y %H:%M",
                                     includes_index  = TRUE)
  data     <- data[data$time_series_doc == task$site, ]
  site_col <- "time_series_doc"

  # ── Step 2: Split into train / validation / test ────────────────────────────
  # Rows are assigned to blocks of 7 consecutive days; those blocks are then
  # allocated to train, val, or test. Using blocks (rather than random rows)
  # prevents the model from "seeing the future" during training.
  splits <- split_train_val_test(data,
                                  train_pct    = 0.75,
                                  val_pct      = 0.125,
                                  block_days   = 7,
                                  use_blocks   = TRUE,
                                  seed         = SEED,
                                  train_blocks = task$train_blocks,
                                  val_blocks   = task$val_blocks,
                                  test_blocks  = task$test_blocks)

  # ── Step 3: Select predictor columns ────────────────────────────────────────
  feature_cols <- get_feature_columns(splits$train)

  # ── Step 4 (LSTM prep): Normalise and create time windows ───────────────────
  # Scale all values to 0–1 using training statistics only (no data leakage).
  scaled  <- lstm_scaling(splits$train, splits$val, splits$test)

  # Create overlapping 2-hour windows — used for the learning curve and final plots.
  # Also create 24-hour windows — used only to align the test set (see Step 6).
  lstm_2h  <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                           window_size = 2,  ts_names_col = site_col)
  lstm_24h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                           window_size = 24, ts_names_col = site_col)

  # ── Step 6: Align the RF and LSTM test sets ──────────────────────────────────
  # The LSTM predicts only at the end of each window. This step trims the RF
  # test set to those same rows so both models are evaluated on identical data.
  rf_test        <- align_test_sets(splits$test, lstm_24h$test_dict,
                                    lstm_24h$index_info, site_col)
  X_test_2h      <- lstm_24h$test_dict$X[, 23:24, , drop = FALSE]  # last 2 hours of each 24h window
  y_test_lstm    <- lstm_24h$test_dict$y
  base_test_lstm <- lstm_24h$test_dict$base_pred   # NicheMapR values (unscaled)

  # ── Tune hyperparameters once using the FULL training set ───────────────────
  # RF: search over combinations of tree depth, node size, and number of features.
  cat("  Tuning RF hyperparameters...\n")
  rf_tuned  <- train_rf(splits$train[, feature_cols], splits$train$residual,
                         tune = TRUE, n_combinations = 5,
                         val_X = splits$val[, feature_cols],
                         val_y = splits$val$residual,
                         seed  = SEED)
  rf_params <- list(max.depth     = rf_tuned$max.depth,
                     min.node.size = rf_tuned$min.node.size,
                     mtry          = rf_tuned$mtry)

  # LSTM: search over network size, dropout, and learning rate combinations.
  cat("  Tuning LSTM hyperparameters...\n")
  hpo         <- lstm_hypertuning(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                                   lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                                   n_trials   = 3,
                                   epochs     = 40,
                                   batch_size = 32,
                                   patience   = 5,
                                   seed       = SEED)
  lstm_params <- hpo$params   # best found configuration

  # ── Learning curve: train on progressively more data and record accuracy ─────
  # The training rows are sorted by time so that "n days" always means
  # the FIRST n days of available data (simulating real deployment).
  train_sorted <- splits$train[order(splits$train$time), ]
  task_results <- list()

  for (n_days in TRAINING_DAYS) {
    n_hours    <- n_days * 24
    # Take the first n_hours rows of training data
    rf_partial <- train_sorted[seq_len(min(nrow(train_sorted), n_hours)), ]
    lstm_n     <- min(length(lstm_2h$train_dict$y), n_hours)
    X_partial  <- lstm_2h$train_dict$X[seq_len(lstm_n), , , drop = FALSE]
    y_partial  <- lstm_2h$train_dict$y[seq_len(lstm_n)]

    for (run in 0:(N_RUNS - 1)) {
      # RF: train with tuned params, different seed per run
      rf_m  <- train_rf(rf_partial[, feature_cols], rf_partial$residual,
                         max.depth     = rf_params$max.depth,
                         min.node.size = rf_params$min.node.size,
                         mtry          = rf_params$mtry,
                         seed          = run)
      m_rf  <- evaluate_correction(rf_m, rf_test[, feature_cols],
                                    rf_test$residual, rf_test$predicted,
                                    model_type = "rf")
      task_results[[length(task_results) + 1]] <- data.frame(
        model = "RF", n_days = n_days, run = run,
        rmse_base = m_rf$rmse_base, rmse_corr = m_rf$rmse_corr)

      # LSTM: train with tuned params, different seed per run
      lstm_m <- train_lstm(X_partial, y_partial,
                            lstm_2h$val_dict$X, lstm_2h$val_dict$y,
                            n_units    = lstm_params$n_units,
                            n_layers   = lstm_params$n_layers,
                            dropout    = lstm_params$dropout,
                            lr         = lstm_params$lr,
                            epochs     = 40,
                            batch_size = 32,
                            patience   = 5,
                            seed       = run)
      m_lstm <- evaluate_correction(lstm_m, X_test_2h, y_test_lstm, base_test_lstm,
                                     model_type = "lstm")
      task_results[[length(task_results) + 1]] <- data.frame(
        model = "LSTM_2h", n_days = n_days, run = run,
        rmse_base = m_lstm$rmse_base, rmse_corr = m_lstm$rmse_corr)
    }
  }

  task_df <- do.call(rbind, task_results)
  task_df$improvement_pct <- (task_df$rmse_base - task_df$rmse_corr) /
                              task_df$rmse_base * 100
  write.csv(task_df, file.path(RESULTS_DIR, paste0(task$name, "_results.csv")),
            row.names = FALSE)
  all_results[[task$name]] <- task_df

  # ── Step 8: Save the full (42-day) model ────────────────────────────────────
  save_correction_model(rf_tuned, scaler = NULL, feature_cols = feature_cols,
                         path = file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds")))

  # ── Prediction plot: show observed vs NicheMapR vs corrected (first 120 h) ──
  rf_preds   <- rf_test$predicted + predict(rf_tuned,
                  data = rf_test[, feature_cols])$predictions

  # Train one LSTM with full data for the prediction plot
  lstm_full  <- train_lstm(lstm_2h$train_dict$X, lstm_2h$train_dict$y,
                            lstm_2h$val_dict$X,   lstm_2h$val_dict$y,
                            n_units    = lstm_params$n_units,
                            n_layers   = lstm_params$n_layers,
                            dropout    = lstm_params$dropout,
                            lr         = lstm_params$lr,
                            epochs     = 40, batch_size = 32, patience = 5,
                            seed       = SEED)
  lstm_preds <- base_test_lstm + predict(lstm_full, X_test_2h, verbose = 0)[, 1]

  plot_df <- head(data.frame(
    time     = rf_test$time,
    measured = rf_test$predicted + rf_test$residual,   # actual logger reading
    base     = rf_test$predicted,                       # NicheMapR (uncorrected)
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
    theme(plot.title       = element_text(face = "bold", hjust = 0.5),
          legend.position  = if (task$name == tasks[[1]]$name) "top" else "none",
          panel.grid.minor = element_blank())
}

# ── Learning curve plot ────────────────────────────────────────────────────────
# Average RMSE across microhabitats and runs at each training size.
# The ribbon shows ±1 standard deviation across the N_RUNS repetitions.
# The red dashed line is the uncorrected NicheMapR baseline.
all_df  <- do.call(rbind, all_results)
lc_summ <- aggregate(cbind(rmse_corr, rmse_base) ~ model + n_days, all_df, mean)
lc_sd   <- aggregate(rmse_corr ~ model + n_days, all_df, sd)
lc_summ$sd_corr <- lc_sd$rmse_corr

p_lc <- ggplot(lc_summ, aes(x = n_days, y = rmse_corr, color = model, fill = model)) +
  geom_ribbon(aes(ymin = rmse_corr - sd_corr, ymax = rmse_corr + sd_corr),
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
  labs(title = "Learning Curves: Valley Habitat (Harod)",
       x     = "Training Data Size (Days)",
       y     = "Test RMSE (°C)  — lower is better",
       color = "Model", fill = "Model") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom", panel.grid.minor = element_blank())

ggsave(file.path(SCENARIO_DIR, "learning_curves_valley.png"),
       p_lc, width = 8, height = 4.5, dpi = 300)
ggsave(file.path(SCENARIO_DIR, "prediction_examples_valley.png"),
       grid.arrange(grobs = plot_list, ncol = 3), width = 15, height = 5, dpi = 300)

cat("\nPerformance at 42 days of training data:\n")
print(aggregate(cbind(rmse_base, rmse_corr, improvement_pct) ~ model,
                all_df[all_df$n_days == 42, ], mean))
cat("=== Scenario 1 complete ===\n")
