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

library(microclCorr)
library(ggplot2)

# ── Settings ──────────────────────────────────────────────────────────────────
SEED     <- 123            # fixing the random seed makes results reproducible
SITE     <- "Ashkelon 15 m"
SITE_COL <- "time_series_site"   # column that identifies each logger

DATA_PATH    <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_2_beach")
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
# Identify which columns to use as model inputs — environmental variables
# (radiation, humidity, wind speed), cyclical time features (hour of day,
# month), and habitat indicators. The function automatically excludes the
# target column (residual), identifiers, and timestamps.
feature_cols <- get_feature_columns(splits$train)

# ── Step 4 (LSTM): Normalise and create 2-hour windows ────────────────────────
# Neural networks train more stably when all inputs are on the same scale.
# Scaling fits the 0–1 range using training data only so the test set
# stays completely unseen.
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# The LSTM looks at a short history of past hours to predict the current
# residual. Here we use a 2-hour window: the model sees hours t-1 and t,
# and predicts the residual at hour t.
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                        window_size = 2, ts_names_col = SITE_COL)

# ── Step 5: Align test sets ───────────────────────────────────────────────────
# The LSTM only produces a prediction at the END of each 2-hour window, not
# for every row. This step trims the RF test set to those same time points
# so both models are evaluated on exactly the same rows — a fair comparison.
rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                   lstm_2h$index_info, SITE_COL)
X_test_lstm    <- lstm_2h$test_dict$X          # 2-hour input windows (test)
y_test_lstm    <- lstm_2h$test_dict$y          # actual residuals (test)
base_test_lstm <- lstm_2h$test_dict$base_pred  # NicheMapR raw predictions (test)

# ── Step 6: Tune and train Random Forest ──────────────────────────────────────
# A Random Forest builds many decision trees and combines their predictions.
# "Tuning" searches over 5 combinations of hyperparameters (tree depth,
# minimum leaf size, number of features per split) and picks the best one
# using the validation set.
cat("  Tuning and training RF...\n")
rf_model <- train_rf(splits$train[, feature_cols], splits$train$residual,
                      tune = TRUE, n_combinations = 5,
                      val_X = splits$val[, feature_cols],
                      val_y = splits$val$residual,
                      seed  = SEED)

# ── Step 7: Tune and train LSTM ───────────────────────────────────────────────
# First, find the best network architecture by testing 5 random configurations:
# how many memory units, how many stacked layers, how much dropout regularisation,
# and what learning rate. The best configuration is chosen based on validation error.
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
                          n_layers   = lstm_params$n_layers,  # stacked LSTM layers
                          dropout    = lstm_params$dropout,   # regularisation rate
                          lr         = lstm_params$lr,        # learning rate
                          epochs = 40, batch_size = 32, patience = 10,
                          seed   = SEED)

# ── Step 8: Evaluate both models ──────────────────────────────────────────────
# Measure how well each model corrected the NicheMapR predictions on the
# held-out test rows. evaluate_correction() returns:
#   rmse_base — RMSE of the raw NicheMapR prediction (before correction)
#   rmse_corr — RMSE after applying the correction model (lower = better)
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
# Saving the model bundles all the information needed to apply corrections to
# new NicheMapR output in the future: the trained model, the scaling parameters
# (so new data is normalised the same way), and the list of input columns.
save_correction_model(rf_model, scaler = NULL, feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "rf_model.rds"))
save_correction_model(lstm_model, scaler = scaled$scaler,
                       feature_cols = feature_cols,
                       path = file.path(RESULTS_DIR, "lstm_model.rds"))

# ── Prediction plot ───────────────────────────────────────────────────────────
# Generate corrected predictions for the test set and plot the first 120 hours.
# Four lines are shown:
#   Observed       — actual temperature recorded by the logger
#   NicheMapR      — original model prediction (before correction)
#   RF Corrected   — NicheMapR + RF-predicted residual
#   LSTM Corrected — NicheMapR + LSTM-predicted residual
rf_preds   <- rf_test$predicted +
              predict(rf_model, data = rf_test[, feature_cols])$predictions
lstm_preds <- base_test_lstm + predict(lstm_model, X_test_lstm, verbose = 0)[, 1]

plot_df <- head(data.frame(
  time     = rf_test$time,
  measured = rf_test$predicted + rf_test$residual,   # actual logger reading
  base     = rf_test$predicted,                       # NicheMapR (uncorrected)
  rf       = rf_preds,
  lstm     = lstm_preds
)[order(rf_test$time), ], 120)   # first 120 hours = 5 days

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
