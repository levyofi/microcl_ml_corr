
suppressPackageStartupMessages({
  library(microclCorr)
  library(ranger)
  library(keras3)
})

pkg_base <- ".."
if (!file.exists(file.path(pkg_base, "package_utils.R"))) {
  pkg_base <- system.file("examples", package = "microclCorr")
}
source(file.path(pkg_base, "package_utils.R"))
source(file.path(pkg_base, "utils.R"))

SCENARIO_DIR <- "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
DATA_PATH    <- file.path(pkg_base, "../extdata/Beach_data_preprocessed.csv")
SEED <- 42

data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL
data <- data[data$time_series_site == "Ashkelon 15 m", ]

splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125, block_days = 7, use_blocks = TRUE, seed = SEED)
feature_cols <- get_feature_columns(splits$train)
scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test, window_size = 2, ts_names_col = "time_series_site")
rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict, lstm_2h$index_info, "time_series_site")

rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, "rf_model.rds"))
lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, "lstm_model.rds"))

m_rf   <- evaluate_correction(rf_bundle$model, rf_test[, feature_cols], rf_test$residual, rf_test$predicted, model_type = "rf")
m_lstm <- evaluate_correction(lstm_bundle$model, lstm_2h$test_dict$X, lstm_2h$test_dict$y, lstm_2h$test_dict$base_pred, model_type = "lstm")

res_df <- rbind(results_row("RF", "Ashkelon_15_m", m_rf), results_row("LSTM_2h", "Ashkelon_15_m", m_lstm))
write.csv(res_df, file.path(RESULTS_DIR, "Ashkelon_15_m_results.csv"), row.names = FALSE)

full_df <- build_pred_df(rf_test, feature_cols, rf_bundle$model, lstm_2h$test_dict$base_pred, lstm_bundle$model, lstm_2h$test_dict$X)
full_df$site <- "Ashkelon_15_m"
write.csv(full_df, file.path(RESULTS_DIR, "scenario_2_predictions.csv"), row.names = FALSE)

ds <- compute_daily_stats(full_df)
ds$logger <- "Ashkelon_15_m"
write.csv(ds, file.path(RESULTS_DIR, "daily_extreme_rmse.csv"), row.names = FALSE)
cat("Scenario 2 stats complete.
")

