
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
DATA_PATH    <- file.path(pkg_base, "../extdata/desert_data_preprocessed.csv")
SEED <- 42

loggers <- c("Rock_S_T_2_W", "Bush_S_T_2_W")
eval_rows  <- list()
pred_dfs   <- list()
daily_rows <- list()

for (logger in loggers) {
  data <- load_prepared_csv_data(DATA_PATH, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
  data <- data[data$site_id == logger, ]
  
  splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125, block_days = 7, use_blocks = TRUE, seed = SEED)
  feature_cols <- get_feature_columns(splits$train)
  scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test, window_size = 2, ts_names_col = "site_id")
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict, lstm_2h$index_info, "site_id")
  
  rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, paste0(logger, "_rf_model.rds")))
  lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, paste0(logger, "_lstm_model.rds")))
  
  m_rf   <- evaluate_correction(rf_bundle$model, rf_test[, feature_cols], rf_test$residual, rf_test$predicted, model_type = "rf")
  m_lstm <- evaluate_correction(lstm_bundle$model, lstm_2h$test_dict$X, lstm_2h$test_dict$y, lstm_2h$test_dict$base_pred, model_type = "lstm")
  
  eval_rows[[length(eval_rows)+1]] <- results_row("RF", logger, m_rf)
  eval_rows[[length(eval_rows)+1]] <- results_row("LSTM_2h", logger, m_lstm)
  
  full_df <- build_pred_df(rf_test, feature_cols, rf_bundle$model, lstm_2h$test_dict$base_pred, lstm_bundle$model, lstm_2h$test_dict$X)
  full_df$site <- logger
  pred_dfs[[logger]] <- full_df
  
  ds <- compute_daily_stats(full_df)
  ds$logger <- logger
  daily_rows[[logger]] <- ds
}

write.csv(do.call(rbind, eval_rows), file.path(RESULTS_DIR, "desert_single_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, daily_rows), file.path(RESULTS_DIR, "daily_extreme_rmse.csv"), row.names = FALSE)
write.csv(do.call(rbind, pred_dfs), file.path(RESULTS_DIR, "scenario_3_predictions.csv"), row.names = FALSE)
cat("Scenario 3 stats complete.
")

