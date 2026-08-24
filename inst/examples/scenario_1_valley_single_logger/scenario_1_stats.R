
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
DATA_PATH    <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")

SEED <- 42

tasks <- list(
  list(name = "harod2_air", site = "harod2_air.csv", title = "Air (1m)"),
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Ground Sun (0cm)"),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Ground Shade (0cm)")
)

eval_rows   <- list()
pred_dfs    <- list()
daily_rows  <- list()

for (task in tasks) {
  data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE,
                                 datetime_format = "%d/%m/%Y %H:%M", includes_index = TRUE)
  data <- data[data$time_series_doc == task$site, ]
  
  splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125,
                                 block_days = 7, use_blocks = TRUE, seed = SEED)
  feature_cols <- get_feature_columns(splits$train)
  scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test, window_size = 2, ts_names_col = "time_series_doc")
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict, lstm_2h$index_info, "time_series_doc")
  
  rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds")))
  lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, paste0(task$name, "_lstm_model.rds")))
  
  m_rf   <- evaluate_correction(rf_bundle$model, rf_test[, feature_cols], rf_test$residual, rf_test$predicted, model_type = "rf")
  m_lstm <- evaluate_correction(lstm_bundle$model, lstm_2h$test_dict$X, lstm_2h$test_dict$y, lstm_2h$test_dict$base_pred, model_type = "lstm")
  
  eval_rows[[length(eval_rows)+1]] <- results_row("RF", task$name, m_rf)
  eval_rows[[length(eval_rows)+1]] <- results_row("LSTM_2h", task$name, m_lstm)
  
  full_df <- build_pred_df(rf_test, feature_cols, rf_bundle$model, lstm_2h$test_dict$base_pred, lstm_bundle$model, lstm_2h$test_dict$X)
  full_df$site <- task$name
  pred_dfs[[task$name]] <- full_df
  
  ds <- compute_daily_stats(full_df)
  ds$logger <- task$name
  daily_rows[[task$name]] <- ds
}

res_df <- do.call(rbind, eval_rows)
write.csv(res_df, file.path(RESULTS_DIR, "harod_all_results.csv"), row.names = FALSE)

daily_df <- do.call(rbind, daily_rows)
write.csv(daily_df, file.path(RESULTS_DIR, "daily_extreme_rmse.csv"), row.names = FALSE)

all_preds <- do.call(rbind, pred_dfs)
write.csv(all_preds, file.path(RESULTS_DIR, "scenario_1_predictions.csv"), row.names = FALSE)

cat("Scenario 1 stats and predictions CSV written successfully.
")

