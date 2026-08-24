
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
DATA_PATH   <- file.path(pkg_base, "../extdata/desert_data_preprocessed.csv")
SPLITS_PATH <- file.path(pkg_base, "../extdata/desert_splits.csv")
SITE_COL    <- "site_id"

data   <- load_prepared_csv_data(DATA_PATH, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)

feature_cols <- get_feature_columns(splits$train)
scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_data    <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test, window_size = 2, ts_names_col = SITE_COL)
rf_test      <- align_test_sets(splits$test, lstm_data$test_dict, lstm_data$index_info, SITE_COL)

rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

X_test_lstm    <- lstm_data$test_dict$X
y_test_lstm    <- lstm_data$test_dict$y
base_test_lstm <- lstm_data$test_dict$base_pred

eval_rows <- list()
pred_dfs  <- list()
daily_rows <- list()

for (i in seq_along(lstm_data$index_info$datasets)) {
  site <- lstm_data$index_info$datasets[i]
  idx_lstm <- lstm_data$index_info$test_indices[[i]] + 1
  
  idx_rf <- which(rf_test[[SITE_COL]] == site)
  sub_rf <- rf_test[idx_rf, ]

  X_sub    <- X_test_lstm[idx_lstm, , , drop = FALSE]
  y_sub    <- y_test_lstm[idx_lstm]
  base_sub <- base_test_lstm[idx_lstm]

  m_rf   <- evaluate_correction(rf_bundle$model, sub_rf[, feature_cols], sub_rf$residual, sub_rf$predicted, model_type = "rf")
  m_lstm <- evaluate_correction(lstm_bundle$model, X_sub, y_sub, base_sub, model_type = "lstm")

  eval_rows[[length(eval_rows)+1]] <- results_row("RF", site, m_rf)
  eval_rows[[length(eval_rows)+1]] <- results_row("LSTM_2h", site, m_lstm)

  pred_rf   <- stats::predict(rf_bundle$model, data = as.data.frame(sub_rf[, feature_cols]))$predictions
  pred_lstm <- as.numeric(lstm_bundle$model |> keras3::predict_on_batch(X_sub))
  
  site_df <- data.frame(
    time     = as.POSIXct(sub_rf$time),
    measured = sub_rf$predicted + sub_rf$residual,
    base     = sub_rf$predicted,
    rf       = sub_rf$predicted + pred_rf,
    lstm     = base_sub + pred_lstm,
    site     = site
  )
  site_df <- site_df[order(site_df$time), ]
  pred_dfs[[site]] <- site_df
  
  ds <- compute_daily_stats(site_df)
  ds$logger <- site
  daily_rows[[site]] <- ds
}

write.csv(do.call(rbind, eval_rows), file.path(RESULTS_DIR, "desert_pooled_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, daily_rows), file.path(RESULTS_DIR, "daily_extreme_rmse.csv"), row.names = FALSE)
write.csv(do.call(rbind, pred_dfs), file.path(RESULTS_DIR, "scenario_6_predictions.csv"), row.names = FALSE)
cat("Scenario 6 stats complete.
")

