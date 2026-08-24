
suppressPackageStartupMessages({
  library(microclCorr)
  library(ranger)
})

pkg_base <- ".."
if (!file.exists(file.path(pkg_base, "package_utils.R"))) {
  pkg_base <- system.file("examples", package = "microclCorr")
}
source(file.path(pkg_base, "package_utils.R"))
source(file.path(pkg_base, "utils.R"))

SCENARIO_DIR <- "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")
DATA_PATH   <- file.path(pkg_base, "../extdata/Beach_data_preprocessed.csv")
SPLITS_PATH <- file.path(pkg_base, "../extdata/beach_splits.csv")
SITE_COL    <- "time_series_site"

data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL

splits       <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)
feature_cols <- get_feature_columns(splits$train)

locations <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
eval_rows  <- list()
pred_dfs   <- list()
daily_rows <- list()

for (loc in locations) {
  rf_bundle <- load_correction_model(file.path(RESULTS_DIR, paste0("rf_zeroshot_", loc, "_model.rds")))
  sub_rf <- splits$test[grep(loc, splits$test[[SITE_COL]]), ]
  
  m_rf <- evaluate_correction(rf_bundle$model, sub_rf[, feature_cols], sub_rf$residual, sub_rf$predicted, model_type = "rf")

  eval_rows[[length(eval_rows)+1]] <- results_row("RF", loc, m_rf)

  pred_rf <- stats::predict(rf_bundle$model, data = as.data.frame(sub_rf[, feature_cols]))$predictions
  
  site_df <- data.frame(
    time     = as.POSIXct(sub_rf$time),
    measured = sub_rf$predicted + sub_rf$residual,
    base     = sub_rf$predicted,
    rf       = sub_rf$predicted + pred_rf,
    site     = loc
  )
  site_df <- site_df[order(site_df$time), ]
  pred_dfs[[loc]] <- site_df
  
  ds <- compute_daily_stats(site_df)
  ds$logger <- loc
  daily_rows[[loc]] <- ds
}

write.csv(do.call(rbind, eval_rows), file.path(RESULTS_DIR, "zero_shot_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, daily_rows), file.path(RESULTS_DIR, "daily_extreme_rmse.csv"), row.names = FALSE)
write.csv(do.call(rbind, pred_dfs), file.path(RESULTS_DIR, "scenario_8_predictions.csv"), row.names = FALSE)
cat("Scenario 8 stats complete.
")

