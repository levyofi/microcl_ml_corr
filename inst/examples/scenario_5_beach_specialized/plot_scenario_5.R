# scenario_5_beach_specialized/plot_scenario_5.R
# Diagnostic plot generation for Scenario 5: Beach Specialized

library(ggplot2)
library(ggpubr)
library(gridExtra)
library(cowplot)

if (!exists("SEED")) SEED <- 123
if (!exists("load_prepared_csv_data")) {
  root_dir <- if (file.exists("utils.R")) "." else ".."
  source(file.path(root_dir, "package_utils.R"))
  source(file.path(root_dir, "utils.R"))
  library(keras3)
  library(reticulate)
  library(microclCorr)
}

cat("\n=== Scenario 5: Beach Specialized ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_5_beach_specialized")) "scenario_5_beach_specialized" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH_B   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SITE_COL_B    <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B, is_continuous_microhabitat = FALSE, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b      <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)

locs_s5        <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()

for (loc in locs_s5) {
  rf_bundle_loc   <- load_correction_model(file.path(RESULTS_DIR, paste0("rf_",   loc, "_model.rds")))
  lstm_bundle_loc <- load_correction_model(file.path(RESULTS_DIR, paste0("lstm_", loc, "_model.rds")))

  train_loc <- splits_b$train[splits_b$train$location == loc, ]
  val_loc   <- splits_b$val  [splits_b$val$location   == loc, ]
  test_loc  <- splits_b$test [splits_b$test$location  == loc, ]

  scaled_loc  <- lstm_scaling(train_loc, val_loc, test_loc)
  lstm_2h_loc <- lstm_specific_preprocessing(scaled_loc$train, scaled_loc$val, scaled_loc$test, window_size = 2, ts_names_col = SITE_COL_B)
  rf_test_loc       <- align_test_sets(test_loc, lstm_2h_loc$test_dict, lstm_2h_loc$index_info, SITE_COL_B)
  X_test_lstm_loc   <- lstm_2h_loc$test_dict$X
  base_test_lstm_loc <- lstm_2h_loc$test_dict$base_pred

  rf_preds_loc <- rf_test_loc$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model, data = as.data.frame(rf_test_loc[, rf_bundle_loc$feature_cols]))$predictions
  lstm_raw_loc   <- as.numeric(predict(lstm_bundle_loc$model, X_test_lstm_loc, verbose=0)[, 1])
  lstm_preds_loc <- rep(NA_real_, nrow(rf_test_loc))
  for (i in seq_along(lstm_2h_loc$index_info$datasets)) {
    sn       <- lstm_2h_loc$index_info$datasets[i]
    li       <- lstm_2h_loc$index_info$test_indices[[i]] + 1L
    ri       <- which(rf_test_loc[[SITE_COL_B]] == sn)
    if (length(li) == length(ri))
      lstm_preds_loc[ri] <- base_test_lstm_loc[li] + lstm_raw_loc[li]
  }
  full_df_loc <- data.frame(
    time     = rf_test_loc$time,
    measured = rf_test_loc$predicted + rf_test_loc$residual,
    base     = rf_test_loc$predicted,
    rf       = rf_preds_loc,
    lstm     = lstm_preds_loc
  )
  full_df_loc[[SITE_COL_B]] <- rf_test_loc[[SITE_COL_B]]

  sites_loc <- unique(rf_test_loc[[SITE_COL_B]])
  for (site in sites_loc) {
    mask    <- rf_test_loc[[SITE_COL_B]] == site
    site_df <- full_df_loc[mask, ]
    site_df <- site_df[order(site_df$time), ]

    site_data <- data_b[data_b[[SITE_COL_B]] == site, ]
    stats_list[[site]] <- logger_temp_stats(site_data, site)

    panel_key <- paste(loc, site, sep = "|")

    temp_panels[[panel_key]]    <- make_pred_plot(head(site_df, 96), site, show_legend = TRUE)
    excerpt_panels[[panel_key]] <- make_pred_plot(head(site_df, 96), site, show_legend = TRUE)
    daily_ext[[panel_key]]      <- compute_daily_stats(site_df)

    xlim_site <- range(c(site_df$measured - site_df$base, site_df$measured - site_df$rf, site_df$measured - site_df$lstm), na.rm = TRUE)
    hist_panels[[panel_key]] <- make_residual_hist(site_df, site, xlim = xlim_site, show_strip = TRUE, show_legend = TRUE, strip_text_size = 11)
  }
}

ncols_s5 <- 2

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach_specialized.png"),
       arrange_with_legend(temp_panels, ncol = ncols_s5, legend_pos = "bottom"),
       width = 14, height = ceiling(length(temp_panels) / ncols_s5) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_specialized.png"),
       arrange_with_legend(excerpt_panels, ncol = ncols_s5, legend_pos = "bottom"),
       width = 14, height = ceiling(length(excerpt_panels) / ncols_s5) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach_specialized.png"),
       arrange_with_legend(hist_panels, ncol = ncols_s5, legend_pos = "bottom"),
       width = 14, height = ceiling(length(hist_panels) / ncols_s5) * 10, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("  Saved all outputs (Scenario 5 Beach Specialized)\n")
