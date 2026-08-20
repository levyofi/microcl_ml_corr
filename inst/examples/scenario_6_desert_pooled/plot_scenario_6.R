# scenario_6_desert_pooled/plot_scenario_6.R
# Diagnostic plot generation for Scenario 6: Desert Pooled

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

cat("\n=== Scenario 6: Desert Pooled ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_6_desert_pooled")) "scenario_6_desert_pooled" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH_D   <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_D <- system.file("extdata", "desert_splits.csv", package = "microclCorr")
SITE_COL_D    <- "site_id"

data_d <- load_prepared_csv_data(DATA_PATH_D, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
splits_d    <- load_splits_from_csv(data_d, SPLITS_PATH_D, SITE_COL_D)
rf_bundle_d   <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_d <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

scaled_d  <- lstm_scaling(splits_d$train, splits_d$val, splits_d$test)
lstm_2h_d <- lstm_specific_preprocessing(scaled_d$train, scaled_d$val, scaled_d$test, window_size = 2, ts_names_col = SITE_COL_D)
rf_test_d        <- align_test_sets(splits_d$test, lstm_2h_d$test_dict, lstm_2h_d$index_info, SITE_COL_D)
X_test_lstm_d    <- lstm_2h_d$test_dict$X
base_test_lstm_d <- lstm_2h_d$test_dict$base_pred

rf_preds_d <- rf_test_d$predicted +
  ranger:::predict.ranger(rf_bundle_d$model, data = as.data.frame(rf_test_d[, rf_bundle_d$feature_cols]))$predictions

lstm_preds_d  <- rep(NA_real_, nrow(rf_test_d))
all_lstm_raw_d <- as.numeric(predict(lstm_bundle_d$model, X_test_lstm_d, verbose=0)[, 1])

for (i in seq_along(lstm_2h_d$index_info$datasets)) {
  site_name <- lstm_2h_d$index_info$datasets[i]
  lstm_idx  <- lstm_2h_d$index_info$test_indices[[i]] + 1L
  rf_idx    <- which(rf_test_d[[SITE_COL_D]] == site_name)
  if (length(lstm_idx) == length(rf_idx))
    lstm_preds_d[rf_idx] <- base_test_lstm_d[lstm_idx] + all_lstm_raw_d[lstm_idx]
}

full_df_d_all <- data.frame(
  time     = rf_test_d$time,
  measured = rf_test_d$predicted + rf_test_d$residual,
  base     = rf_test_d$predicted,
  rf       = rf_preds_d,
  lstm     = lstm_preds_d
)
full_df_d_all[[SITE_COL_D]] <- rf_test_d[[SITE_COL_D]]

sites_d        <- unique(rf_test_d[[SITE_COL_D]])
sites_d_sample <- head(sites_d, 6)
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()

for (site in sites_d) {
  mask    <- rf_test_d[[SITE_COL_D]] == site
  full_df <- full_df_d_all[mask, ]
  full_df <- full_df[order(full_df$time), ]

  site_data <- data_d[data_d[[SITE_COL_D]] == site, ]
  stats_list[[site]] <- logger_temp_stats(site_data, site)
  daily_ext[[site]]  <- compute_daily_stats(full_df)

  if (site %in% sites_d_sample) {
    temp_panels[[site]]    <- make_pred_plot(full_df, site, show_legend = TRUE)
    excerpt_panels[[site]] <- make_pred_plot(head(full_df, 96), site, show_legend = TRUE)
    xlim_site <- range(c(full_df$measured - full_df$base, full_df$measured - full_df$rf, full_df$measured - full_df$lstm), na.rm = TRUE)
    hist_panels[[site]] <- make_residual_hist(full_df, site, xlim = xlim_site, show_strip = TRUE, show_legend = TRUE, strip_text_size = 11)
  }
}

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_desert_pooled.png"),
       arrange_with_legend(temp_panels, ncol = 2, legend_pos = "bottom"),
       width = 14, height = ceiling(length(temp_panels) / 2) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert_pooled.png"),
       arrange_with_legend(excerpt_panels, ncol = 2, legend_pos = "bottom"),
       width = 14, height = ceiling(length(excerpt_panels) / 2) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert_pooled.png"),
       arrange_with_legend(hist_panels, ncol = 2, legend_pos = "bottom"),
       width = 14, height = ceiling(length(hist_panels) / 2) * 8, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("  Saved all outputs (Scenario 6 Desert Pooled)\n")
