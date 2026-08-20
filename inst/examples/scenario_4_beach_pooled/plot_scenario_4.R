# scenario_4_beach_pooled/plot_scenario_4.R
# Diagnostic plot generation for Scenario 4: Beach Pooled

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

cat("\n=== Scenario 4: Beach Pooled ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_4_beach_pooled")) "scenario_4_beach_pooled" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH_B   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SITE_COL_B    <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B, is_continuous_microhabitat = FALSE, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b      <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)
rf_bundle_b   <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_b <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

scaled_b  <- lstm_scaling(splits_b$train, splits_b$val, splits_b$test)
lstm_2h_b <- lstm_specific_preprocessing(scaled_b$train, scaled_b$val, scaled_b$test, window_size = 2, ts_names_col = SITE_COL_B)
rf_test_b       <- align_test_sets(splits_b$test, lstm_2h_b$test_dict, lstm_2h_b$index_info, SITE_COL_B)
X_test_lstm_b   <- lstm_2h_b$test_dict$X
base_test_lstm_b <- lstm_2h_b$test_dict$base_pred

rf_preds_b <- rf_test_b$predicted +
  ranger:::predict.ranger(rf_bundle_b$model, data = as.data.frame(rf_test_b[, rf_bundle_b$feature_cols]))$predictions

lstm_preds_b <- rep(NA_real_, nrow(rf_test_b))
all_lstm_raw <- as.numeric(predict(lstm_bundle_b$model, X_test_lstm_b, verbose=0)[, 1])

for (i in seq_along(lstm_2h_b$index_info$datasets)) {
  site_name <- lstm_2h_b$index_info$datasets[i]
  lstm_idx  <- lstm_2h_b$index_info$test_indices[[i]] + 1L
  rf_idx    <- which(rf_test_b[[SITE_COL_B]] == site_name)
  if (length(lstm_idx) == length(rf_idx))
    lstm_preds_b[rf_idx] <- base_test_lstm_b[lstm_idx] + all_lstm_raw[lstm_idx]
}

full_df_b <- data.frame(
  time     = rf_test_b$time,
  measured = rf_test_b$predicted + rf_test_b$residual,
  base     = rf_test_b$predicted,
  rf       = rf_preds_b,
  lstm     = lstm_preds_b
)
full_df_b[[SITE_COL_B]] <- rf_test_b[[SITE_COL_B]]

sites_b      <- unique(rf_test_b[[SITE_COL_B]])
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()

site_letters_s4 <- c(
  "Range_24 25 m" = "(a)", "Rosh_HaNikra 15 m" = "(b)", "Ashkelon 10 m" = "(c)",
  "Range_24 45 m" = "(d)", "Rosh_HaNikra 25 m" = "(e)", "Ashkelon 15 m" = "(f)",
  "Rosh_HaNikra 45 m" = "(g)"
)

for (site in sites_b) {
  mask    <- rf_test_b[[SITE_COL_B]] == site
  site_df <- full_df_b[mask, ]
  site_df <- site_df[order(site_df$time), ]

  site_data <- data_b[data_b[[SITE_COL_B]] == site, ]
  stats_list[[site]] <- logger_temp_stats(site_data, site)

  let_val <- site_letters_s4[site]
  if (is.na(let_val)) let_val <- NULL

  temp_panels[[site]]    <- make_pred_plot(head(site_df, 96), site, show_legend = TRUE, panel_letter = let_val)
  excerpt_panels[[site]] <- make_pred_plot(head(site_df, 96), site, show_legend = TRUE, panel_letter = let_val)
  daily_ext[[site]]      <- compute_daily_stats(site_df)

  xlim_site <- range(c(site_df$measured - site_df$base, site_df$measured - site_df$rf, site_df$measured - site_df$lstm), na.rm = TRUE)
  hist_panels[[site]] <- make_residual_hist(site_df, site, xlim = xlim_site, show_strip = TRUE, show_legend = TRUE, strip_text_size = 11)
}

grid_temp <- list(
  temp_panels[["Range_24 25 m"]], temp_panels[["Rosh_HaNikra 15 m"]], temp_panels[["Ashkelon 10 m"]],
  temp_panels[["Range_24 45 m"]], temp_panels[["Rosh_HaNikra 25 m"]], temp_panels[["Ashkelon 15 m"]],
  NULL,                           temp_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = unname(grid_temp), ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 12, dpi = 300)

grid_excerpt <- list(
  excerpt_panels[["Range_24 25 m"]], excerpt_panels[["Rosh_HaNikra 15 m"]], excerpt_panels[["Ashkelon 10 m"]],
  excerpt_panels[["Range_24 45 m"]], excerpt_panels[["Rosh_HaNikra 25 m"]], excerpt_panels[["Ashkelon 15 m"]],
  NULL,                              excerpt_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = unname(grid_excerpt), ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 12, dpi = 300)

grid_hist <- list(
  hist_panels[["Range_24 25 m"]], hist_panels[["Rosh_HaNikra 15 m"]], hist_panels[["Ashkelon 10 m"]],
  hist_panels[["Range_24 45 m"]], hist_panels[["Rosh_HaNikra 25 m"]], hist_panels[["Ashkelon 15 m"]],
  NULL,                           hist_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = unname(grid_hist), ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 15, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("  Saved all outputs (Scenario 4 Beach Pooled)\n")
