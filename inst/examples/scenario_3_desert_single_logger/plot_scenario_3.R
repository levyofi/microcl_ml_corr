# scenario_3_desert_single_logger/plot_scenario_3.R
# Diagnostic plot generation for Scenario 3: Judean Desert (Tzeelim)

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

cat("\n=== Scenario 3: Judean Desert (Tzeelim) ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_3_desert_single_logger")) "scenario_3_desert_single_logger" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

SITE_COL_3 <- "site_id"
DATA_PATH  <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")

tasks_s3 <- list(
  list(name = "Rock_S_T_2_W", site = "Rock_S_T_2_W", title = "Desert - Rock"),
  list(name = "Bush_S_T_2_W", site = "Bush_S_T_2_W", title = "Desert - Bush")
)

temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()

for (task in tasks_s3) {
  cat(sprintf("  %s ...\n", task$name))
  data <- load_prepared_csv_data(DATA_PATH, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
  data <- data[data[[SITE_COL_3]] == task$site, ]
  stats_list[[task$name]] <- logger_temp_stats(data, task$title)

  splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125, block_days = 7, use_blocks = TRUE, seed = SEED)
  feature_cols <- get_feature_columns(splits$train)
  scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test, window_size = 2, ts_names_col = SITE_COL_3)
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict, lstm_2h$index_info, SITE_COL_3)
  X_test_lstm    <- lstm_2h$test_dict$X
  base_test_lstm <- lstm_2h$test_dict$base_pred

  rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds")))
  lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, paste0(task$name, "_lstm_model.rds")))

  full_df <- build_pred_df(rf_test, rf_bundle$feature_cols, rf_bundle$model, base_test_lstm, lstm_bundle$model, X_test_lstm)
  full_df <- full_df[order(full_df$time), ]

  temp_panels[[task$name]]    <- make_pred_plot(head(full_df, 96), task$title, show_legend = TRUE)
  excerpt_panels[[task$name]] <- make_pred_plot(head(full_df, 96), task$title, show_legend = TRUE)
  hist_panels[[task$name]]    <- make_residual_hist(full_df, task$title, show_legend = TRUE, strip_text_size = 11)
  daily_ext[[task$name]]      <- compute_daily_stats(full_df)
}

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert.png"),
       ggpubr::ggarrange(plotlist = unname(excerpt_panels), ncol = 2, common.legend = TRUE, legend = "top"),
       width = 12, height = 5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert.png"),
       ggpubr::ggarrange(plotlist = unname(hist_panels), ncol = 2, common.legend = TRUE, legend = "bottom"),
       width = 12, height = 12, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("  Saved all outputs (Scenario 3 Desert)\n")
