# scenario_1_valley_single_logger/plot_scenario_1.R
# Diagnostic plot generation for Scenario 1: Valley (Harod)

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

cat("\n=== Scenario 1: Valley (Harod) ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_1_valley_single_logger")) "scenario_1_valley_single_logger" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")

tasks_s1 <- list(
  list(name = "harod2_air", site = "harod2_air.csv", title = "Air"),
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Sun"),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Shade")
)

temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()
full_dfs       <- list()

for (task in tasks_s1) {
  cat(sprintf("  %s ...\n", task$name))
  data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE,
                                  datetime_format = "%d/%m/%Y %H:%M",
                                  includes_index  = TRUE)
  data <- data[data$time_series_doc == task$site, ]
  stats_list[[task$name]] <- logger_temp_stats(data, task$title)

  splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125,
                                  block_days = 7, use_blocks = TRUE, seed = SEED)
  feature_cols <- get_feature_columns(splits$train)
  scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                               window_size = 2,
                                               ts_names_col = "time_series_doc")
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                    lstm_2h$index_info, "time_series_doc")
  X_test_lstm    <- lstm_2h$test_dict$X
  base_test_lstm <- lstm_2h$test_dict$base_pred

  rf_bundle   <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds")))
  lstm_bundle <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_lstm_model.rds")))

  full_df <- build_pred_df(rf_test, rf_bundle$feature_cols, rf_bundle$model,
                            base_test_lstm, lstm_bundle$model, X_test_lstm)
  full_df <- full_df[order(full_df$time), ]
  full_dfs[[task$name]] <- full_df

  letter_val <- c("harod2_air"="(a)", "harod2_sun"="(b)", "harod2_shd"="(c)")[task$name]
  temp_panels[[task$name]]    <- make_pred_plot(full_df, task$title,
                                                 show_legend = TRUE, panel_letter = letter_val)
  excerpt_panels[[task$name]] <- make_pred_plot(head(full_df, 96), task$title,
                                                 show_legend = TRUE, panel_letter = letter_val)
  daily_ext[[task$name]]      <- compute_daily_stats(full_df)
}

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_valley.png"),
       ggpubr::ggarrange(plotlist = unname(temp_panels), ncol = 1, common.legend = TRUE, legend = "top"),
       width = 12, height = 10, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_valley.png"),
       ggpubr::ggarrange(plotlist = unname(excerpt_panels), ncol = 1, common.legend = TRUE, legend = "top"),
       width = 12, height = 10, dpi = 300)

col_letters_s1 <- list(
  harod2_air = c("(a)", "(d)", "(g)"),
  harod2_sun = c("(b)", "(e)", "(h)"),
  harod2_shd = c("(c)", "(f)", "(i)")
)

for (i in seq_along(tasks_s1)) {
  task <- tasks_s1[[i]]
  df   <- full_dfs[[task$name]]
  xlim_task <- range(c(df$measured - df$base,
                       df$measured - df$rf,
                       df$measured - df$lstm), na.rm = TRUE)
  hist_panels[[task$name]] <- make_residual_hist(
    df, task$title,
    xlim = xlim_task, show_strip = (i == length(tasks_s1)),
    show_legend = TRUE, panel_letters = col_letters_s1[[i]],
    axis_title_size = 18, axis_text_size = 15, strip_text_size = 15)
}

p_grid_s1 <- cowplot::plot_grid(
  hist_panels[[1]], hist_panels[[2]], hist_panels[[3]],
  ncol = 3, align = "h", axis = "tb"
)
leg_s1 <- cowplot::get_legend(hist_panels[[1]] + ggplot2::theme(legend.position = "bottom", legend.direction = "horizontal"))
fig4_final <- cowplot::plot_grid(p_grid_s1, leg_s1, ncol = 1, rel_heights = c(1, 0.08))

ggsave(file.path(SCENARIO_DIR, "residual_histogram_valley.png"),
       fig4_final, width = 15, height = 8, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)
cat("  Saved all outputs (Scenario 1 Valley)\n")
