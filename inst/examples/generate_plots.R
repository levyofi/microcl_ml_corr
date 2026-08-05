# =============================================================================
# generate_plots.R
# Regenerate all diagnostic plots and statistics for scenarios 1–8 using
# saved model .rds/.keras files — no re-training required.
#
# Outputs per scenario:
#   residual_histogram_*.png   — before/after residual distributions (all models)
#   prediction_examples_*.png  — 120-hour excerpt panels
#   temporal_predictions_*.png — full test-set time series
#   logger_temp_stats.csv      — measured vs NicheMapR temperature statistics
#   daily_extreme_rmse.csv     — RMSE of daily min/mean/max per logger / model
# =============================================================================

Sys.setenv(UV_OFFLINE="1", KERAS_HOME=getwd())
source("package_utils.R")
source("utils.R")
library(keras3)
# setup_tensorflow()
library(reticulate)
# py_require("tensorflow")
library(microclCorr)
library(ggplot2)
library(gridExtra)

SEED <- 123

# =============================================================================
# SCENARIO 1: Valley (Harod) — Sun, Shade, Air
# =============================================================================
cat("\n=== Scenario 1: Valley (Harod) ===\n")

DATA_PATH    <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")
SCENARIO_DIR <- "scenario_1_valley_single_logger"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

tasks_s1 <- list(
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Valley - Sun"),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Valley - Shade"),
  list(name = "harod2_air", site = "harod2_air.csv", title = "Valley - Air")
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

  is_first <- task$name == tasks_s1[[1]]$name
  temp_panels[[task$name]]    <- make_pred_plot(full_df, task$title,
                                                 show_legend = is_first)
  excerpt_panels[[task$name]] <- make_pred_plot(head(full_df, 120), task$title,
                                                 show_legend = is_first)
  daily_ext[[task$name]]      <- compute_daily_stats(full_df)
}

# Per-task x limits: each column uses its own range so narrow distributions
# (e.g. Air) are not squished by wide ones (e.g. Sun).
for (i in seq_along(tasks_s1)) {
  task <- tasks_s1[[i]]
  df   <- full_dfs[[task$name]]
  xlim_task <- range(c(df$measured - df$base,
                        df$measured - df$rf,
                        df$measured - df$lstm), na.rm = TRUE)
  hist_panels[[task$name]] <- make_residual_hist(
    df, task$title,
    xlim = xlim_task, show_strip = (i == length(tasks_s1)),
    show_legend = TRUE)
}

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_valley.png"),
       grid.arrange(grobs = temp_panels, ncol = 1),
       width = 12, height = 10, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_valley.png"),
       grid.arrange(grobs = excerpt_panels, ncol = 3),
       width = 15, height = 5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_valley.png"),
       ggpubr::ggarrange(plotlist = hist_panels, ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 5.5, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
for (task in tasks_s1) print_daily_stats(daily_ext[[task$name]], task$title)
cat("  Saved all outputs (Valley)\n")

# =============================================================================
# SCENARIO 2: Coastal Beach (Ashkelon 15 m)
# =============================================================================
cat("\n=== Scenario 2: Coastal Beach (Ashkelon 15 m) ===\n")

SITE      <- "Ashkelon 15 m"
SITE_COL  <- "time_series_site"
DATA_PATH <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- "scenario_2_beach_single_logger"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL
data <- data[data[[SITE_COL]] == SITE &
             data$time > as.POSIXct("2025-05-25", tz = "UTC") &
             data$time < as.POSIXct("2025-08-26", tz = "UTC"), ]

temp_stats_s2 <- logger_temp_stats(data, paste0("Beach - ", SITE))
write.csv(temp_stats_s2, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125,
                                block_days = 7, use_blocks = TRUE, seed = SEED)
feature_cols <- get_feature_columns(splits$train)
scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                             window_size = 2, ts_names_col = SITE_COL)
rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                   lstm_2h$index_info, SITE_COL)
X_test_lstm    <- lstm_2h$test_dict$X
base_test_lstm <- lstm_2h$test_dict$base_pred

rf_bundle   <- load_correction_model(file.path(RESULTS_DIR, "rf_model.rds"))
lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, "lstm_model.rds"))

full_df <- build_pred_df(rf_test, rf_bundle$feature_cols, rf_bundle$model,
                          base_test_lstm, lstm_bundle$model, X_test_lstm)
full_df <- full_df[order(full_df$time), ]

label_s2 <- paste0("Coastal Beach (", SITE, ")")

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach.png"),
       make_pred_plot(full_df, paste0(label_s2, " — Full Test Set"),
                      show_legend = TRUE),
       width = 12, height = 5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach.png"),
       make_pred_plot(head(full_df, 120), paste0(label_s2, " — First 120 Hours"),
                      show_legend = TRUE),
       width = 8, height = 4.5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach.png"),
       make_residual_hist(full_df, label_s2),
       width = 7, height = 10, dpi = 300)

ds_s2 <- compute_daily_stats(full_df)
cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
print_daily_stats(ds_s2, label_s2)
cat("  Saved all outputs (Beach)\n")

# =============================================================================
# SCENARIO 3: Judean Desert (Tzeelim) — Rock, Bush
# =============================================================================
cat("\n=== Scenario 3: Judean Desert (Tzeelim) ===\n")

SITE_COL_3 <- "site_id"
DATA_PATH  <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- "scenario_3_desert_single_logger"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

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
  data <- load_prepared_csv_data(DATA_PATH,
                                  datetime_format = "%Y-%m-%d %H:%M:%S",
                                  includes_index  = TRUE)
  data <- data[data[[SITE_COL_3]] == task$site, ]
  stats_list[[task$name]] <- logger_temp_stats(data, task$title)

  splits <- split_train_val_test(data, train_pct = 0.75, val_pct = 0.125,
                                  block_days = 7, use_blocks = TRUE, seed = SEED)
  feature_cols <- get_feature_columns(splits$train)
  scaled       <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h      <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                               window_size = 2, ts_names_col = SITE_COL_3)
  rf_test        <- align_test_sets(splits$test, lstm_2h$test_dict,
                                    lstm_2h$index_info, SITE_COL_3)
  X_test_lstm    <- lstm_2h$test_dict$X
  base_test_lstm <- lstm_2h$test_dict$base_pred

  rf_bundle   <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds")))
  lstm_bundle <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_lstm_model.rds")))

  full_df <- build_pred_df(rf_test, rf_bundle$feature_cols, rf_bundle$model,
                            base_test_lstm, lstm_bundle$model, X_test_lstm)
  full_df <- full_df[order(full_df$time), ]

  is_first <- task$name == tasks_s3[[1]]$name
  temp_panels[[task$name]]    <- make_pred_plot(full_df, task$title,
                                                 show_legend = is_first)
  excerpt_panels[[task$name]] <- make_pred_plot(head(full_df, 120), task$title,
                                                 show_legend = is_first)
  hist_panels[[task$name]]    <- make_residual_hist(full_df, task$title)
  daily_ext[[task$name]]      <- compute_daily_stats(full_df)
}

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_desert.png"),
       grid.arrange(grobs = temp_panels, ncol = 1),
       width = 12, height = 9, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert.png"),
       grid.arrange(grobs = excerpt_panels, ncol = 2),
       width = 12, height = 5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert.png"),
       grid.arrange(grobs = hist_panels, ncol = 2),
       width = 12, height = 12, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
for (task in tasks_s3) print_daily_stats(daily_ext[[task$name]], task$title)
cat("  Saved all outputs (Desert)\n")

# =============================================================================
# SCENARIO 4: Beach Pooled
# =============================================================================
cat("\n=== Scenario 4: Beach Pooled ===\n")

DATA_PATH_B   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SCENARIO_DIR <- "scenario_4_beach_pooled"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")
SITE_COL_B    <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B, is_continuous_microhabitat = FALSE,
                                  datetime_format = "%Y-%m-%d %H:%M:%S",
                                  includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b      <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)
rf_bundle_b   <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_b <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

# Build LSTM windows over the full test set (all sites together)
scaled_b  <- lstm_scaling(splits_b$train, splits_b$val, splits_b$test)
lstm_2h_b <- lstm_specific_preprocessing(scaled_b$train, scaled_b$val, scaled_b$test,
                                          window_size = 2, ts_names_col = SITE_COL_B)
rf_test_b       <- align_test_sets(splits_b$test, lstm_2h_b$test_dict,
                                    lstm_2h_b$index_info, SITE_COL_B)
X_test_lstm_b   <- lstm_2h_b$test_dict$X
base_test_lstm_b <- lstm_2h_b$test_dict$base_pred

# Build RF predictions (aligned to rf_test_b rows)
rf_preds_b <- rf_test_b$predicted +
  ranger:::predict.ranger(rf_bundle_b$model,
    data = as.data.frame(rf_test_b[, rf_bundle_b$feature_cols]))$predictions

# Build LSTM predictions site by site using index_info, then map back to rf_test_b rows
lstm_preds_b <- rep(NA_real_, nrow(rf_test_b))
all_lstm_raw <- as.numeric(predict(lstm_bundle_b$model, X_test_lstm_b, verbose=0)[, 1])

for (i in seq_along(lstm_2h_b$index_info$datasets)) {
  site_name <- lstm_2h_b$index_info$datasets[i]
  # 0-based LSTM indices → 1-based
  lstm_idx  <- lstm_2h_b$index_info$test_indices[[i]] + 1L
  # corresponding rf_test rows for this site
  rf_idx    <- which(rf_test_b[[SITE_COL_B]] == site_name)
  if (length(lstm_idx) != length(rf_idx))
    warning(sprintf("Site %s: LSTM rows %d != RF rows %d", site_name,
                    length(lstm_idx), length(rf_idx)))
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
full_dfs_b     <- list()

for (site in sites_b) {
  mask    <- rf_test_b[[SITE_COL_B]] == site
  site_df <- full_df_b[mask, ]
  site_df <- site_df[order(site_df$time), ]
  full_dfs_b[[site]] <- site_df

  site_data <- data_b[data_b[[SITE_COL_B]] == site, ]
  stats_list[[site]] <- logger_temp_stats(site_data, site)

  temp_panels[[site]]    <- make_pred_plot(site_df, site, show_legend = TRUE)
  excerpt_panels[[site]] <- make_pred_plot(head(site_df, 120), site,
                                            show_legend = TRUE)
  daily_ext[[site]]      <- compute_daily_stats(site_df)

  xlim_site <- range(c(site_df$measured - site_df$base,
                        site_df$measured - site_df$rf,
                        site_df$measured - site_df$lstm), na.rm = TRUE)
  hist_panels[[site]] <- make_residual_hist(site_df, site, xlim = xlim_site,
                                             show_strip = (site == sites_b[length(sites_b)]))
}

ncols_b <- min(2, length(sites_b))

grid_temp <- list(
  temp_panels[["Range_24 25 m"]], temp_panels[["Rosh_HaNikra 15 m"]], temp_panels[["Ashkelon 10 m"]],
  temp_panels[["Range_24 45 m"]], temp_panels[["Rosh_HaNikra 25 m"]], temp_panels[["Ashkelon 15 m"]],
  NULL,                           temp_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = grid_temp, ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 12, dpi = 300)

grid_excerpt <- list(
  excerpt_panels[["Range_24 25 m"]], excerpt_panels[["Rosh_HaNikra 15 m"]], excerpt_panels[["Ashkelon 10 m"]],
  excerpt_panels[["Range_24 45 m"]], excerpt_panels[["Rosh_HaNikra 25 m"]], excerpt_panels[["Ashkelon 15 m"]],
  NULL,                              excerpt_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = grid_excerpt, ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 12, dpi = 300)

grid_hist <- list(
  hist_panels[["Range_24 25 m"]], hist_panels[["Rosh_HaNikra 15 m"]], hist_panels[["Ashkelon 10 m"]],
  hist_panels[["Range_24 45 m"]], hist_panels[["Rosh_HaNikra 25 m"]], hist_panels[["Ashkelon 15 m"]],
  NULL,                           hist_panels[["Rosh_HaNikra 45 m"]], NULL
)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach_pooled.png"),
       ggpubr::ggarrange(plotlist = grid_hist, ncol = 3, nrow = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 15, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
for (site in sites_b) print_daily_stats(daily_ext[[site]], site)
cat("  Saved all outputs (Beach Pooled)\n")

# =============================================================================
# SCENARIO 5: Beach Specialized — per-location models
# =============================================================================
cat("\n=== Scenario 5: Beach Specialized ===\n")

SCENARIO_DIR <- "scenario_5_beach_specialized"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

# splits_b / data_b / SITE_COL_B reused from Scenario 4
locs_s5        <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()
all_panel_keys <- character(0)

for (loc in locs_s5) {
  rf_bundle_loc   <- load_correction_model(
    file.path(RESULTS_DIR, paste0("rf_",   loc, "_model.rds")))
  lstm_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("lstm_", loc, "_model.rds")))

  train_loc <- splits_b$train[splits_b$train$location == loc, ]
  val_loc   <- splits_b$val  [splits_b$val$location   == loc, ]
  test_loc  <- splits_b$test [splits_b$test$location  == loc, ]

  scaled_loc  <- lstm_scaling(train_loc, val_loc, test_loc)
  lstm_2h_loc <- lstm_specific_preprocessing(scaled_loc$train, scaled_loc$val,
                                              scaled_loc$test,
                                              window_size = 2,
                                              ts_names_col = SITE_COL_B)
  rf_test_loc       <- align_test_sets(test_loc, lstm_2h_loc$test_dict,
                                        lstm_2h_loc$index_info, SITE_COL_B)
  X_test_lstm_loc   <- lstm_2h_loc$test_dict$X
  base_test_lstm_loc <- lstm_2h_loc$test_dict$base_pred

  rf_preds_loc <- rf_test_loc$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model,
      data = as.data.frame(rf_test_loc[, rf_bundle_loc$feature_cols]))$predictions
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
    all_panel_keys <- c(all_panel_keys, panel_key)
    is_first  <- (loc == locs_s5[1]) && (site == sites_loc[1])

    temp_panels[[panel_key]]    <- make_pred_plot(site_df, site, show_legend = is_first)
    excerpt_panels[[panel_key]] <- make_pred_plot(head(site_df, 120), site,
                                                   show_legend = is_first)
    daily_ext[[panel_key]]      <- compute_daily_stats(site_df)

    xlim_site <- range(c(site_df$measured - site_df$base,
                          site_df$measured - site_df$rf,
                          site_df$measured - site_df$lstm), na.rm = TRUE)
    is_last <- panel_key == tail(all_panel_keys, 1)
    hist_panels[[panel_key]] <- make_residual_hist(site_df, site,
                                                    xlim = xlim_site,
                                                    show_strip = is_last)
  }
}

ncols_s5 <- 2

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_beach_specialized.png"),
       grid.arrange(grobs = temp_panels, ncol = ncols_s5),
       width = 14, height = ceiling(length(temp_panels) / ncols_s5) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_specialized.png"),
       grid.arrange(grobs = excerpt_panels, ncol = ncols_s5),
       width = 14, height = ceiling(length(excerpt_panels) / ncols_s5) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach_specialized.png"),
       grid.arrange(grobs = hist_panels, ncol = ncols_s5),
       width = 14, height = ceiling(length(hist_panels) / ncols_s5) * 10, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat("\nDaily min / mean / max — RMSE, ME, SD (°C):\n")
for (key in names(daily_ext)) print_daily_stats(daily_ext[[key]], key)
cat("  Saved all outputs (Beach Specialized)\n")

# =============================================================================
# SCENARIO 6: Desert Pooled — RF only
# =============================================================================
cat("\n=== Scenario 6: Desert Pooled ===\n")

DATA_PATH_D   <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_D <- system.file("extdata", "desert_splits.csv", package = "microclCorr")
SCENARIO_DIR <- "scenario_6_desert_pooled"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")
SITE_COL_D    <- "site_id"

data_d <- load_prepared_csv_data(DATA_PATH_D,
                                  datetime_format = "%Y-%m-%d %H:%M:%S",
                                  includes_index  = TRUE)
splits_d    <- load_splits_from_csv(data_d, SPLITS_PATH_D, SITE_COL_D)
rf_bundle_d   <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_d <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

# Build LSTM windows over the full desert test set
scaled_d  <- lstm_scaling(splits_d$train, splits_d$val, splits_d$test)
lstm_2h_d <- lstm_specific_preprocessing(scaled_d$train, scaled_d$val, scaled_d$test,
                                          window_size = 2, ts_names_col = SITE_COL_D)
rf_test_d        <- align_test_sets(splits_d$test, lstm_2h_d$test_dict,
                                     lstm_2h_d$index_info, SITE_COL_D)
X_test_lstm_d    <- lstm_2h_d$test_dict$X
base_test_lstm_d <- lstm_2h_d$test_dict$base_pred

rf_preds_d <- rf_test_d$predicted +
  ranger:::predict.ranger(rf_bundle_d$model,
    data = as.data.frame(rf_test_d[, rf_bundle_d$feature_cols]))$predictions

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
    is_first <- site == sites_d_sample[1]
    temp_panels[[site]]    <- make_pred_plot(full_df, site, show_legend = is_first)
    excerpt_panels[[site]] <- make_pred_plot(head(full_df, 120), site,
                                              show_legend = is_first)
    xlim_site <- range(c(full_df$measured - full_df$base,
                          full_df$measured - full_df$rf,
                          full_df$measured - full_df$lstm), na.rm = TRUE)
    hist_panels[[site]] <- make_residual_hist(full_df, site, xlim = xlim_site,
                                               show_strip = (site == sites_d_sample[length(sites_d_sample)]))
  }
}

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_desert_pooled.png"),
       grid.arrange(grobs = temp_panels, ncol = 2),
       width = 14, height = ceiling(length(temp_panels) / 2) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert_pooled.png"),
       grid.arrange(grobs = excerpt_panels, ncol = 2),
       width = 14, height = ceiling(length(excerpt_panels) / 2) * 4, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert_pooled.png"),
       grid.arrange(grobs = hist_panels, ncol = 2),
       width = 14, height = ceiling(length(hist_panels) / 2) * 8, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat(sprintf("\n  %d loggers total\n", length(sites_d)))
cat("\nDaily min / mean / max — RMSE, ME, SD (°C) [sample]:\n")
for (site in sites_d_sample) print_daily_stats(daily_ext[[site]], site)
cat("  Saved all outputs (Desert Pooled)\n")

# =============================================================================
# SCENARIO 7: Desert Specialized — per-region RF models
# =============================================================================
cat("\n=== Scenario 7: Desert Specialized ===\n")

SCENARIO_DIR <- "scenario_7_desert_specialized"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

# splits_d / data_d / SITE_COL_D reused from Scenario 6
locs_s7        <- c("Mishmar", "Tzeelim")
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()
sample_sites   <- character(0)

for (loc in locs_s7) {
  rf_bundle_loc   <- load_correction_model(
    file.path(RESULTS_DIR, paste0("rf_",   loc, "_model.rds")))
  lstm_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("lstm_", loc, "_model.rds")))

  train_reg <- splits_d$train[splits_d$train$Location == loc, ]
  val_reg   <- splits_d$val  [splits_d$val$Location   == loc, ]
  test_reg  <- splits_d$test [splits_d$test$Location  == loc, ]

  scaled_reg  <- lstm_scaling(train_reg, val_reg, test_reg)
  lstm_2h_reg <- lstm_specific_preprocessing(scaled_reg$train, scaled_reg$val,
                                              scaled_reg$test,
                                              window_size = 2,
                                              ts_names_col = SITE_COL_D)
  rf_test_reg        <- align_test_sets(test_reg, lstm_2h_reg$test_dict,
                                         lstm_2h_reg$index_info, SITE_COL_D)
  X_test_lstm_reg    <- lstm_2h_reg$test_dict$X
  base_test_lstm_reg <- lstm_2h_reg$test_dict$base_pred

  rf_preds_reg <- rf_test_reg$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model,
      data = as.data.frame(rf_test_reg[, rf_bundle_loc$feature_cols]))$predictions
  lstm_raw_reg   <- as.numeric(predict(lstm_bundle_loc$model, X_test_lstm_reg, verbose=0)[, 1])
  lstm_preds_reg <- rep(NA_real_, nrow(rf_test_reg))
  for (i in seq_along(lstm_2h_reg$index_info$datasets)) {
    sn <- lstm_2h_reg$index_info$datasets[i]
    li <- lstm_2h_reg$index_info$test_indices[[i]] + 1L
    ri <- which(rf_test_reg[[SITE_COL_D]] == sn)
    if (length(li) == length(ri))
      lstm_preds_reg[ri] <- base_test_lstm_reg[li] + lstm_raw_reg[li]
  }
  full_df_reg <- data.frame(
    time     = rf_test_reg$time,
    measured = rf_test_reg$predicted + rf_test_reg$residual,
    base     = rf_test_reg$predicted,
    rf       = rf_preds_reg,
    lstm     = lstm_preds_reg
  )
  full_df_reg[[SITE_COL_D]] <- rf_test_reg[[SITE_COL_D]]

  sites_loc <- unique(rf_test_reg[[SITE_COL_D]])
  sample_loc <- head(sites_loc, 3)
  sample_sites <- c(sample_sites, sample_loc)

  for (site in sites_loc) {
    mask    <- rf_test_reg[[SITE_COL_D]] == site
    full_df <- full_df_reg[mask, ]
    full_df <- full_df[order(full_df$time), ]

    site_data <- data_d[data_d[[SITE_COL_D]] == site, ]
    stats_list[[site]] <- logger_temp_stats(site_data, site)
    daily_ext[[site]]  <- compute_daily_stats(full_df)

    if (site %in% sample_loc) {
      clean_names <- c(
        "Bush_S_M_1_W" = "Small bush",
        "Bush_M_M_1_W" = "Medium bush",
        "Rock_L_M_1_W" = "Large rock",
        "Bush_M_T_1_W" = "Medium bush",
        "Rock_M_T_1_W" = "Medium rock",
        "Rock_L_T_2_W" = "Large rock"
      )
      panel_title <- clean_names[site]
      if (is.na(panel_title)) panel_title <- site
      
      temp_panels[[site]]    <- make_pred_plot(full_df, panel_title, show_legend = TRUE)
      excerpt_panels[[site]] <- make_pred_plot(head(full_df, 120), panel_title, show_legend = TRUE)
      
      xlim_site <- range(c(full_df$measured - full_df$base,
                            full_df$measured - full_df$rf,
                            full_df$measured - full_df$lstm), na.rm = TRUE)
      hist_panels[[site]] <- make_residual_hist(full_df, panel_title,
                                                 xlim = xlim_site,
                                                 show_strip = TRUE)
    }
  }
}

arrange_with_rows <- function(panels) {
  leg <- ggpubr::get_legend(panels[[1]])
  
  row1 <- ggpubr::ggarrange(plotlist = panels[1:3], ncol = 3, nrow = 1, legend = "none")
  row1 <- ggpubr::annotate_figure(row1, right = ggpubr::text_grob("Mishmar", rot = 270, face = "bold", size = 20))
  
  row2 <- ggpubr::ggarrange(plotlist = panels[4:6], ncol = 3, nrow = 1, legend = "none")
  row2 <- ggpubr::annotate_figure(row2, right = ggpubr::text_grob("Zeelim", rot = 270, face = "bold", size = 20))
  
  ggpubr::ggarrange(row1, row2, leg, ncol = 1, nrow = 3, heights = c(1, 1, 0.1))
}

grid_s7_temp <- list(
  temp_panels[["Bush_S_M_1_W"]], temp_panels[["Bush_M_M_1_W"]], temp_panels[["Rock_L_M_1_W"]],
  temp_panels[["Bush_M_T_1_W"]], temp_panels[["Rock_M_T_1_W"]], temp_panels[["Rock_L_T_2_W"]]
)

ggsave(file.path(SCENARIO_DIR, "temporal_predictions_desert_specialized.png"),
       arrange_with_rows(grid_s7_temp),
       width = 15, height = 8, dpi = 300)

grid_s7_excerpt <- list(
  excerpt_panels[["Bush_S_M_1_W"]], excerpt_panels[["Bush_M_M_1_W"]], excerpt_panels[["Rock_L_M_1_W"]],
  excerpt_panels[["Bush_M_T_1_W"]], excerpt_panels[["Rock_M_T_1_W"]], excerpt_panels[["Rock_L_T_2_W"]]
)

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert_specialized.png"),
       arrange_with_rows(grid_s7_excerpt),
       width = 15, height = 8, dpi = 300)

grid_s7_hist <- list(
  hist_panels[["Bush_S_M_1_W"]], hist_panels[["Bush_M_M_1_W"]], hist_panels[["Rock_L_M_1_W"]],
  hist_panels[["Bush_M_T_1_W"]], hist_panels[["Rock_M_T_1_W"]], hist_panels[["Rock_L_T_2_W"]]
)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert_specialized.png"),
       arrange_with_rows(grid_s7_hist),
       width = 15, height = 10, dpi = 300)

stats_df <- do.call(rbind, stats_list)
write.csv(stats_df, file.path(RESULTS_DIR, "logger_temp_stats.csv"), row.names = FALSE)

cat("\nDaily min / mean / max — RMSE, ME, SD (°C) [sample]:\n")
for (site in sample_sites) print_daily_stats(daily_ext[[site]], site)
cat("  Saved all outputs (Desert Specialized)\n")

# =============================================================================
# SCENARIO 8: Zero-Shot Transfer — RF only, no reloadable per-strategy models
# =============================================================================
cat("\n=== Scenario 8: Zero-Shot Transfer ===\n")

SCENARIO_DIR <- "scenario_8_zero_shot_transfer"
RESULTS_DIR <- file.path("../../NichMapR_ml_corr/microcl_ml_corr/inst/examples", SCENARIO_DIR, "results")

# Residual histogram: before correction only (no per-strategy saved models)
# Use beach test split residuals (splits_b reused from S4/S5)
full_df_s8_base <- data.frame(
  measured = splits_b$test$predicted + splits_b$test$residual,
  base     = splits_b$test$predicted,
  rf       = splits_b$test$predicted   # placeholder — no single model for S8
)
# For S8 only show NicheMapR before vs zero-shot strategy A residuals per location
# Re-derive corrected predictions from saved results CSV (no model files to reload)
res8 <- read.csv(file.path(RESULTS_DIR, "zero_shot_results.csv"))

# S8: no per-strategy models saved — show NicheMapR residuals only (1 facet)
res_base_s8 <- splits_b$test$residual  # measured - NicheMapR (already the residual column)
df_s8 <- data.frame(residual = res_base_s8, model = "NicheMapR (before)")
df_s8$model <- factor(df_s8$model, levels = "NicheMapR (before)")
mn_s8 <- round(mean(res_base_s8), 2); sd_s8 <- round(sd(res_base_s8), 2)
p_s8 <- ggplot2::ggplot(df_s8, ggplot2::aes(x = residual)) +
  ggplot2::geom_histogram(bins = 50, fill = "#ef4444", colour = "white", alpha = 0.85) +
  ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.7, colour = "#333333") +
  ggplot2::annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, size = 3.2,
                    label = sprintf("mean %+.2f °C\nSD %.2f °C", mn_s8, sd_s8)) +
  ggplot2::labs(title = "Beach — NicheMapR Residuals (Test Set, All Locations)\n(No after-correction: per-strategy models not saved; see run_scenario_8.R)",
                x = "Residual: Measured − NicheMapR (°C)", y = "Count") +
  ggplot2::theme_minimal(base_size = 10) +
  ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 9),
                 panel.grid.minor = ggplot2::element_blank())
ggsave(file.path(SCENARIO_DIR, "residual_histogram_zero_shot.png"),
       p_s8, width = 7, height = 4.5, dpi = 300)

# Re-generate the bar chart from saved results
res8$strategy <- factor(res8$strategy,
  levels = c("B: Specialized (Local Data)", "C: Pooled (All Sites)",
             "D: Pooled (Downsampled to N)", "A: Zero-Shot (Nearby Sites)"))

p8 <- ggplot(res8, aes(x = target, y = rmse_corr, fill = strategy)) +
  geom_bar(stat = "identity", position = position_dodge(0.85), width = 0.75) +
  geom_hline(aes(yintercept = rmse_base), linetype = "dashed",
             color = "#ef4444", linewidth = 0.7) +
  scale_fill_manual(values = c(
    "B: Specialized (Local Data)"    = "#10b981",
    "C: Pooled (All Sites)"          = "#3b82f6",
    "D: Pooled (Downsampled to N)"   = "#8b5cf6",
    "A: Zero-Shot (Nearby Sites)"    = "#f59e0b")) +
  labs(title    = "Zero-Shot Transfer: Can we correct an unseen location?",
       subtitle = "Red dashed line = uncorrected NicheMapR error",
       x        = "Target Location (held out during training)",
       y        = "Corrected RMSE (°C)",
       fill     = "Training Strategy") +
  theme_minimal(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "#666666"),
        legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2))

ggsave(file.path(SCENARIO_DIR, "zero_shot_transfer.png"),
       p8, width = 10, height = 6, dpi = 300)

cat("  Saved residual histogram + zero-shot bar chart (S8)\n")
cat("  Note: S8 histogram shows NicheMapR residuals only — no reloadable\n")
cat("        per-strategy models; see run_scenario_8.R to re-train.\n")

cat("\n=== All done ===\n")
