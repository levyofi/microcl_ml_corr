# scenario_2_beach_single_logger/plot_scenario_2.R
# Diagnostic plot generation for Scenario 2: Coastal Beach (Ashkelon 15 m)

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

cat("\n=== Scenario 2: Coastal Beach (Ashkelon 15 m) ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_2_beach_single_logger")) "scenario_2_beach_single_logger" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

SITE      <- "Ashkelon 15 m"
SITE_COL  <- "time_series_site"
DATA_PATH <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")

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

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach.png"),
       make_pred_plot(head(full_df, 96), paste0(label_s2, " — First 120 Hours"),
                      show_legend = TRUE),
       width = 8, height = 4.5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach.png"),
       make_residual_hist(full_df, label_s2, strip_text_size = 11),
       width = 7, height = 10, dpi = 300)

ds_s2 <- compute_daily_stats(full_df)
cat("  Saved all outputs (Scenario 2 Beach)\n")
