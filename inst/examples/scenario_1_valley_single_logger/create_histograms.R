# Find project root
if (file.exists("R/utils.R")) {
  root <- "."
} else if (file.exists("../../../R/utils.R")) {
  root <- "../../..."
} else {
  root <- "../.."
}

Sys.setenv(UV_OFFLINE="1", KERAS_HOME=normalizePath(file.path(root)))
source(file.path(root, "R/utils.R"))
source(file.path(root, "inst/examples/utils.R"))
setup_tensorflow()
library(reticulate)
py_require("tensorflow")
library(microclCorr)
library(ggplot2)
library(ggpubr)

SEED <- 11
DATA_PATH    <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_1_valley_single_logger")
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

tasks_s1 <- list(
  list(name = "harod2_sun", site = "harod2_sun.csv", title = "Valley - Sun"),
  list(name = "harod2_shd", site = "harod2_shd.csv", title = "Valley - Shade"),
  list(name = "harod2_air", site = "harod2_air.csv", title = "Valley - Air")
)

full_dfs <- list()
hist_panels <- list()

for (task in tasks_s1) {
  data <- load_prepared_csv_data(DATA_PATH, is_continuous_microhabitat = FALSE,
                                  datetime_format = "%d/%m/%Y %H:%M",
                                  includes_index  = TRUE)
  data <- data[data$time_series_doc == task$site, ]

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
}

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

       p_hist <- ggarrange(plotlist = hist_panels, labels = paste0("(", letters[1:length(hist_panels)], ")"), label.x = 0.15, label.y = 0.9, font.label = list(size = 24), ncol = 3, nrow = 1, common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "residual_histogram_valley.jpg"),
       p_hist, width = 15, height = 5.5, bg = "white", dpi = 300)
cat("Scenario 1 histograms created successfully!\n")
