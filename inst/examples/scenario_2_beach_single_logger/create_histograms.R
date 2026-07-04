# Find project root
if (file.exists("R/utils.R")) {
  root <- "."
} else if (file.exists("../../../R/utils.R")) {
  root <- "../../..."
} else {
  root <- "../.."
}

Sys.setenv(UV_OFFLINE = "1", KERAS_HOME = normalizePath(file.path(root)))
source(file.path(root, "R/utils.R"))
source(file.path(root, "inst/examples/utils.R"))
setup_tensorflow()
library(reticulate)
py_require("tensorflow")
library(microclCorr)
library(ggplot2)
library(ggpubr)

SEED <- 123
SITE <- "Ashkelon 15 m"
SITE_COL <- "time_series_site"
DATA_PATH <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_2_beach_single_logger")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")

data <- load_prepared_csv_data(DATA_PATH,
  is_continuous_microhabitat = FALSE,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index = TRUE
)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL
data <- data[data[[SITE_COL]] == SITE &
  data$time > as.POSIXct("2025-05-25", tz = "UTC") &
  data$time < as.POSIXct("2025-08-26", tz = "UTC"), ]

splits <- split_train_val_test(data,
  train_pct = 0.75, val_pct = 0.125,
  block_days = 7, use_blocks = TRUE, seed = SEED
)
feature_cols <- get_feature_columns(splits$train)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)
lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
  window_size = 2, ts_names_col = SITE_COL
)
rf_test <- align_test_sets(
  splits$test, lstm_2h$test_dict,
  lstm_2h$index_info, SITE_COL
)
X_test_lstm <- lstm_2h$test_dict$X
base_test_lstm <- lstm_2h$test_dict$base_pred

rf_bundle <- load_correction_model(file.path(RESULTS_DIR, "rf_model.rds"))
lstm_bundle <- load_correction_model(file.path(RESULTS_DIR, "lstm_model.rds"))

full_df <- build_pred_df(
  rf_test, rf_bundle$feature_cols, rf_bundle$model,
  base_test_lstm, lstm_bundle$model, X_test_lstm
)
full_df <- full_df[order(full_df$time), ]

label_s2 <- paste0("Coastal Beach (", SITE, ")")
p_hist <- make_residual_hist(full_df, NULL, show_legend = TRUE) +
  ggplot2::guides(fill = ggplot2::guide_legend(nrow = 3, byrow = TRUE)) +
  ggplot2::theme(legend.text = ggplot2::element_text(size = 12), legend.key.size = ggplot2::unit(1, "lines"))

p_arranged <- ggpubr::ggarrange(p_hist, label.y = 0.9, font.label = list(size = 24), ncol = 1, nrow = 1, common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "residual_histogram_beach.jpg"),
  p_arranged,
  width = 5, height = 5.5, bg = "white", dpi = 300
)
cat("Scenario 2 histograms created successfully!\n")
