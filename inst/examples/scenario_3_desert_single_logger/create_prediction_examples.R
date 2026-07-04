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
SITE_COL_3 <- "site_id"
DATA_PATH <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_3_desert_single_logger")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")

tasks_s3 <- list(
  list(name = "Rock_S_T_2_W", site = "Rock_S_T_2_W", title = "Desert - Rock"),
  list(name = "Bush_S_T_2_W", site = "Bush_S_T_2_W", title = "Desert - Bush")
)

excerpt_panels <- list()

for (task in tasks_s3) {
  data <- load_prepared_csv_data(DATA_PATH,
    datetime_format = "%Y-%m-%d %H:%M:%S",
    includes_index  = TRUE
  )
  data <- data[data[[SITE_COL_3]] == task$site, ]

  splits <- split_train_val_test(data,
    train_pct = 0.75, val_pct = 0.125,
    block_days = 7, use_blocks = TRUE, seed = SEED
  )
  feature_cols <- get_feature_columns(splits$train)
  scaled <- lstm_scaling(splits$train, splits$val, splits$test)
  lstm_2h <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
    window_size = 2, ts_names_col = SITE_COL_3
  )
  rf_test <- align_test_sets(
    splits$test, lstm_2h$test_dict,
    lstm_2h$index_info, SITE_COL_3
  )
  X_test_lstm <- lstm_2h$test_dict$X
  base_test_lstm <- lstm_2h$test_dict$base_pred

  rf_bundle <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_rf_model.rds"))
  )
  lstm_bundle <- load_correction_model(
    file.path(RESULTS_DIR, paste0(task$name, "_lstm_model.rds"))
  )

  full_df <- build_pred_df(
    rf_test, rf_bundle$feature_cols, rf_bundle$model,
    base_test_lstm, lstm_bundle$model, X_test_lstm
  )
  full_df <- full_df[order(full_df$time), ]

  excerpt_panels[[task$name]] <- make_pred_plot(head(full_df, 120), task$title, show_legend = TRUE)
}

p_excerpt <- ggarrange(plotlist = excerpt_panels, labels = paste0("(", letters[1:length(excerpt_panels)], ")"), label.x = 0.15, label.y = 0.9, font.label = list(size = 24), ncol = 2, nrow = 1, common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "prediction_examples_desert.jpg"),
  p_excerpt,
  width = 10, height = 5.5, bg = "white", dpi = 300
)
cat("Scenario 3 prediction examples created successfully!\n")
