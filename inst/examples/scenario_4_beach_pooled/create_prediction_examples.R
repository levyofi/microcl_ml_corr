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
DATA_PATH_B <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_4_beach_pooled")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")
SITE_COL_B <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B,
  is_continuous_microhabitat = FALSE,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index = TRUE
)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)
rf_bundle_b <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_b <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

scaled_b <- lstm_scaling(splits_b$train, splits_b$val, splits_b$test)
lstm_2h_b <- lstm_specific_preprocessing(scaled_b$train, scaled_b$val, scaled_b$test,
  window_size = 2, ts_names_col = SITE_COL_B
)
rf_test_b <- align_test_sets(
  splits_b$test, lstm_2h_b$test_dict,
  lstm_2h_b$index_info, SITE_COL_B
)
X_test_lstm_b <- lstm_2h_b$test_dict$X
base_test_lstm_b <- lstm_2h_b$test_dict$base_pred

rf_preds_b <- rf_test_b$predicted +
  ranger:::predict.ranger(rf_bundle_b$model,
    data = as.data.frame(rf_test_b[, rf_bundle_b$feature_cols])
  )$predictions

lstm_preds_b <- rep(NA_real_, nrow(rf_test_b))
all_lstm_raw <- as.numeric(predict(lstm_bundle_b$model, X_test_lstm_b, verbose = 0)[, 1])

for (i in seq_along(lstm_2h_b$index_info$datasets)) {
  site_name <- lstm_2h_b$index_info$datasets[i]
  lstm_idx <- lstm_2h_b$index_info$test_indices[[i]] + 1L
  rf_idx <- which(rf_test_b[[SITE_COL_B]] == site_name)
  if (length(lstm_idx) == length(rf_idx)) {
    lstm_preds_b[rf_idx] <- base_test_lstm_b[lstm_idx] + all_lstm_raw[lstm_idx]
  }
}

full_df_b <- data.frame(
  time     = rf_test_b$time,
  measured = rf_test_b$predicted + rf_test_b$residual,
  base     = rf_test_b$predicted,
  rf       = rf_preds_b,
  lstm     = lstm_preds_b
)
full_df_b[[SITE_COL_B]] <- rf_test_b[[SITE_COL_B]]

sites_b <- unique(rf_test_b[[SITE_COL_B]])
excerpt_panels <- list()

for (site in sites_b) {
  mask <- rf_test_b[[SITE_COL_B]] == site
  site_df <- full_df_b[mask, ]
  site_df <- site_df[order(site_df$time), ]
  title_str <- gsub(" (\\d+) m$", ", \\1 m distance from water", site)
  title_str <- gsub("_", " ", title_str)
  excerpt_panels[[site]] <- make_pred_plot(head(site_df, 120), title_str, show_legend = TRUE)
}

ncols_b <- min(2, length(sites_b))
p_excerpt <- ggarrange(plotlist = excerpt_panels, labels = paste0("(", letters[1:length(excerpt_panels)], ")"), label.x = 0.1, label.y = 0.9, font.label = list(size = 24), ncol = ncols_b, nrow = ceiling(length(excerpt_panels) / ncols_b), common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_pooled.jpg"),
  p_excerpt,
  width = 15, height = ceiling(length(excerpt_panels) / ncols_b) * 5.5, bg = "white", dpi = 300
)
cat("Scenario 4 prediction examples created successfully!\n")
