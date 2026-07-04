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
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_5_beach_specialized")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")
SITE_COL_B <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B,
  is_continuous_microhabitat = FALSE,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index = TRUE
)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)

locs_s5 <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
excerpt_panels <- list()
all_panel_keys <- character(0)

for (loc in locs_s5) {
  rf_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("rf_", loc, "_model.rds"))
  )
  lstm_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("lstm_", loc, "_model.rds"))
  )

  train_loc <- splits_b$train[splits_b$train$location == loc, ]
  val_loc <- splits_b$val[splits_b$val$location == loc, ]
  test_loc <- splits_b$test[splits_b$test$location == loc, ]

  scaled_loc <- lstm_scaling(train_loc, val_loc, test_loc)
  lstm_2h_loc <- lstm_specific_preprocessing(scaled_loc$train, scaled_loc$val,
    scaled_loc$test,
    window_size = 2,
    ts_names_col = SITE_COL_B
  )
  rf_test_loc <- align_test_sets(
    test_loc, lstm_2h_loc$test_dict,
    lstm_2h_loc$index_info, SITE_COL_B
  )
  X_test_lstm_loc <- lstm_2h_loc$test_dict$X
  base_test_lstm_loc <- lstm_2h_loc$test_dict$base_pred

  rf_preds_loc <- rf_test_loc$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model,
      data = as.data.frame(rf_test_loc[, rf_bundle_loc$feature_cols])
    )$predictions
  lstm_raw_loc <- as.numeric(predict(lstm_bundle_loc$model, X_test_lstm_loc, verbose = 0)[, 1])
  lstm_preds_loc <- rep(NA_real_, nrow(rf_test_loc))
  for (i in seq_along(lstm_2h_loc$index_info$datasets)) {
    sn <- lstm_2h_loc$index_info$datasets[i]
    li <- lstm_2h_loc$index_info$test_indices[[i]] + 1L
    ri <- which(rf_test_loc[[SITE_COL_B]] == sn)
    if (length(li) == length(ri)) {
      lstm_preds_loc[ri] <- base_test_lstm_loc[li] + lstm_raw_loc[li]
    }
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
    mask <- rf_test_loc[[SITE_COL_B]] == site
    site_df <- full_df_loc[mask, ]
    site_df <- site_df[order(site_df$time), ]

    panel_key <- paste(loc, site, sep = "|")
    all_panel_keys <- c(all_panel_keys, panel_key)
    is_last <- panel_key == tail(all_panel_keys, 1)
    title_str <- gsub(" (\\d+) m$", ", \\1 m distance from water", site)
    title_str <- gsub("_", " ", title_str)
    excerpt_panels[[panel_key]] <- make_pred_plot(head(site_df, 120), title_str, show_legend = TRUE)
  }
}

ncols_s5 <- 2
p_excerpt <- ggarrange(plotlist = excerpt_panels, labels = paste0("(", letters[1:length(excerpt_panels)], ")"), label.x = 0.10, label.y = 0.9, font.label = list(size = 24), ncol = ncols_s5, nrow = ceiling(length(excerpt_panels) / ncols_s5), common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "prediction_examples_beach_specialized.jpg"),
  p_excerpt,
  width = 15, height = ceiling(length(excerpt_panels) / ncols_s5) * 5.5, bg = "white", dpi = 300
)
cat("Scenario 5 prediction examples created successfully!\n")
