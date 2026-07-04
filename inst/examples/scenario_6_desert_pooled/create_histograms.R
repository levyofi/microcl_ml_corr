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
SITE_COL_D <- "site_id"
DATA_PATH_D <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_D <- system.file("extdata", "desert_splits.csv", package = "microclCorr")
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_6_desert_pooled")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")

data_d <- load_prepared_csv_data(DATA_PATH_D,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index  = TRUE
)
splits_d <- load_splits_from_csv(data_d, SPLITS_PATH_D, SITE_COL_D)
rf_bundle_d <- load_correction_model(file.path(RESULTS_DIR, "rf_pooled_model.rds"))
lstm_bundle_d <- load_correction_model(file.path(RESULTS_DIR, "lstm_pooled_model.rds"))

scaled_d <- lstm_scaling(splits_d$train, splits_d$val, splits_d$test)
lstm_2h_d <- lstm_specific_preprocessing(scaled_d$train, scaled_d$val, scaled_d$test,
  window_size = 2, ts_names_col = SITE_COL_D
)
rf_test_d <- align_test_sets(
  splits_d$test, lstm_2h_d$test_dict,
  lstm_2h_d$index_info, SITE_COL_D
)
X_test_lstm_d <- lstm_2h_d$test_dict$X
base_test_lstm_d <- lstm_2h_d$test_dict$base_pred

rf_preds_d <- rf_test_d$predicted +
  ranger:::predict.ranger(rf_bundle_d$model,
    data = as.data.frame(rf_test_d[, rf_bundle_d$feature_cols])
  )$predictions

lstm_preds_d <- rep(NA_real_, nrow(rf_test_d))
all_lstm_raw_d <- as.numeric(predict(lstm_bundle_d$model, X_test_lstm_d, verbose = 0)[, 1])

for (i in seq_along(lstm_2h_d$index_info$datasets)) {
  site_name <- lstm_2h_d$index_info$datasets[i]
  lstm_idx <- lstm_2h_d$index_info$test_indices[[i]] + 1L
  rf_idx <- which(rf_test_d[[SITE_COL_D]] == site_name)
  if (length(lstm_idx) == length(rf_idx)) {
    lstm_preds_d[rf_idx] <- base_test_lstm_d[lstm_idx] + all_lstm_raw_d[lstm_idx]
  }
}

full_df_d_all <- data.frame(
  time     = rf_test_d$time,
  measured = rf_test_d$predicted + rf_test_d$residual,
  base     = rf_test_d$predicted,
  rf       = rf_preds_d,
  lstm     = lstm_preds_d
)
full_df_d_all[[SITE_COL_D]] <- rf_test_d[[SITE_COL_D]]

sites_d <- unique(rf_test_d[[SITE_COL_D]])
sites_d_sample <- head(sites_d, 6)
hist_panels <- list()

for (site in sites_d) {
  mask <- rf_test_d[[SITE_COL_D]] == site
  full_df <- full_df_d_all[mask, ]
  full_df <- full_df[order(full_df$time), ]

  site_data <- data_d[data_d[[SITE_COL_D]] == site, ]
  title_str <- paste0(site_data$Location[1], ", ", site_data$Object[1], ", ", site_data$Size[1])


  if (site %in% sites_d_sample) {
    xlim_site <- range(c(
      full_df$measured - full_df$base,
      full_df$measured - full_df$rf,
      full_df$measured - full_df$lstm
    ), na.rm = TRUE)
    hist_panels[[site]] <- make_residual_hist(full_df, title_str,
      xlim = xlim_site,
      show_strip = (site == sites_d_sample[length(sites_d_sample)]),
      show_legend = TRUE
    )
  }
}

p_hist <- ggarrange(plotlist = hist_panels, labels = paste0("(", letters[1:length(hist_panels)], ")"), label.x = 0.15, label.y = 0.9, font.label = list(size = 24), ncol = 2, nrow = ceiling(length(hist_panels) / 2), common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert_pooled.jpg"),
  p_hist,
  width = 10, height = ceiling(length(hist_panels) / 2) * 5.5, bg = "white", dpi = 300
)
cat("Scenario 6 histograms created successfully!\n")
