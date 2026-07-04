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
SCENARIO_DIR <- file.path(root, "inst", "examples", "scenario_7_desert_specialized")
RESULTS_DIR <- file.path(SCENARIO_DIR, "results")

data_d <- load_prepared_csv_data(DATA_PATH_D,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index = TRUE
)
splits_d <- load_splits_from_csv(data_d, SPLITS_PATH_D, SITE_COL_D)

locs_s7 <- c("Mishmar", "Tzeelim")
sample_sites <- character(0)
hist_panels <- list()

for (loc in locs_s7) {
  rf_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("rf_", loc, "_model.rds"))
  )
  lstm_bundle_loc <- load_correction_model(
    file.path(RESULTS_DIR, paste0("lstm_", loc, "_model.rds"))
  )

  train_reg <- splits_d$train[splits_d$train$Location == loc, ]
  val_reg <- splits_d$val[splits_d$val$Location == loc, ]
  test_reg <- splits_d$test[splits_d$test$Location == loc, ]

  scaled_reg <- lstm_scaling(train_reg, val_reg, test_reg)
  lstm_2h_reg <- lstm_specific_preprocessing(scaled_reg$train, scaled_reg$val,
    scaled_reg$test,
    window_size = 2,
    ts_names_col = SITE_COL_D
  )
  rf_test_reg <- align_test_sets(
    test_reg, lstm_2h_reg$test_dict,
    lstm_2h_reg$index_info, SITE_COL_D
  )
  X_test_lstm_reg <- lstm_2h_reg$test_dict$X
  base_test_lstm_reg <- lstm_2h_reg$test_dict$base_pred

  rf_preds_reg <- rf_test_reg$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model,
      data = as.data.frame(rf_test_reg[, rf_bundle_loc$feature_cols])
    )$predictions
  lstm_raw_reg <- as.numeric(predict(lstm_bundle_loc$model, X_test_lstm_reg, verbose = 0)[, 1])
  lstm_preds_reg <- rep(NA_real_, nrow(rf_test_reg))
  for (i in seq_along(lstm_2h_reg$index_info$datasets)) {
    sn <- lstm_2h_reg$index_info$datasets[i]
    li <- lstm_2h_reg$index_info$test_indices[[i]] + 1L
    ri <- which(rf_test_reg[[SITE_COL_D]] == sn)
    if (length(li) == length(ri)) {
      lstm_preds_reg[ri] <- base_test_lstm_reg[li] + lstm_raw_reg[li]
    }
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
    mask <- rf_test_reg[[SITE_COL_D]] == site
    full_df <- full_df_reg[mask, ]
    full_df <- full_df[order(full_df$time), ]

    site_data <- data_d[data_d[[SITE_COL_D]] == site, ]
    title_str <- paste0(site_data$Location[1], ", ", site_data$Object[1], ", ", site_data$Size[1])


    if (site %in% sample_loc) {
      panel_title <- paste0(site_data$Location[1], ", ", site_data$Object[1], ", ", site_data$Size[1])
      xlim_site <- range(c(
        full_df$measured - full_df$base,
        full_df$measured - full_df$rf,
        full_df$measured - full_df$lstm
      ), na.rm = TRUE)
      hist_panels[[site]] <- make_residual_hist(full_df, panel_title,
        xlim = xlim_site,
        show_strip = (site == tail(sample_loc, 1) &&
          loc == tail(locs_s7, 1)),
        show_legend = TRUE
      )
    }
  }
}

p_hist <- ggarrange(plotlist = hist_panels, labels = paste0("(", letters[1:length(hist_panels)], ")"), label.x = 0.15, label.y = 0.9, font.label = list(size = 24), ncol = 2, nrow = ceiling(length(hist_panels) / 2), common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "residual_histogram_desert_specialized.jpg"),
  p_hist,
  width = 10, height = ceiling(length(hist_panels) / 2) * 5.5, bg = "white", dpi = 300
)
cat("Scenario 7 histograms created successfully!\n")
