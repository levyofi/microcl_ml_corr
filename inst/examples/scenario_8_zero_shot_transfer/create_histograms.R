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
library(ranger)

SEED <- 123
DATA_PATH_B   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SCENARIO_DIR  <- file.path(root, "inst", "examples", "scenario_8_zero_shot_transfer")
RESULTS_DIR   <- file.path(SCENARIO_DIR, "results")
SITE_COL_B    <- "time_series_site"

data_b <- load_prepared_csv_data(DATA_PATH_B, is_continuous_microhabitat = FALSE,
                                  datetime_format = "%Y-%m-%d %H:%M:%S",
                                  includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data_b)) data_b$microhabitat_sun <- NULL
splits_b      <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)

feature_cols <- get_feature_columns(splits_b$train)

targets <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
hist_panels <- list()

for (target in targets) {
  test_loc  <- splits_b$test[splits_b$test$location == target, ]
  
  # Load the zero-shot model that was trained on the OTHER locations
  model_path <- file.path(RESULTS_DIR, paste0("rf_zeroshot_", target, "_model.rds"))
  rf_bundle <- load_correction_model(model_path)
  
  # Predict on this target location
  zs_preds <- predict(rf_bundle$model, data = test_loc[, feature_cols])$predictions
  
  full_df_loc <- data.frame(
    measured = test_loc$predicted + test_loc$residual,
    base     = test_loc$predicted,
    rf       = test_loc$predicted + zs_preds
  )
  
  title_str <- paste0("Tested on ", gsub("_", " ", target))
  hist_panels[[target]] <- make_residual_hist(full_df_loc, title_str, has_lstm = FALSE)
}

       p_arranged <- ggpubr::ggarrange(plotlist = hist_panels, labels = paste0("(", letters[1:length(hist_panels)], ")"), label.x = 0.15, label.y = 0.9, font.label = list(size = 24), ncol = 3, nrow = 1, common.legend = TRUE, legend = "top")

ggsave(file.path(SCENARIO_DIR, "residual_histogram_zero_shot.jpg"),
       p_arranged, width = 15, height = 5.5, bg = "white", dpi = 300)
cat("Scenario 8 histograms created successfully!\n")
