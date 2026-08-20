# scenario_7_desert_specialized/plot_scenario_7.R
# Diagnostic plot generation for Scenario 7: Desert Specialized

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

cat("\n=== Scenario 7: Desert Specialized ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_7_desert_specialized")) "scenario_7_desert_specialized" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH_D   <- system.file("extdata", "desert_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_D <- system.file("extdata", "desert_splits.csv", package = "microclCorr")
SITE_COL_D    <- "site_id"

data_d  <- load_prepared_csv_data(DATA_PATH_D, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
splits_d <- load_splits_from_csv(data_d, SPLITS_PATH_D, SITE_COL_D)

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
      site_letters_s7 <- c(
        "Bush_S_M_1_W" = "(a)", "Bush_M_M_1_W" = "(b)", "Rock_L_M_1_W" = "(c)",
        "Bush_M_T_1_W" = "(d)", "Rock_M_T_1_W" = "(e)", "Rock_L_T_2_W" = "(f)"
      )
      panel_title <- clean_names[site]
      if (is.na(panel_title)) panel_title <- site
      let_val <- site_letters_s7[site]
      if (is.na(let_val)) let_val <- NULL
      
      temp_panels[[site]]    <- make_pred_plot(head(full_df, 96), panel_title, show_legend = TRUE, panel_letter = let_val)
      excerpt_panels[[site]] <- make_pred_plot(head(full_df, 96), panel_title, show_legend = TRUE, panel_letter = let_val)
      
      xlim_site <- range(c(full_df$measured - full_df$base,
                            full_df$measured - full_df$rf,
                            full_df$measured - full_df$lstm), na.rm = TRUE)
      hist_panels[[site]] <- make_residual_hist(full_df, panel_title,
                                                 xlim = xlim_site,
                                                 show_strip = TRUE,
                                                 strip_text_size = 11)
    }
  }
}

arrange_with_rows <- function(panels) {
  leg <- ggpubr::get_legend(panels[[1]])
  
  p_noleg <- lapply(panels, function(p) p + ggplot2::theme(legend.position = "none"))
  
  row1 <- ggpubr::ggarrange(plotlist = p_noleg[1:3], ncol = 3, nrow = 1, legend = "none")
  row1 <- ggpubr::annotate_figure(row1, right = ggpubr::text_grob("Mishmar", rot = 270, face = "bold", size = 20))
  
  row2 <- ggpubr::ggarrange(plotlist = p_noleg[4:6], ncol = 3, nrow = 1, legend = "none")
  row2 <- ggpubr::annotate_figure(row2, right = ggpubr::text_grob("Zeelim", rot = 270, face = "bold", size = 20))
  
  grid_body <- ggpubr::ggarrange(row1, row2, ncol = 1, nrow = 2)
  ggpubr::ggarrange(grid_body, leg, ncol = 1, heights = c(1, 0.1))
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
cat("  Saved all outputs (Scenario 7 Desert Specialized)\n")
