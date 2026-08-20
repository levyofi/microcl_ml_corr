# scenario_8_zero_shot_transfer/plot_scenario_8.R
# Diagnostic plot generation for Scenario 8: Zero-Shot Transfer

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

cat("\n=== Scenario 8: Zero-Shot Transfer ===\n")
SCENARIO_DIR <- if (dir.exists("scenario_8_zero_shot_transfer")) "scenario_8_zero_shot_transfer" else "."
RESULTS_DIR  <- file.path(SCENARIO_DIR, "results")

DATA_PATH_B   <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH_B <- system.file("extdata", "beach_splits.csv", package = "microclCorr")
SITE_COL_B    <- "time_series_site"

data_b   <- load_prepared_csv_data(DATA_PATH_B, is_continuous_microhabitat = FALSE, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE)
splits_b <- load_splits_from_csv(data_b, SPLITS_PATH_B, SITE_COL_B)

locs_s8        <- c("Ashkelon", "Range_24", "Rosh_HaNikra")
temp_panels    <- list()
excerpt_panels <- list()
hist_panels    <- list()
daily_ext      <- list()
stats_list     <- list()

for (i in seq_along(locs_s8)) {
  loc <- locs_s8[i]
  rf_bundle_loc <- load_correction_model(file.path(RESULTS_DIR, paste0("rf_zeroshot_", loc, "_model.rds")))
  test_loc <- splits_b$test[splits_b$test$location == loc, ]
  
  rf_preds_loc <- test_loc$predicted +
    ranger:::predict.ranger(rf_bundle_loc$model, data = as.data.frame(test_loc[, rf_bundle_loc$feature_cols]))$predictions

  full_df_loc <- data.frame(
    time     = test_loc$time,
    measured = test_loc$predicted + test_loc$residual,
    base     = test_loc$predicted,
    rf       = rf_preds_loc
  )
  full_df_loc[[SITE_COL_B]] <- test_loc[[SITE_COL_B]]

  site_df <- aggregate(cbind(measured, base, rf) ~ time, data = full_df_loc, FUN = mean)
  site_df <- site_df[order(site_df$time), ]

  letter_val <- c("(a)", "(b)", "(c)")[i]

  temp_panels[[loc]]    <- make_pred_plot(head(site_df, 96), loc, show_legend = TRUE, has_lstm = FALSE, panel_letter = letter_val)
  excerpt_panels[[loc]] <- make_pred_plot(head(site_df, 96), loc, show_legend = TRUE, has_lstm = FALSE, panel_letter = letter_val)
  daily_ext[[loc]]      <- compute_daily_stats(full_df_loc)

  xlim_site <- range(c(full_df_loc$measured - full_df_loc$base, full_df_loc$measured - full_df_loc$rf), na.rm = TRUE)
  hist_panels[[loc]] <- make_residual_hist(full_df_loc, loc, has_lstm = FALSE, xlim = xlim_site, show_strip = TRUE, show_legend = TRUE, panel_letter = letter_val, strip_text_size = 11)
}

ggsave(file.path(SCENARIO_DIR, "prediction_examples_zero_shot.png"),
       ggpubr::ggarrange(plotlist = unname(excerpt_panels), ncol = 3, common.legend = TRUE, legend = "top"),
       width = 15, height = 5, dpi = 300)

ggsave(file.path(SCENARIO_DIR, "residual_histogram_zero_shot.png"),
       ggpubr::ggarrange(plotlist = unname(hist_panels), ncol = 3, common.legend = TRUE, legend = "bottom"),
       width = 15, height = 6, dpi = 300)

res8 <- read.csv(file.path(RESULTS_DIR, "zero_shot_results.csv"))
res8$strategy <- factor(res8$strategy,
  levels = c("B: Specialized (Local Data)", "C: Pooled (All Sites)",
             "D: Pooled (Downsampled to N)", "A: Zero-Shot (Nearby Sites)"))

p8 <- ggplot(res8, aes(x = target, y = rmse_corr, fill = strategy)) +
  geom_bar(stat = "identity", position = position_dodge(0.85), width = 0.75) +
  geom_hline(aes(yintercept = rmse_base), linetype = "dashed", color = "#dc2626", linewidth = 0.7) +
  scale_fill_manual(values = c(
    "B: Specialized (Local Data)"    = "#059669",
    "C: Pooled (All Sites)"          = "#2563eb",
    "D: Pooled (Downsampled to N)"   = "#8b5cf6",
    "A: Zero-Shot (Nearby Sites)"    = "#f59e0b")) +
  labs(title    = "Zero-Shot Spatial Transfer: Performance Across Target Locations",
       subtitle = "Dashed red line = uncorrected NicheMapR physical model baseline error",
       x        = "Target Location (Held-Out During Training)",
       y        = expression("Corrected RMSE (" * degree * "C)"),
       fill     = "Training Strategy") +
  theme_minimal(base_size = 14) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5, size = 16),
        plot.subtitle = element_text(hjust = 0.5, color = "#444444", size = 12),
        legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2))

ggsave(file.path(SCENARIO_DIR, "zero_shot_transfer.png"),
       p8, width = 10, height = 6, dpi = 300)

cat("  Saved all outputs (Scenario 8 Zero-Shot Transfer)\n")
