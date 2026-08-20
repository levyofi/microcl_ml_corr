# =============================================================================
# generate_plots.R
# Master plot coordinator script. Calls per-scenario plotting scripts (1–8)
# using saved model .rds/.keras files — no re-training required.
# =============================================================================

Sys.setenv(UV_OFFLINE="1", KERAS_HOME=getwd())
source("package_utils.R")
source("utils.R")
library(keras3)
library(reticulate)
library(microclCorr)
library(ggplot2)
library(gridExtra)
library(cowplot)

SEED <- 123

scenarios <- c(
  "scenario_1_valley_single_logger/plot_scenario_1.R",
  "scenario_2_beach_single_logger/plot_scenario_2.R",
  "scenario_3_desert_single_logger/plot_scenario_3.R",
  "scenario_4_beach_pooled/plot_scenario_4.R",
  "scenario_5_beach_specialized/plot_scenario_5.R",
  "scenario_6_desert_pooled/plot_scenario_6.R",
  "scenario_7_desert_specialized/plot_scenario_7.R",
  "scenario_8_zero_shot_transfer/plot_scenario_8.R"
)

for (sc in scenarios) {
  cat(sprintf("\n=== Sourcing %s ===\n", sc))
  source(sc)
}

cat("\n=== All scenario plots generated successfully ===\n")
