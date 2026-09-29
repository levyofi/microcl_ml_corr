# Scenario 1: Mediterranean Valley Habitat (Harod)

Local microclimate correction for a single logger in a Mediterranean valley.
A Random Forest and an LSTM model are each trained on logger data from the Harod site
and used to correct NicheMapR temperature predictions.

## Prerequisites

This scenario trains and evaluates an LSTM neural network in addition to Random Forest.
Running LSTM models requires Python with TensorFlow and Keras, as well as the R packages `reticulate`, `tensorflow`, and `keras3`.

It is not good practice to install software packages automatically without explicit user instruction. Before running this scenario for the first time, install and configure the required environment by running:

```r
library(microclCorr)
setup_tensorflow()
```

`setup_tensorflow()` detects an existing Python environment with TensorFlow or installs the required packages into a dedicated virtual environment (`microcl_env`).

## Run

Once TensorFlow is configured, run the scenario:

```r
# 1. Run the ML pipeline (trains models, evaluates correction, saves results)
source(system.file("examples", "scenario_1_valley_single_logger", "run_scenario_1.R", package = "microclCorr"))

# 2. (Optional) Generate diagnostic figures (requires ggplot2, gridExtra, ggpubr, cowplot)
source(system.file("examples", "scenario_1_valley_single_logger", "plot_scenario_1.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Harod_dataset.csv` | Pre-aligned logger + NicheMapR data for the Harod valley site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | Per-microhabitat CSV with RMSE before and after correction |
| `prediction_examples_valley.png` | 120-hour observed vs. corrected predictions |
| `scenario_1_report.md` | Full results report |

## Key result

Average RMSE across microhabitats reduced from ~5.2 °C (NicheMapR) to ~2.6 °C (RF, 42%)
and ~2.9 °C (LSTM, 39%). RF leads on Sun and Shade; LSTM leads on Air.
To explore how many days of data are needed, use `find_min_training_days()` in `examples_utility_functions.R`.
