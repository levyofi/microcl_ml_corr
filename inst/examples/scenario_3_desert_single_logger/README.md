# Scenario 3: Judean Desert Habitat (Tzeelim)

Local microclimate correction for loggers placed under a desert bush and on a desert rock
at the Tzeelim site in the Judean Desert.
Desert environments have high daily meteorological consistency,
making them the easiest habitat for the correction models.

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
source(system.file("examples", "scenario_3_desert_single_logger", "run_scenario_3.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for the Tzeelim desert site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | Per-microhabitat CSV with RMSE before and after correction |
| `prediction_examples_desert.png` | 120-hour observed vs. corrected predictions |
| `scenario_3_report.md` | Full results report |

## Key result

Average RMSE reduced from ~6.4 °C (NicheMapR baseline) to ~1.9 °C for both models (~71% improvement).
RF leads on Rock; LSTM leads on Bush. To find the minimum training days needed,
use `find_min_training_days()` in `examples_utility_functions.R`.
