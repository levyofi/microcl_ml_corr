# Scenario 2: Coastal Beach Habitat (Ashkelon)

Local microclimate correction for a single logger at a coastal beach site.
Conditions are strongly influenced by marine winds and sea temperatures,
making this a more challenging correction target than inland habitats.

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
source(system.file("examples", "scenario_2_beach_single_logger", "run_scenario_2.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for the Ashkelon beach site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | CSV with RMSE and R² before and after correction |
| `prediction_examples_beach.png` | 120-hour observed vs. corrected predictions |
| `scenario_2_report.md` | Full results report |

## Key result

RMSE reduced from ~11.9 °C (NicheMapR baseline) to ~1.4 °C (RF, 88%) and ~2.5 °C (LSTM, 79%).
RF outperforms LSTM at this training set size.
To explore how many days of data are needed, use `find_min_training_days()` in `examples_utility_functions.R`.
