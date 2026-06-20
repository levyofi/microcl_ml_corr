# microclCorr: Machine Learning Correction of Microclimate Model Predictions

[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen.svg)](https://github.com/levyofi/microcl_ml_corr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `microclCorr` R package provides a robust, production-grade machine learning pipeline to correct systematic biases in physical microclimate models (such as **NicheMapR**) using temperature logger measurements. 

By combining physical microclimatic baseline simulations with Random Forests (`ranger`) and sequential Deep Learning models (`keras3` / `tensorflow`), `microclCorr` reduces simulation errors by **80% to 94%** across diverse Mediterranean, coastal, and desert environments.

---

## Key Features

- **Data Preprocessing & Scaling**: Robust datetime alignment (enforcing UTC), index matching, and sequence windowing.
- **Sequential Deep Learning**: Fully integrated LSTM networks for correction of temporal lag and shadow geometries.
- **Robust Random Forests**: Fast out-of-bag correction using optimized `ranger` ensembles.
- **Spatial Generalization Scenarios**: Built-in support for Pooled (multi-site) and Specialized (local) training pipelines.
- **Zero-Shot Spatial Transfer**: Capabilities to transfer corrections to unseen locations using neighboring site data.
- **Automated Hyperparameter Optimization (HPO)**: Grid search and keras-based validation tuning.

---

## Installation Instructions

You can install `microclCorr` directly from GitHub.

### 1. Install System and R Dependencies

Before installing, ensure you have a functional R environment (>= 4.0.0). Open R and run:

```R
# Install devtools if you haven't already
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install core CRAN dependencies
install.packages(c("dplyr", "lubridate", "ranger", "data.table", "reticulate"))
```

### 2. Install the Package from GitHub

Install the package directly from the repository:

```R
devtools::install_github("levyofi/microcl_ml_corr")
```

### 3. Configure TensorFlow & Keras Backend (For LSTM Models)

The sequential deep learning models rely on the Python `tensorflow` backend. We recommend configuring a dedicated Conda/Mamba environment:

```R
library(reticulate)

# Option A: Automatic installation via keras3 (creates a conda/virtualenv)
library(keras3)
install_keras()

# Option B: Link to an existing conda environment containing TensorFlow (Recommended for GPU support)
# Replace the path with your conda/mamba environment location
use_condaenv("microclimate_gpu", required = TRUE)
```

---

## Quick Start Example: Local Correction

Below is a simple example showing how to train a Random Forest correction model on a single location:

```R
library(microclCorr)
library(ranger)

# 1. Load your preprocessed logger data
# Columns required: 'time', 'predicted' (physical model), 'residual' (measured - predicted), and features
data_path <- system.file("extdata", "Harod_dataset.csv", package = "microclCorr")
data <- load_prepared_csv_data(data_path, is_continuous_microhabitat = FALSE)

# 2. Split data into train, validation, and test blocks (e.g. 7-day blocks)
splits <- split_train_val_test(
  data, 
  train_pct = 0.75, 
  val_pct = 0.125, 
  block_days = 7, 
  use_blocks = TRUE, 
  seed = 42
)

# 3. Identify physical features for training
features <- get_feature_columns(splits$train)

# 4. Train a Random Forest Correction Model
rf_model <- ranger::ranger(
  x = splits$train[, features, drop = FALSE],
  y = splits$train$residual,
  num.trees = 500,
  seed = 42
)

# 5. Correct predictions on the test set
test_X <- splits$test[, features, drop = FALSE]
predicted_residuals <- predict(rf_model, data = test_X)$predictions
corrected_predictions <- splits$test$predicted + predicted_residuals

# 6. Evaluate error reduction
rmse_base <- sqrt(mean(splits$test$residual^2))
rmse_corrected <- sqrt(mean((splits$test$residual - predicted_residuals)^2))
cat(sprintf("Baseline RMSE: %.3f°C\nCorrected RMSE: %.3f°C\n", rmse_base, rmse_corrected))
```

---

## Package Scenarios & Examples

The package contains self-contained executable scenarios in `inst/examples/`:

- **Scenario 1**: Mediterranean Valley Habitat (Harod Local Correction)
- **Scenario 2**: Beach Habitat (Local Correction)
- **Scenario 3**: Judean Desert (Local Correction)
- **Scenario 4**: Beach Pooled Spatial Generalization
- **Scenario 5**: Beach Specialized Location-Specific Models
- **Scenario 6**: Desert Pooled Spatial Generalization (48 loggers)
- **Scenario 7**: Desert Specialized Models
- **Scenario 8**: Zero-Shot Spatial Transfer (Training on nearby sites to correct a new site)

To run any scenario, you can source the script directly:
```R
# Run Scenario 8: Zero-Shot Transfer
source(system.file("examples", "scenario_8_zero_shot_transfer", "run_scenario_8.R", package = "microclCorr"))
```

---

## License

This package is licensed under the MIT License. See the `LICENSE` file for details.
