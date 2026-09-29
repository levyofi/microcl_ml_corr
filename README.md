# microclCorr: Correcting Timeseries Microclimate Model Predictions with Machine Learning

[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`microclCorr` is an R package that improves the accuracy of microclimate temperature predictions produced by physical models (such as **NicheMapR**).

Physical models are fundamental tools for understanding how species interact with their thermal environment. By simulating temperatures from first principles — solar radiation, wind, terrain geometry, and heat transfer — they provide mechanistic insight that purely statistical approaches cannot. However, in microhabitats with complex heat-balance conditions not yet fully captured by current physical models, or poor parameterizations due to low-resolution input data, a residual gap remains between model predictions and field measurements.

`microclCorr` bridges that gap using machine learning. Rather than modifying or replacing the physical model, it learns to predict and correct the residual error from a small set of temperature logger measurements. This eliminates the need to manually re-parameterise the physical model for each specific microhabitat. The package includes case studies where ML reduced prediction errors by **58% to 90%** across Mediterranean, coastal, and desert environments.

---

## How it works

A physical model predicts a temperature. A field logger measures the actual temperature. The difference between the two is called the **residual**:

```
residual = measured temperature − physical model prediction
```

`microclCorr` trains a model to predict that residual. The corrected temperature is then:

```
corrected temperature = physical model prediction + predicted residual
```

Two model types are available and compared:

- **Random Forest** — an ensemble of decision trees. Fast, robust, and works well even with small datasets.
- **LSTM** — a neural network designed for time-series data. Uses the past 2 hours of measurements to predict the current residual.

---

## Workflow

### 1. What you need to provide

The pipeline begins with two standard data sources:

- **Logger data CSV** — field measurements containing timestamps, measured temperatures, microhabitat label, and environmental covariates (e.g. solar radiation, wind speed, relative humidity).
- **Physical model predictions CSV** — mechanistic model predictions (e.g. NicheMapR) for the corresponding location and time period.

### 2. Pre-processing: Creating the Aligned CSV

Before running `microclCorr`, combine both sources into a single **aligned CSV** (done outside the package):

1. **Align by timestamp**: Match measured and predicted values for each time step.
2. **Compute residuals**: Add a `residual` column (`measured − predicted`).
3. *(If using multiple loggers)*: Add a microhabitat column if not already present, add a site identifier (e.g. `Site_ID = "Mishmar River"`), and stack all logger tables into one file.

Ready-to-run helper scripts for this step are provided in the [preprocessing examples](inst/examples/preprocessing_examples/).

### 3. Machine Learning Correction Pipeline (`microclCorr`)

Once the aligned CSV is prepared, you can use `microclCorr` functions to run the correction pipeline through four main stages (see the [workflow figure](vignettes/workflow_combined.png)):

1. **Data preparation**: Ingest the aligned CSV (`load_prepared_csv_data`), generate cyclical diurnal time features (`add_cyclical_time`), identify predictor columns (`get_feature_columns`), and partition data into training, validation, and test subsets using block-temporal splitting (`split_train_val_test`).
2. **Model training**: Train either model or both to predict the residual:
   - **Random Forest**: Fits decision tree ensembles (`train_rf`). Fast, robust, and effective across sample sizes without requiring deep learning dependencies.
   - **LSTM**: Scales inputs, constructs sliding time windows, and trains a recurrent neural network (`train_lstm`) to capture temporal inertia.
3. **Evaluation and model selection**: Evaluate prediction accuracy against uncorrected baseline predictions (`evaluate_correction`). If both models were trained, align their test sets (`align_test_sets`) to compare performance and select the best one. Then, export the chosen model bundle (`save_correction_model`).
4. **Apply to new data**: Load the saved model bundle (`load_correction_model`) and apply it to correct new physical model predictions across unmeasured periods or microhabitats (`correct_predictions`).

---

## Package Verification (`R CMD check`)

For reviewers performing package verification:

```bash
# Install all required and suggested dependencies
Rscript -e 'install.packages(c("knitr", "testthat", "ranger", "keras3", "tensorflow", "reticulate", "rmarkdown"), repos = "https://cloud.r-project.org")'

# Download the microclCorr_0.1.0.tar.gz file from the repository

# Run R CMD check on the built tarball (--no-manual skips PDF manual generation if LaTeX/pdflatex is not installed)
R CMD check --no-manual microclCorr_0.1.0.tar.gz
```

---

## Installation

### 1. Download and install locally

#### Option A: Install from the pre-built tarball (`microclCorr_0.1.0.tar.gz`)

Download `microclCorr_0.1.0.tar.gz` from the repository and install it directly:

```R
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_local("microclCorr_0.1.0.tar.gz")
```

Alternatively, from the terminal:
```bash
R CMD INSTALL microclCorr_0.1.0.tar.gz
```

#### Option B: Download the full repository (to access examples and raw datasets)

The pre-built `.tar.gz` package contains all package functions and compiled vignettes. If you also want to inspect or reproduce the empirical scenario scripts (`inst/examples/`) and accompanying datasets (`inst/extdata/`), download the full repository as a ZIP archive from:

> **[https://anonymous.4open.science/r/microcl_ml_corr-3E14/](https://anonymous.4open.science/r/microcl_ml_corr-3E14/)**

You can install the package directly from the downloaded ZIP file (dependencies such as `ranger` are installed automatically):

```R
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_local("microcl_ml_corr-3E14.zip")
```

### 2. Set up TensorFlow (required for LSTM models only)

If the user wants to train an LSTM model, the model uses Python's TensorFlow library under the hood. To configure your environment for TensorFlow:

```R
library(microclCorr)
setup_tensorflow()
```

`setup_tensorflow()` automatically discovers an existing Python environment with TensorFlow configured (such as an active Conda environment, a virtual environment, or `RETICULATE_PYTHON`). If not already available, it installs any missing required R packages (`reticulate`, `tensorflow`, `keras3`) and configures a dedicated virtual environment with TensorFlow and Keras.

---

## Examples

The package includes eight fully worked scenario scripts in [`inst/examples/`](inst/examples/). These correspond directly to the **4 Examples** presented in the accompanying manuscript (in preparation), alongside supplementary scenarios:

| Scenario / Script | Paper Example | Question answered |
|----------|---------------|-----------------|
| [Preprocessing](inst/examples/preprocessing_examples/) | — | How do I prepare my CSV files before running the pipeline? |
| [1 — Valley](inst/examples/scenario_1_valley_single_logger/) | **Example 1** | How well does local correction work? How much logger data do I need? |
| [2 — Beach](inst/examples/scenario_2_beach_single_logger/) | — | Same as above for a coastal site, where NicheMapR errors are larger. |
| [3 — Desert](inst/examples/scenario_3_desert_single_logger/) | — | Same as above for a desert site, where even 1–2 days of data is enough. |
| [4 — Beach Pooled](inst/examples/scenario_4_beach_pooled/) | **Example 2** | Does training on ALL loggers at once improve accuracy? |
| [5 — Beach Specialized](inst/examples/scenario_5_beach_specialized/) | — | Does training one model per location beat a single pooled model? |
| [6 — Desert Pooled](inst/examples/scenario_6_desert_pooled/) | — | Same as Scenario 4, but across 48 desert loggers. |
| [7 — Desert Specialized](inst/examples/scenario_7_desert_specialized/) | **Example 3** | Same as Scenario 5, but per desert region. |
| [8 — Zero-Shot Transfer](inst/examples/scenario_8_zero_shot_transfer/) | **Example 4** | Can the package correct a site where no logger data exists at all? |

Each example is a self-contained R script with plain-English comments throughout. To run an example, open the corresponding `run_scenario_N.R` file in RStudio and click **Source**.

See the [examples README](inst/examples/README.md) for a summary of results across all scenarios.

---

## License

This package is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
