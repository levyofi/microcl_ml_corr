# microclCorr — Function Reference

All examples below are runnable and verified against package version 0.1.0 using the bundled dataset `data(microclimate_sample)` or on-demand datasets via `get_example_data()`.

---

## Setup for examples

```r
library(microclCorr)
library(ranger)   # for Random Forest examples

# Option 1: Bundled 7-day microclimate sample (instant, no external files required)
data(microclimate_sample)
df <- microclimate_sample

# Option 2: Full multi-week field dataset retrieved on demand
# csv_path <- get_example_data("Harod_dataset.csv")
# df <- load_prepared_csv_data(csv_path, datetime_format = "%d/%m/%Y %H:%M")
```

---

## Environment Setup

### `check_lstm_environment(error = TRUE)`

Verifies that the required R packages (`reticulate`, `tensorflow`, `keras3`) and a Python environment with TensorFlow and Keras are installed and available. If any component is missing, it either halts execution and instructs the user to run `setup_tensorflow()` (when `error = TRUE`), or returns `FALSE` silently (when `error = FALSE`). Does not automatically install packages.

**Parameters**

- `error`: Logical. If `TRUE` (default), raises an informative error if the environment is not ready. If `FALSE`, returns `FALSE` without error.

**Returns** Invisible `TRUE` if the environment is ready; `FALSE` if not ready and `error = FALSE`.

**Example**

```r
# As an assertion:
check_lstm_environment()

# As a conditional check:
if (check_lstm_environment(error = FALSE)) {
  message("LSTM environment is ready!")
}
```

### `setup_tensorflow()`

Configures a Python environment with TensorFlow and Keras, and sets environment variables (`RETICULATE_PYTHON`, `KERAS_HOME`) before `reticulate` binds to Python. Installs any missing required R packages (`reticulate`, `tensorflow`, `keras3`) and creates/configures a virtual environment (`microcl_env`) with TensorFlow and Keras if no existing environment is found.

**Parameters**

- `envname`: Character string. Name of the virtual environment to use or create (default: `"microcl_env"`).
- `install_if_missing`: Logical. If `TRUE` (default), automatically installs missing required R packages and creates/configures the Python virtual environment if needed.

**Returns** Invisible file path to the Python executable, or `NULL`.

**Example**

```r
if (requireNamespace("reticulate", quietly = TRUE) &&
    requireNamespace("tensorflow", quietly = TRUE)) {
  py_path <- setup_tensorflow()
}
```

---

## Data Loading

### `get_example_data()`

Retrieves the local path to an example dataset. If the dataset exists locally in the package or user cache directory, its local path is returned immediately. Otherwise, it is downloaded on demand from the anonymous repository (`https://anonymous.4open.science/r/microcl_ml_corr-3E14/inst/extdata/`) and cached.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | character | — | Dataset filename (e.g. `"Harod_dataset.csv"`, `"Beach_data_preprocessed.csv"`, `"desert_data_preprocessed.csv"`, `"beach_splits.csv"`, `"desert_splits.csv"`) |
| `dest_dir` | character | `NULL` | Directory where downloaded files are cached. Defaults to `tools::R_user_dir("microclCorr", "data")` |
| `base_url` | character | `"https://anonymous.4open.science/r/microcl_ml_corr-3E14/inst/extdata/"` | Base URL for on-demand downloads |
| `force` | logical | `FALSE` | Force re-download even if the file exists |

**Returns**: `character` scalar containing the absolute path to the local CSV dataset.

**Example**

```r
# Retrieve local or cached example dataset:
csv_path <- get_example_data("Harod_dataset.csv")
file.exists(csv_path)  # TRUE
```

---

### `prepare_dataframe()`

Prepares an in-memory data.frame for ML training: parses/validates the datetime column (supporting pre-parsed POSIXct or strings), one-hot encodes categorical microhabitat, and filters complete cases. This is the in-memory equivalent of `load_prepared_csv_data()`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | data.frame | — | Input data.frame |
| `is_continuous_microhabitat` | logical | `FALSE` | Skip one-hot encoding if microhabitat is numeric |
| `datetime_format` | character | `"%Y-%m-%d %H:%M:%S"` | `strptime` format for string datetimes |
| `microhabitat_col` | character | `"microhabitat"` | Name of the microhabitat column |
| `datetime_col` | character | `"time"` | Name of the datetime column |
| `microhabitat_levels` | character vector | `NULL` | Optional expected microhabitat levels to guarantee matching dummy columns |
| `complete_cases` | logical | `TRUE` | Whether to drop incomplete rows with a warning |

**Returns** `data.frame` with parsed POSIXct datetime and one-hot microhabitat columns appended.

**Example**

```r
data(microclimate_sample)
prepared <- prepare_dataframe(microclimate_sample)
head(prepared[, c("time", "microhabitat", "predicted", "residual")])
```

---

### `load_prepared_csv_data()`

Reads a pre-aligned CSV, parses the datetime column, and one-hot encodes a categorical microhabitat column. Delegates preprocessing directly to `prepare_dataframe()`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | character | — | Path to CSV file |
| `is_continuous_microhabitat` | logical | `FALSE` | Skip one-hot encoding if microhabitat is numeric |
| `datetime_format` | character | `"%Y-%m-%d %H:%M:%S"` | `strptime` format for the datetime column |
| `includes_index` | logical | `TRUE` | Whether the CSV has a leading row-index column (written by `write.csv`) |
| `microhabitat_col` | character | `"microhabitat"` | Name of the microhabitat column |
| `datetime_col` | character | `"time"` | Name of the datetime column |
| `microhabitat_levels` | character vector | `NULL` | Optional expected microhabitat levels |
| `na.strings` | character vector | `c("NA", "N/A", ...)` | Strings to treat as NA when reading CSV |

**Returns** `data.frame` with parsed POSIXct datetime and one-hot microhabitat columns appended (original column kept as `microhabitat`).

**Example**

```r
# Load real-world field logger data directly
csv_path <- get_example_data("desert_data_preprocessed.csv")
loaded <- load_prepared_csv_data(csv_path)

# (For datasets with custom datetime formatting, pass datetime_format:
#  harod_path <- get_example_data("Harod_dataset.csv")
#  harod_df   <- load_prepared_csv_data(harod_path, datetime_format = "%d/%m/%Y %H:%M"))

head(loaded[, c("time", "microhabitat", "predicted", "residual")])
```

---

## Feature Engineering

### `add_cyclical_time()`

Adds sine/cosine encodings of hour-of-day (and optionally month-of-year) so that time wraps continuously (e.g. hour 23 is close to hour 0).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | data.frame | — | Input data |
| `datetime_col` | character | `"time"` | Name of the POSIXct column |
| `add_month` | logical | `FALSE` | Also add `Month_sin` / `Month_cos` |

**Returns** The same data.frame with columns `Hour_sin`, `Hour_cos` added (and `Month_sin`, `Month_cos` if `add_month = TRUE`).

**Example**

```r
data(microclimate_sample)
df <- add_cyclical_time(microclimate_sample, datetime_col = "time", add_month = TRUE)
# Added: Hour_sin, Hour_cos, Month_sin, Month_cos
names(df)
```

---

### `get_feature_columns()`

Returns the column names suitable for model input by excluding target, datetime, microhabitat (raw), and other metadata columns.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | data.frame | — | Input data |
| `avoid_cols` | character vector | internal list | Columns always excluded (e.g. `"time"`, `"time_series_doc"`) |
| `target_col` | character | `"residual"` | Target column to exclude |
| `microhabitat_col` | character | `"microhabitat"` | Raw microhabitat column to exclude |
| `prediction_col` | character | `"predicted"` | Base prediction column to exclude |

**Returns** Character vector of feature column names.

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
print(feat_cols)
# [1] "sun_temp"   "shade_temp" "air_temp"   "TAREF"      "RH"
# [6] "VREF"       "SOLR"       "TSKYC"      "DEW"        "Hour_sin"
# [11] "Hour_cos"
```

---

## Data Splitting

### `split_train_val_test()`

Splits a dataset into train / validation / test using shuffled N-day blocks. Block-shuffle prevents the model from seeing contiguous future data during training while still mixing seasons.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | data.frame | — | Input data with a datetime column |
| `train_pct` | numeric | `0.75` | Fraction of blocks assigned to training |
| `val_pct` | numeric | `0.125` | Fraction of blocks assigned to validation |
| `block_days` | integer | `7` | Number of days per block |
| `use_blocks` | logical | `TRUE` | Use block-shuffle split; `FALSE` for simple chronological |
| `datetime_col` | character | `"time"` | Datetime column name |
| `seed` | integer | `123` | Random seed |
| `train_blocks` | integer vector | `NULL` | Override: pre-defined block indices for training |
| `val_blocks` | integer vector | `NULL` | Override: pre-defined block indices for validation |
| `test_blocks` | integer vector | `NULL` | Override: pre-defined block indices for test |

**Returns** List with elements `train`, `val`, `test` (data.frames, each sorted by datetime).

**Notes**
- Falls back to a simple chronological split and emits a warning if the dataset has fewer than 3 blocks.
- Pass explicit `train_blocks` / `val_blocks` / `test_blocks` to reproduce a Python pipeline's exact split.

**Example**

```r
data(microclimate_sample)
# Split 7 days of data into 2-day blocks: 75% train, 12.5% val, 12.5% test
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
cat(sprintf("Train: %d | Val: %d | Test: %d (total = %d)\n",
            nrow(splits$train), nrow(splits$val), nrow(splits$test), nrow(microclimate_sample)))
# Train: 216 | Val: 144 | Test: 144 (total = 504)
```

---

### `stratified_split_train_val_test()`

Like `split_train_val_test()` but performs the block-shuffle independently per site, ensuring every site contributes data to all three splits. Block numbering is relative to each site's own date range.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | data.frame | — | Input data |
| `train_pct` | numeric | `0.75` | Training fraction |
| `val_pct` | numeric | `0.125` | Validation fraction |
| `stratify_col` | character | — | **Required.** Column to stratify by (e.g. `"time_series_doc"`) |
| `block_days` | integer | `7` | Days per block |
| `datetime_col` | character | `"time"` | Datetime column name |
| `seed` | integer | `123` | Random seed |

**Returns** List with `train`, `val`, `test` data.frames. Rows are disjoint and `nrow(train) + nrow(val) + nrow(test) == nrow(data)`.

**Notes**
- Requires enough blocks per site for at least one validation block: `floor(n_blocks_per_site × val_pct) ≥ 1`. With `block_days = 7` and `val_pct = 0.125` this needs ≥ 8 blocks (≥ 56 days) per site. Use smaller `block_days` (e.g. 1 or 2) for shorter series.

**Example**

```r
data(microclimate_sample)
# 3 microhabitats × 7 days (1-day blocks per logger)
splits_s <- stratified_split_train_val_test(
  microclimate_sample,
  stratify_col = "time_series_doc",
  train_pct    = 0.6,
  val_pct      = 0.2,
  block_days   = 1,
  seed         = 42
)
cat(sprintf("Train: %d | Val: %d | Test: %d (total = %d)\n",
            nrow(splits_s$train), nrow(splits_s$val), nrow(splits_s$test), nrow(microclimate_sample)))
# Train: 288 | Val: 72 | Test: 144 (total = 504)
```

---

## Scaling and Windowing (LSTM)

### `lstm_scaling()`

Applies MinMax scaling to feature columns. The scaler is **fit only on training data** to prevent data leakage, then applied to val and test.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train` | data.frame | — | Training split |
| `val` | data.frame | — | Validation split |
| `test` | data.frame | — | Test split |
| `avoid_cols` | character vector | internal list | Columns excluded from scaling |
| `target_col` | character | `"residual"` | Target column (excluded from scaling) |
| `microhabitat_col` | character | `"microhabitat"` | Raw microhabitat column (excluded) |
| `prediction_col` | character | `"predicted"` | Base prediction column (excluded) |

**Returns** List with `train`, `val`, `test` (scaled data.frames) and `scaler` (list with `min`, `range`, `cols`). Pass `scaler` to `save_correction_model()` so it can be reused at inference.

**Example**

```r
data(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# Scaler attributes:
scaled$scaler$cols
# c("sun_temp", "shade_temp", "air_temp", "TAREF", "RH", ...)
```

---

### `make_windows()`

Reshapes a time series into overlapping sliding windows for LSTM input. Windows that span a temporal gap larger than `max_gap_hours` are silently skipped.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X_mat` | matrix | — | Feature matrix (n_samples × n_features) |
| `y_vec` | numeric | — | Target vector (length n_samples) |
| `base_pred_vec` | numeric | — | Base model predictions (length n_samples) |
| `datetime_vec` | POSIXct | — | Timestamps (length n_samples) |
| `window_size` | integer | — | Number of time steps per window |
| `max_gap_hours` | numeric | `1` | Maximum allowed gap between consecutive timestamps; `NULL` disables the check |

**Returns** List with:
- `X` — 3-D array `(n_windows, window_size, n_features)`
- `y` — numeric vector of targets (one per window, taken from the **last** step)
- `base_pred` — numeric vector of base predictions (last step)
- `datetime` — POSIXct vector (last step of each window)

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# Create 6-hour sliding windows for one logger's training subset
sun_train <- scaled$train[scaled$train$time_series_doc == "harod2_sun.csv", ]

win <- make_windows(
  X_mat         = as.matrix(sun_train[, feat_cols]),
  y_vec         = sun_train$residual,
  base_pred_vec = sun_train$predicted,
  datetime_vec  = sun_train$time,
  window_size   = 6,
  max_gap_hours = 1
)
dim(win$X)        # (n_windows, 6, n_features)
length(win$y)     # n_windows
```

---

### `lstm_specific_preprocessing()`

Runs `make_windows()` for every site in all three splits and concatenates the results. Returns per-site window indices so results can later be mapped back to individual loggers.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train` | data.frame | — | Scaled training data |
| `val` | data.frame | — | Scaled validation data |
| `test` | data.frame | — | Scaled test data |
| `window_size` | integer | — | Window size in time steps |
| `ts_names_col` | character | `"time_series_doc"` | Column identifying individual sites / loggers |

**Returns** List with:
- `train_dict`, `val_dict`, `test_dict` — each a list with `X` (3-D array), `y`, `base_pred`, `datetime`
- `index_info` — list with `datasets` (site names) and `train_indices`, `val_indices`, `test_indices` (per-site window offsets)

**Example**

```r
data(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size  = 6,
  ts_names_col = "time_series_doc"
)
dim(lstm_data$train_dict$X)        # (186, 6, 11)
lstm_data$index_info$datasets      # c("harod2_air.csv", "harod2_shd.csv", "harod2_sun.csv")
```

---

### `align_test_sets()`

Filters the point-based test data.frame to keep only the rows that correspond to the **last time step** of each LSTM test window. Use this before RF evaluation when you want RF and LSTM to be compared on exactly the same set of time points.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `test_dataset` | data.frame | — | Original (unwindowed) test data |
| `lstm_test_dict` | list | — | `test_dict` from `lstm_specific_preprocessing()` |
| `ts_index_info` | list | — | `index_info` from `lstm_specific_preprocessing()` |
| `site_name_col` | character | — | Column identifying sites |
| `datetime_col` | character | `"time"` | Datetime column name |

**Returns** data.frame with only the rows matching LSTM window endpoints, in the same order as the LSTM test dict.

**Example**

```r
data(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size = 6, ts_names_col = "time_series_doc"
)

rf_test_aligned <- align_test_sets(
  test_dataset   = splits$test,
  lstm_test_dict = lstm_data$test_dict,
  ts_index_info  = lstm_data$index_info,
  site_name_col  = "time_series_doc"
)
# Exact 1-to-1 row alignment:
nrow(rf_test_aligned) == length(lstm_data$test_dict$y)  # TRUE
```

---

## Model Training

### `train_rf()`

Trains a `ranger` Random Forest to predict residuals. Optionally performs a random hyperparameter search using either a held-out validation set or out-of-bag error, then retrains with the best parameters and `num_trees`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_X` | data.frame or matrix | — | Training features |
| `train_y` | numeric | — | Training targets (residuals) |
| `num_trees` | integer | `500` | Number of trees in the final model |
| `tune` | logical | `TRUE` | Whether to search hyperparameters |
| `n_combinations` | integer | `5` | How many random HP combinations to try |
| `max_depth_options` | integer vector | `c(10,20,30,0)` | `max.depth` candidates (`0` = unlimited) |
| `min_node_size_options` | integer vector | `c(2,5,10)` | `min.node.size` candidates |
| `mtry_options` | integer vector | `NULL` | `mtry` candidates; defaults to `{√p, p/3, p}` |
| `val_X` | data.frame or matrix | `NULL` | Validation features for HP scoring (uses OOB error if `NULL`) |
| `val_y` | numeric | `NULL` | Validation targets |
| `seed` | integer | `123` | Random seed |

**Returns** A fitted `ranger` model object.

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)

rf <- train_rf(
  train_X        = splits$train[, feat_cols],
  train_y        = splits$train$residual,
  num_trees      = 100,
  tune           = TRUE,
  n_combinations = 3,
  val_X          = splits$val[, feat_cols],
  val_y          = splits$val$residual,
  seed           = 42
)
# RF HPO: Best MSE = ... | max_depth=..., min_node_size=..., mtry=...
```

---

### `build_lstm()`

Builds a compiled Keras sequential model with stacked LSTM layers, dropout, and a single linear output neuron. Requires the optional `keras3` package.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_shape` | numeric vector | — | `c(window_size, n_features)` |
| `n_units` | integer | `64` | LSTM units per layer |
| `n_layers` | integer | `2` | Number of stacked LSTM layers |
| `dropout` | numeric | `0.1` | Dropout rate (0–1) applied after the last LSTM layer |
| `lr` | numeric | `0.001` | Adam learning rate; `NULL` uses Keras default |

**Returns** A compiled `keras` model (loss = MSE, optimizer = Adam).

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)

# Build a 2-layer LSTM model for 6-step sequences
if (requireNamespace("keras3", quietly = TRUE)) {
  model <- build_lstm(
    input_shape = c(6, length(feat_cols)),
    n_units     = 32,
    n_layers    = 2,
    dropout     = 0.1,
    lr          = 0.001
  )
}
```

---

### `train_lstm()`

Builds and trains a stacked LSTM using early stopping on validation loss. Requires the optional `keras3` package.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_X` | 3-D array | — | `(n_windows, window_size, n_features)` |
| `train_y` | numeric | — | Training targets |
| `val_X` | 3-D array | — | Validation features |
| `val_y` | numeric | — | Validation targets |
| `n_units` | integer | `64` | LSTM units |
| `n_layers` | integer | `2` | Stacked LSTM layers |
| `dropout` | numeric | `0.1` | Dropout rate |
| `lr` | numeric | `0.001` | Learning rate |
| `epochs` | integer | `100` | Maximum training epochs |
| `batch_size` | integer | `32` | Mini-batch size |
| `patience` | integer | `10` | Early-stopping patience (epochs without improvement) |
| `seed` | integer | `42` | Random seed (TensorFlow + R) |

**Returns** A trained `keras` model with the best weights restored.

**Example**

```r
if (requireNamespace("keras3", quietly = TRUE)) {
  lstm_model <- train_lstm(
    train_X    = lstm_data$train_dict$X,
    train_y    = lstm_data$train_dict$y,
    val_X      = lstm_data$val_dict$X,
    val_y      = lstm_data$val_dict$y,
    n_units    = 32,
    n_layers   = 2,
    dropout    = 0.1,
    lr         = 0.001,
    epochs     = 50,
    batch_size = 32,
    patience   = 10,
    seed       = 42
  )
}
```

---

### `lstm_hypertuning()`

Performs a random search over LSTM hyperparameters (`n_units`, `n_layers`, `dropout`, `lr`) using validation loss for model selection. Requires the optional `keras3` and `tensorflow` packages.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_X` | 3-D array | — | `(n_windows, window_size, n_features)` |
| `train_y` | numeric | — | Training targets |
| `val_X` | 3-D array | — | Validation features |
| `val_y` | numeric | — | Validation targets |
| `n_trials` | integer | `5` | Number of random hyperparameter combinations to try |
| `units_range` | integer vector | `c(32, 512)` | Min/max for `n_units` (searched in steps of 32) |
| `layers_range` | integer vector | `c(1, 3)` | Min/max for stacked LSTM layers |
| `dropout_range` | numeric vector | `c(0, 0.3)` | Min/max for dropout rate |
| `lr_range` | numeric vector | `c(1e-4, 0.01)` | Min/max for Adam learning rate (log-uniform) |
| `epochs` | integer | `100` | Max epochs per trial |
| `batch_size` | integer | `32` | Batch size |
| `patience` | integer | `10` | Early stopping patience |
| `seed` | integer | `123` | Random seed |

**Returns** List with `model` (best fitted Keras model), `params` (list of best hyperparameters), and `val_mse`.

**Example**

```r
if (requireNamespace("keras3", quietly = TRUE) &&
    requireNamespace("tensorflow", quietly = TRUE)) {
  hpo <- lstm_hypertuning(
    train_X  = lstm_data$train_dict$X,
    train_y  = lstm_data$train_dict$y,
    val_X    = lstm_data$val_dict$X,
    val_y    = lstm_data$val_dict$y,
    n_trials = 2,
    epochs   = 2
  )
  print(hpo$params)
}
```

---

## Prediction

### `correct_predictions()`

Applies a trained RF or LSTM correction model to new data and returns base predictions, predicted corrections, and corrected predictions.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | ranger or keras model | — | Trained model |
| `new_data` | data.frame | — | New data including feature and base-prediction columns |
| `model_type` | character | — | `"rf"` or `"lstm"` |
| `scaler` | list | `NULL` | Scaler from `lstm_scaling()` (required for LSTM) |
| `feature_cols` | character vector | `NULL` | Feature column names; inferred via `get_feature_columns()` if `NULL` |
| `prediction_col` | character | `"predicted"` | Base model prediction column |
| `window_size` | integer | `2` | Window size for LSTM windowing |
| `datetime_col` | character | `"time"` | Datetime column for LSTM windowing |

**Returns** data.frame with columns `datetime`, `base_prediction`, `correction`, `corrected_prediction`.

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
rf <- train_rf(splits$train[, feat_cols], splits$train$residual, num_trees = 50, tune = FALSE)

# Generate corrections on test data
corrected_rf <- correct_predictions(
  model        = rf,
  new_data     = splits$test,
  model_type   = "rf",
  feature_cols = feat_cols
)
head(corrected_rf)
```

---

## Evaluation

### `evaluate_correction()`

Computes RMSE and R² for both the uncorrected base predictions and the ML-corrected predictions.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | ranger or keras model | — | Trained model |
| `X` | data.frame or 3-D array | — | Test features (data.frame for RF, 3-D array for LSTM) |
| `y` | numeric | — | Test targets (residuals) |
| `base_prediction` | numeric | — | NicheMapR base predictions |
| `model_type` | character | — | `"rf"` or `"lstm"` |

**Returns** List with `rmse_base`, `rmse_corr`, `r2_base`, `r2_corr`.

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
rf <- train_rf(splits$train[, feat_cols], splits$train$residual, num_trees = 50, tune = FALSE)

metrics <- evaluate_correction(
  model           = rf,
  X               = splits$test[, feat_cols],
  y               = splits$test$residual,
  base_prediction = splits$test$predicted,
  model_type      = "rf"
)

cat(sprintf("Baseline RMSE:  %.2f°C\n", metrics$rmse_base))
cat(sprintf("Corrected RMSE: %.2f°C\n", metrics$rmse_corr))
cat(sprintf("Improvement:    %.1f%%\n",
            (metrics$rmse_base - metrics$rmse_corr) / metrics$rmse_base * 100))
```

---

## Persistence

### `save_correction_model()`

Saves a trained correction model (Random Forest or LSTM) and its metadata to a single, portable `.rds` file. For LSTM models, the Keras model is serialized as raw binary bytes directly inside the `.rds` file, ensuring single-file portability across R sessions.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | ranger or keras model | — | Trained model |
| `scaler` | list or NULL | — | Scaler from `lstm_scaling()`; `NULL` for RF-only deployments |
| `feature_cols` | character vector | — | Feature column names |
| `path` | character | — | Output file path (should end in `.rds`) |

**Returns** `path` invisibly.

**Example**

```r
data(microclimate_sample)
feat_cols <- get_feature_columns(microclimate_sample)
rf <- train_rf(microclimate_sample[, feat_cols], microclimate_sample$residual, num_trees = 20, tune = FALSE)

tmp_file <- tempfile(fileext = ".rds")
save_correction_model(rf, scaler = NULL, feature_cols = feat_cols, path = tmp_file)
```

---

### `load_correction_model()`

Loads a model bundle saved by `save_correction_model()`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | character | — | Path to `.rds` file |

**Returns** List with `model`, `scaler`, `feature_cols`, `model_type` (`"rf"` or `"lstm"`).

**Example**

```r
bundle <- load_correction_model(tmp_file)
bundle$model_type     # "rf"
bundle$feature_cols   # feature column names
unlink(tmp_file)

# Re-use loaded model for predictions:
corrected <- correct_predictions(
  model        = bundle$model,
  new_data     = microclimate_sample[1:10, ],
  model_type   = bundle$model_type,
  feature_cols = bundle$feature_cols
)
head(corrected)
```

---

## Typical workflow

```r
library(microclCorr)
library(ranger)

# 1. Load data
# Retrieve real field dataset via get_example_data():
csv_path <- get_example_data("Harod_dataset.csv")
df <- load_prepared_csv_data(csv_path, datetime_format = "%d/%m/%Y %H:%M")

# (Or for an instant in-memory quick-start: data(microclimate_sample); df <- microclimate_sample)

# 2. Feature engineering
df <- add_cyclical_time(df, datetime_col = "time", add_month = TRUE)
feat_cols <- get_feature_columns(df)

# 3. Split into 7-day blocks
splits <- split_train_val_test(df, block_days = 7, seed = 123)

# 4. Scale features (required for LSTM; harmless for RF)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# 5. Train Random Forest
rf <- train_rf(
  train_X   = splits$train[, feat_cols],
  train_y   = splits$train$residual,
  val_X     = splits$val[, feat_cols],
  val_y     = splits$val$residual,
  num_trees = 100
)

# 6. Evaluate correction on held-out test data
metrics <- evaluate_correction(
  model           = rf,
  X               = splits$test[, feat_cols],
  y               = splits$test$residual,
  base_prediction = splits$test$predicted,
  model_type      = "rf"
)

cat(sprintf("Baseline RMSE:  %.2f°C\n", metrics$rmse_base))
cat(sprintf("Corrected RMSE: %.2f°C (%.1f%% improvement)\n",
            metrics$rmse_corr,
            (metrics$rmse_base - metrics$rmse_corr) / metrics$rmse_base * 100))

# 7. Save portable single-file model bundle
save_correction_model(rf, scaler = scaled$scaler, feature_cols = feat_cols, path = "rf_model.rds")
```
