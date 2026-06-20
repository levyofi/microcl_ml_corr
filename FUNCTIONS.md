# microclCorr — Function Reference

All examples below were verified against package version 0.1.0.

---

## Setup for examples

```r
library(microclCorr)
library(ranger)   # for RF examples

set.seed(42)
n <- 1500         # hourly observations per site (~62 days → 8 blocks of 7 d)
times <- seq(as.POSIXct("2023-01-01", tz = "UTC"), by = "hour", length.out = n)

make_site <- function(name, mu_res = 0, seed = 1) {
  set.seed(seed)
  data.frame(
    time            = times,
    residual        = rnorm(n, mu_res, 1.5),   # measured − NicheMapR
    predicted       = rnorm(n, 22, 4),          # NicheMapR output
    microhabitat    = sample(c("open", "shade"), n, replace = TRUE),
    temp_air        = rnorm(n, 25, 5),
    solar           = pmax(0, rnorm(n, 400, 200)),
    time_series_doc = name,
    stringsAsFactors = FALSE
  )
}

df <- rbind(make_site("site_A", 0, 1),
            make_site("site_B", 2, 2))
```

---

## Data Loading

### `load_prepared_csv_data()`

Reads a pre-aligned CSV, parses the datetime column, and one-hot encodes a categorical microhabitat column.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | character | — | Path to CSV file |
| `is_continuous_microhabitat` | logical | `FALSE` | Skip one-hot encoding if microhabitat is numeric |
| `datetime_format` | character | `"%Y-%m-%d %H:%M:%S"` | `strptime` format for the datetime column |
| `includes_index` | logical | `TRUE` | Whether the CSV has a leading row-index column (written by `write.csv`) |
| `microhabitat_col` | character | `"microhabitat"` | Name of the microhabitat column |
| `datetime_col` | character | `"time"` | Name of the datetime column |

**Returns** `data.frame` with parsed POSIXct datetime and one-hot microhabitat columns appended (original column kept as `microhabitat`).

**Example**

```r
tmp <- tempfile(fileext = ".csv")
write.csv(df[df$time_series_doc == "site_A",
             c("time", "residual", "predicted",
               "microhabitat", "temp_air", "solar", "time_series_doc")],
          tmp)

site_a <- load_prepared_csv_data(
  tmp,
  is_continuous_microhabitat = FALSE,
  datetime_format = "%Y-%m-%d %H:%M:%S",
  includes_index  = TRUE
)

# New columns: microhabitat_open, microhabitat_shade, microhabitat
names(site_a)
nrow(site_a)   # 1500
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
df <- add_cyclical_time(df, datetime_col = "time", add_month = TRUE)
# Added: Hour_sin, Hour_cos, Month_sin, Month_cos
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
feat_cols <- get_feature_columns(df)
# "temp_air"  "solar"  "Hour_sin"  "Hour_cos"  "Month_sin"  "Month_cos"
# (one-hot microhabitat columns appear here if present)
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
splits <- split_train_val_test(df, block_days = 7, seed = 42)
# train: 2328 rows   val: 336 rows   test: 336 rows  (total = 3000)

# Reproduce a specific Python-matched split
splits_fixed <- split_train_val_test(
  df, block_days = 7,
  train_blocks = c(0, 1, 2, 3, 4, 7),
  val_blocks   = c(5),
  test_blocks  = c(6)
)
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
- Requires enough blocks per site for at least one validation block: `floor(n_blocks_per_site × val_pct) ≥ 1`. With `block_days = 7` and `val_pct = 0.125` this needs ≥ 8 blocks (≥ 56 days) per site. Use `block_days = 3` for shorter series.
- Sites whose block count is too small will have no validation or test rows.

**Example**

```r
# df has 2 sites × 1500 rows → 8 blocks of 7 days per site
splits_s <- stratified_split_train_val_test(
  df,
  stratify_col = "time_series_doc",
  block_days   = 7,
  seed         = 42
)
# train: 2004   val: 324   test: 672   (total = 3000)
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
scaled <- lstm_scaling(splits$train, splits$val, splits$test)
# scaled$scaler$cols  →  c("temp_air", "solar", "Hour_sin", ...)

# Manual scale of new data at inference
new_data <- splits$test[1:5, ]
for (col in scaled$scaler$cols) {
  new_data[[col]] <- (new_data[[col]] - scaled$scaler$min[col]) /
                      scaled$scaler$range[col]
}
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
site_a_train <- splits$train[splits$train$time_series_doc == "site_A", ]

win <- make_windows(
  X_mat        = as.matrix(site_a_train[, feat_cols]),
  y_vec        = site_a_train$residual,
  base_pred_vec = site_a_train$predicted,
  datetime_vec = site_a_train$time,
  window_size  = 6,
  max_gap_hours = 1
)
dim(win$X)        # (n_windows, 6, 6)
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
lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size  = 6,
  ts_names_col = "time_series_doc"
)
# lstm_data$train_dict$X   dim: (n_train_windows, 6, 6)
# lstm_data$index_info$datasets   →  c("site_A", "site_B")
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
lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size = 6, ts_names_col = "time_series_doc"
)

rf_test_aligned <- align_test_sets(
  test_dataset  = splits$test,
  lstm_test_dict = lstm_data$test_dict,
  ts_index_info  = lstm_data$index_info,
  site_name_col  = "time_series_doc"
)
# nrow(rf_test_aligned) == length(lstm_data$test_dict$y)
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
rf <- train_rf(
  train_X       = splits$train[, feat_cols],
  train_y       = splits$train$residual,
  num_trees     = 500,
  tune          = TRUE,
  n_combinations = 5,
  val_X         = splits$val[, feat_cols],
  val_y         = splits$val$residual,
  seed          = 42
)
# RF HPO: Best MSE = ... | max_depth=..., min_node_size=..., mtry=...
```

---

### `build_lstm()`

Builds a compiled Keras sequential model with stacked LSTM layers, dropout, and a single linear output neuron.

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
model <- build_lstm(
  input_shape = c(6, length(feat_cols)),
  n_units  = 64,
  n_layers = 2,
  dropout  = 0.1,
  lr       = 0.001
)
```

---

### `train_lstm()`

Builds and trains a stacked LSTM using early stopping on validation loss.

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
lstm_data <- lstm_specific_preprocessing(
  scaled$train, scaled$val, scaled$test,
  window_size = 6, ts_names_col = "time_series_doc"
)

lstm_model <- train_lstm(
  train_X    = lstm_data$train_dict$X,
  train_y    = lstm_data$train_dict$y,
  val_X      = lstm_data$val_dict$X,
  val_y      = lstm_data$val_dict$y,
  n_units    = 64,
  n_layers   = 2,
  dropout    = 0.1,
  lr         = 0.001,
  epochs     = 100,
  batch_size = 32,
  patience   = 10,
  seed       = 42
)
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
# RF
corrected_rf <- correct_predictions(
  model        = rf,
  new_data     = splits$test,
  model_type   = "rf",
  feature_cols = feat_cols
)
head(corrected_rf)

# LSTM
# corrected_lstm <- correct_predictions(
#   model        = lstm_model,
#   new_data     = splits$test,
#   model_type   = "lstm",
#   scaler       = scaled$scaler,
#   feature_cols = feat_cols,
#   window_size  = 6
# )
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
metrics <- evaluate_correction(
  model           = rf,
  X               = splits$test[, feat_cols],
  y               = splits$test$residual,
  base_prediction = splits$test$predicted,
  model_type      = "rf"
)

cat(sprintf("RMSE  base: %.4f  corrected: %.4f\n",
            metrics$rmse_base, metrics$rmse_corr))
cat(sprintf("R²    base: %.4f  corrected: %.4f\n",
            metrics$r2_base,   metrics$r2_corr))
cat(sprintf("Improvement: %.1f%%\n",
            (1 - metrics$rmse_corr / metrics$rmse_base) * 100))
```

---

## Persistence

### `save_correction_model()`

Serialises a trained model together with its scaler and feature column list to an `.rds` file.

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
save_correction_model(rf, scaled$scaler, feat_cols, "rf_correction.rds")
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
bundle <- load_correction_model("rf_correction.rds")
bundle$model_type     # "rf"
bundle$feature_cols   # c("temp_air", "solar", ...)

# Re-use for prediction
corrected <- correct_predictions(
  model        = bundle$model,
  new_data     = splits$test,
  model_type   = bundle$model_type,
  feature_cols = bundle$feature_cols
)
```

---

## Typical workflow

```r
library(microclCorr)
library(ranger)

# 1. Load data
df <- load_prepared_csv_data("harod_dataset.csv",
                             datetime_format = "%d/%m/%Y %H:%M")

# 2. Feature engineering
df <- add_cyclical_time(df, add_month = TRUE)
feat_cols <- get_feature_columns(df)

# 3. Split
splits <- split_train_val_test(df, block_days = 7, seed = 123)

# 4. Scale (required for LSTM; harmless for RF)
scaled <- lstm_scaling(splits$train, splits$val, splits$test)

# 5a. Train RF
rf <- train_rf(splits$train[, feat_cols], splits$train$residual,
               val_X = splits$val[, feat_cols], val_y = splits$val$residual)

# 5b. Train LSTM
lstm_data <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
                                         window_size = 6)
lstm_model <- train_lstm(lstm_data$train_dict$X, lstm_data$train_dict$y,
                         lstm_data$val_dict$X,   lstm_data$val_dict$y)

# 6. Align test sets for fair comparison
rf_test <- align_test_sets(splits$test, lstm_data$test_dict,
                           lstm_data$index_info, site_name_col = "time_series_doc")

# 7. Evaluate
evaluate_correction(rf, rf_test[, feat_cols], rf_test$residual,
                    rf_test$predicted, model_type = "rf")

evaluate_correction(lstm_model, lstm_data$test_dict$X, lstm_data$test_dict$y,
                    lstm_data$test_dict$base_pred, model_type = "lstm")

# 8. Save
save_correction_model(rf, scaled$scaler, feat_cols, "rf_model.rds")
```
