test_that("correct_predictions and model save/load works for Random Forest", {
  set.seed(42)
  # create a random dataset for test
  train_X <- data.frame(temp_env = rnorm(20, 15, 2))
  train_y <- rnorm(20, 0, 1)
  
  # train a RF model on the random dataset
  rf_model <- train_rf(train_X, train_y, num_trees = 10, tune = FALSE, seed = 42)
  
  # create a random dataset for test
  df <- data.frame(
    time = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 01:00:00"), tz = "UTC"),
    predicted = c(20, 25),
    temp_env = c(15, 18)
  )
  
  # test correct_predictions function
  res <- correct_predictions(rf_model, df, model_type = "rf", feature_cols = "temp_env")
  expect_equal(nrow(res), 2)
  expect_named(res, c("datetime", "base_prediction", "correction", "corrected_prediction"))
  expect_equal(res$corrected_prediction, res$base_prediction + res$correction)
  
  tmp_path <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp_path), add = TRUE)
  
  # save the trained model
  save_correction_model(rf_model, scaler = NULL, feature_cols = "temp_env", path = tmp_path)
  # load the model again
  loaded <- load_correction_model(tmp_path)
  
  # Check that the loaded model is a ranger model and has the same feature columns
  expect_equal(loaded$model_type, "rf")
  expect_equal(loaded$feature_cols, "temp_env")

  # Test predicting with loaded model (verifies S3 predict dispatch)
  res_loaded <- correct_predictions(loaded$model, df, model_type = loaded$model_type, feature_cols = loaded$feature_cols)
  expect_equal(res_loaded$corrected_prediction, res$corrected_prediction)
})

test_that("load_prepared_csv_data throws informative error on datetime format mismatch", {
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv), add = TRUE)
  
  sample_df <- data.frame(
    id = 1:3,
    time = c("02/04/2025 1:00", "02/04/2025 2:00", "02/04/2025 3:00"),
    predicted = c(10, 11, 12),
    residual = c(1, -1, 0),
    microhabitat = c("sun", "shade", "air")
  )
  write.csv(sample_df, tmp_csv, row.names = TRUE)
  
  # Default format (%Y-%m-%d %H:%M:%S) should fail informatively on day/month/year dates
  expect_error(
    load_prepared_csv_data(tmp_csv, datetime_format = "%Y-%m-%d %H:%M:%S", includes_index = TRUE),
    "Failed to parse datetime column"
  )
  
  # Correct format should succeed
  loaded <- load_prepared_csv_data(tmp_csv, datetime_format = "%d/%m/%Y %H:%M", includes_index = TRUE)
  expect_equal(nrow(loaded), 3)
})

test_that("split_train_val_test chronological split handles small n without row duplication or NAs", {
  df3 <- data.frame(
    time = seq(as.POSIXct("2024-01-01", tz = "UTC"), by = "hour", length.out = 3),
    val = 1:3
  )
  sp3 <- split_train_val_test(df3, use_blocks = FALSE, train_pct = 0.75, val_pct = 0.125)
  total_rows <- nrow(sp3$train) + nrow(sp3$val) + nrow(sp3$test)
  expect_equal(total_rows, 3)
  expect_false(any(is.na(sp3$train$val)))
  expect_false(any(is.na(sp3$val$val)))
  expect_false(any(is.na(sp3$test$val)))
  # Check no row duplication
  all_times <- c(sp3$train$time, sp3$val$time, sp3$test$time)
  expect_equal(length(unique(all_times)), 3)
})

test_that("align_test_sets deduplicates test dataset timestamps without fan-out", {
  dt <- as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 01:00:00"), tz = "UTC")
  test_dict <- list(datetime = dt[1])
  index_info <- list(datasets = "site1", test_indices = list(site1 = 0L))
  
  # test_dataset has duplicated timestamp for site1
  test_df <- data.frame(
    time = c(dt[1], dt[1], dt[2]),
    time_series_doc = c("site1", "site1", "site1"),
    temp = c(10, 10, 15)
  )
  expect_warning(
    aligned <- align_test_sets(test_df, test_dict, index_info, "time_series_doc", "time"),
    "duplicate"
  )
  expect_equal(nrow(aligned), 1)
})

test_that("Issue 11: microhabitat level mismatch between train/test CSVs is handled cleanly", {
  tmp_train <- tempfile(fileext = ".csv")
  tmp_test  <- tempfile(fileext = ".csv")
  on.exit(unlink(c(tmp_train, tmp_test)))

  df_train <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"),
    TAREF = c(20, 22, 21),
    microhabitat = c("open", "shade", "open"),
    predicted = c(19, 21, 20),
    residual = c(1, 1, 1),
    stringsAsFactors = FALSE
  )
  df_test <- data.frame(
    time = c("2024-01-02 00:00:00", "2024-01-02 01:00:00"),
    TAREF = c(25, 26),
    microhabitat = c("open", "open"), # "shade" absent
    predicted = c(24, 25),
    residual = c(1, 1),
    stringsAsFactors = FALSE
  )
  write.csv(df_train, tmp_train)
  write.csv(df_test, tmp_test)

  # Test explicit microhabitat_levels argument in load_prepared_csv_data
  loaded_test_explicit <- load_prepared_csv_data(tmp_test, microhabitat_levels = c("open", "shade"))
  expect_true("microhabitat_shade" %in% names(loaded_test_explicit))
  expect_equal(loaded_test_explicit$microhabitat_shade, c(0, 0))

  # Test separate loading without microhabitat_levels (train has shade, test does not)
  train_loaded <- load_prepared_csv_data(tmp_train)
  test_loaded  <- load_prepared_csv_data(tmp_test)
  expect_true("microhabitat_shade" %in% names(train_loaded))
  expect_false("microhabitat_shade" %in% names(test_loaded))

  feat_cols <- c("TAREF", "microhabitat_open", "microhabitat_shade")
  rf <- train_rf(train_loaded[, feat_cols], train_loaded$residual, num_trees = 5, tune = FALSE, seed = 42)

  # Calling correct_predictions should synthesize missing microhabitat_shade with 0 instead of failing
  expect_message(
    res <- correct_predictions(rf, new_data = test_loaded, model_type = "rf", feature_cols = feat_cols),
    "Synthesized missing one-hot microhabitat"
  )
  expect_equal(nrow(res), 2)
  expect_false(any(is.na(res$corrected_prediction)))
})

test_that("Issue 12: wrong or whitespace-mangled column names produce clear, argument-naming errors", {
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv))

  df <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00"),
    TAREF = c(20, 22),
    predicted = c(19, 21),
    residual = c(1, 1),
    stringsAsFactors = FALSE
  )
  write.csv(df, tmp_csv)

  # load_prepared_csv_data with trailing whitespace in datetime_col
  expect_error(
    load_prepared_csv_data(tmp_csv, datetime_col = "time "),
    "datetime_col 'time ' not found.*Did you mean 'time'?"
  )

  # add_cyclical_time with bad datetime_col
  expect_error(
    add_cyclical_time(df, datetime_col = "bad_time"),
    "datetime_col 'bad_time' not found"
  )

  # split_train_val_test with bad datetime_col
  expect_error(
    split_train_val_test(df, datetime_col = "time_wrong"),
    "datetime_col 'time_wrong' not found"
  )

  # correct_predictions with bad prediction_col
  rf <- train_rf(df[, "TAREF", drop = FALSE], df$residual, num_trees = 5, tune = FALSE, seed = 42)
  expect_error(
    correct_predictions(rf, new_data = df, prediction_col = "predicted_typo"),
    "prediction_col 'predicted_typo' not found"
  )

  # correct_predictions with missing non-microhabitat feature_cols
  expect_error(
    correct_predictions(rf, new_data = df, feature_cols = c("TAREF", "MISSING_VAR")),
    "The following required feature columns specified in 'feature_cols' are missing from 'new_data': MISSING_VAR"
  )
})

test_that("Issue 13: stray non-numeric values in numeric columns are parsed as NA or warned about", {
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv))

  # CSV with "N/A" string in otherwise numeric column
  raw_lines <- c(
    ",time,TAREF,predicted,residual",
    "1,2024-01-01 00:00:00,20.5,20.0,0.5",
    "2,2024-01-01 01:00:00,N/A,21.0,0.0",
    "3,2024-01-01 02:00:00,22.1,21.5,0.6"
  )
  writeLines(raw_lines, tmp_csv)

  # load_prepared_csv_data should parse "N/A" as NA, remaining rows keep TAREF numeric
  expect_warning(
    loaded <- load_prepared_csv_data(tmp_csv),
    "dropped due to missing values"
  )
  expect_true(is.numeric(loaded$TAREF))
  expect_equal(nrow(loaded), 2)

  # get_feature_columns warns on character column with mostly numbers and stray strings
  mixed_df <- data.frame(
    time = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"), tz = "UTC"),
    TAREF = c("20.5", "N/A", "22.1"),
    predicted = c(20, 21, 22),
    residual = c(0.5, 0.0, 0.6),
    stringsAsFactors = FALSE
  )
  expect_warning(
    expect_warning(
      feat_cols <- get_feature_columns(mixed_df),
      "No numeric feature columns found"
    ),
    "Column 'TAREF' was excluded from feature columns because it is character"
  )
  expect_equal(length(feat_cols), 0)

  # train_rf throws informative error when 0 feature columns are passed
  expect_error(
    train_rf(mixed_df[, feat_cols, drop = FALSE], mixed_df$residual, num_trees = 5, tune = FALSE),
    "train_X contains 0 feature columns"
  )
})

test_that("Issue 14: load_prepared_csv_data complete.cases filter warns on sensor dropout NAs with breakdown", {
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv))

  df <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00", "2024-01-01 03:00:00"),
    TAREF = c(20.0, NA, 22.0, 23.0),
    RH = c(50.0, 55.0, NA, 52.0),
    predicted = c(19.0, 20.0, 21.0, 22.0),
    residual = c(1.0, 1.0, 1.0, 1.0),
    stringsAsFactors = FALSE
  )
  write.csv(df, tmp_csv)

  # 2 of 4 rows have NAs (1 in TAREF, 1 in RH)
  expect_warning(
    loaded <- load_prepared_csv_data(tmp_csv),
    "2 of 4 rows \\(50\\.0%\\) dropped due to missing values .* across columns: TAREF \\(1 NAs\\), RH \\(1 NAs\\)"
  )
  expect_equal(nrow(loaded), 2)
})

test_that("check_lstm_environment returns boolean when error = FALSE", {
  res <- check_lstm_environment(error = FALSE)
  expect_true(is.logical(res))
})

test_that("prepare_dataframe handles in-memory data frames with character and POSIXct datetimes", {
  # 1. Non-data.frame error
  expect_error(prepare_dataframe("not_a_df"), "df must be a data.frame")

  # 2. Missing datetime column with typo hint
  bad_col_df <- data.frame(time_col = "2024-01-01 00:00:00", val = 1)
  expect_error(
    prepare_dataframe(bad_col_df, datetime_col = "time_col "),
    "datetime_col 'time_col ' not found in data.frame.*Did you mean 'time_col'?"
  )

  # 3. Categorical microhabitat one-hot encoding with levels
  df_char <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"),
    microhabitat = c("open", "shade", "open"),
    TAREF = c(20, 21, 22),
    predicted = c(19, 20, 21),
    residual = c(1, 1, 1),
    stringsAsFactors = FALSE
  )

  prep_char <- prepare_dataframe(df_char, microhabitat_levels = c("open", "shade", "burrow"))
  expect_true(inherits(prep_char$time, "POSIXct"))
  expect_equal(prep_char$microhabitat_open, c(1, 0, 1))
  expect_equal(prep_char$microhabitat_shade, c(0, 1, 0))
  expect_equal(prep_char$microhabitat_burrow, c(0, 0, 0))
  expect_equal(prep_char$microhabitat, c("open", "shade", "open"))

  # 4. In-memory data frame with pre-parsed POSIXct
  df_posix <- df_char
  df_posix$time <- as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"), tz = "UTC")
  prep_posix <- prepare_dataframe(df_posix)
  expect_equal(prep_posix$time, df_posix$time)
  expect_equal(prep_posix$microhabitat_open, c(1, 0, 1))

  # 5. Missing values / complete.cases filtering
  df_na <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"),
    TAREF = c(20, NA, 22),
    residual = c(1, 1, 1),
    stringsAsFactors = FALSE
  )
  expect_warning(
    prep_na <- prepare_dataframe(df_na, complete_cases = TRUE),
    "1 of 3 rows \\(33\\.3%\\) dropped due to missing values .* across columns: TAREF \\(1 NAs\\)"
  )
  expect_equal(nrow(prep_na), 2)

  # With complete_cases = FALSE, rows with NAs are preserved
  prep_keep_na <- prepare_dataframe(df_na, complete_cases = FALSE)
  expect_equal(nrow(prep_keep_na), 3)

  # When all rows are dropped, error is thrown
  df_all_na <- data.frame(
    time = c("2024-01-01 00:00:00", "2024-01-01 01:00:00"),
    TAREF = c(NA, NA),
    stringsAsFactors = FALSE
  )
  expect_error(
    prepare_dataframe(df_all_na, complete_cases = TRUE),
    "All rows were dropped after parsing due to missing values"
  )

  # 6. load_prepared_csv_data gives identical results to prepare_dataframe on CSV read
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv), add = TRUE)
  write.csv(df_char, tmp_csv, row.names = FALSE)

  loaded_csv <- load_prepared_csv_data(tmp_csv, includes_index = FALSE)
  direct_prep <- prepare_dataframe(df_char)
  expect_equal(loaded_csv, direct_prep)
})





