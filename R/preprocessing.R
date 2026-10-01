# ---- microclCorr: Preprocessing ----
# Port of preprocessing.py from the Python pipeline

#' Prepare an in-memory dataset for ML training
#'
#' Takes an in-memory data.frame produced by data alignment and prepares it
#' for ML training: parses/validates datetime, one-hot encodes categorical microhabitat,
#' and optionally filters complete cases.
#'
#' @param df A data.frame.
#' @param is_continuous_microhabitat Logical. TRUE if microhabitat is a continuous variable.
#' @param datetime_format strptime format string for parsing the datetime column (used if datetime column is character or factor).
#' @param microhabitat_col Name of the microhabitat column.
#' @param datetime_col Name of the datetime column.
#' @param microhabitat_levels Optional character vector specifying all expected microhabitat
#'   levels. Useful when preparing train and test sets separately so that one-hot encoded
#'   columns match even if some levels are absent from a split.
#' @param complete_cases Logical. Whether to drop incomplete rows with a warning (default `TRUE`).
#' @param source_label Character string describing the source data in error messages (default `"data.frame"`).
#' @return A data.frame with parsed datetime and (optionally) one-hot encoded microhabitat.
#' @seealso \code{\link{load_prepared_csv_data}} for loading and preparing directly from a CSV file.
#' @examples
#' data(microclimate_sample)
#' prepared <- prepare_dataframe(microclimate_sample)
#' head(prepared[, c("time", "microhabitat", "predicted", "residual")])
#' @export
prepare_dataframe <- function(df,
                              is_continuous_microhabitat = FALSE,
                              datetime_format = "%Y-%m-%d %H:%M:%S",
                              microhabitat_col = "microhabitat",
                              datetime_col = "time",
                              microhabitat_levels = NULL,
                              complete_cases = TRUE,
                              source_label = "data.frame") {

  if (!is.data.frame(df)) {
    stop("df must be a data.frame, got: ", class(df)[1], call. = FALSE)
  }
  df <- as.data.frame(df)

  # Validate datetime_col existence (Issue 12)
  if (!(datetime_col %in% names(df))) {
    trimmed_cand <- names(df)[trimws(names(df)) == trimws(datetime_col)]
    case_cand <- names(df)[tolower(names(df)) == tolower(datetime_col)]
    cand <- unique(c(trimmed_cand, case_cand))
    hint <- if (length(cand) > 0) {
      sprintf(" Did you mean '%s'?", cand[1])
    } else {
      ""
    }
    stop(sprintf(
      "datetime_col '%s' not found in %s.%s Available columns: %s",
      datetime_col, source_label, hint, paste(names(df), collapse = ", ")
    ), call. = FALSE)
  }

  # One-hot encode categorical microhabitat (Issue 11)
  if (!is_continuous_microhabitat && microhabitat_col %in% names(df)) {
    orig_micro <- df[[microhabitat_col]]
    df[[microhabitat_col]] <- NULL
    if (!is.null(microhabitat_levels)) {
      levels_micro <- unique(c(as.character(microhabitat_levels), as.character(unique(orig_micro))))
    } else {
      levels_micro <- sort(unique(orig_micro))
    }
    for (lvl in levels_micro) {
      df[[paste0(microhabitat_col, "_", lvl)]] <- as.numeric(orig_micro == lvl)
    }
    df[[microhabitat_col]] <- orig_micro
  }

  # Datetime normalization & parsing
  time_vals <- df[[datetime_col]]
  if (inherits(time_vals, "POSIXt")) {
    parsed_time <- as.POSIXct(time_vals, tz = "UTC")
  } else if (inherits(time_vals, "Date")) {
    parsed_time <- as.POSIXct(time_vals, tz = "UTC")
  } else {
    time_char <- as.character(time_vals)
    no_colon <- !grepl(":", time_char, fixed = TRUE)
    if (datetime_format == "%Y-%m-%d %H:%M:%S") {
      time_char[no_colon] <- paste0(time_char[no_colon], " 0:00:00")
    } else if (datetime_format == "%d/%m/%Y %H:%M") {
      time_char[no_colon] <- paste0(time_char[no_colon], " 0:00")
    }

    parsed_time <- as.POSIXct(time_char, format = datetime_format, tz = "UTC")
    if (length(time_char) > 0 && all(is.na(parsed_time))) {
      stop(sprintf(
        "Failed to parse datetime column '%s' using format '%s'. All values converted to NA. Example value from dataset: '%s'",
        datetime_col, datetime_format, as.character(time_char[1])
      ), call. = FALSE)
    }

    na_count <- sum(is.na(parsed_time))
    if (na_count > 0) {
      warning(sprintf(
        "%d of %d timestamps in column '%s' could not be parsed with format '%s' and resulted in NA.",
        na_count, length(time_char), datetime_col, datetime_format
      ), call. = FALSE)
    }
  }

  df[[datetime_col]] <- parsed_time

  # Complete cases filtering (Issue 14)
  if (complete_cases) {
    n_rows_before <- nrow(df)
    na_per_col <- colSums(is.na(df))
    cols_with_na <- names(na_per_col)[na_per_col > 0]

    df <- df[complete.cases(df), , drop = FALSE]
    n_dropped <- n_rows_before - nrow(df)

    if (n_rows_before > 0 && nrow(df) == 0) {
      stop("All rows were dropped after parsing due to missing values (complete.cases). Check your datetime_format and input data.", call. = FALSE)
    }

    if (n_dropped > 0) {
      col_breakdown <- paste(sprintf("%s (%d NAs)", cols_with_na, na_per_col[cols_with_na]), collapse = ", ")
      warning(sprintf(
        "%d of %d rows (%.1f%%) dropped due to missing values (sensor dropouts / NAs) across columns: %s.",
        n_dropped, n_rows_before, 100 * n_dropped / n_rows_before, col_breakdown
      ), call. = FALSE)
    }
  }

  return(df)
}

#' Load a prepared CSV dataset
#'
#' Loads a CSV produced by the data alignment pipeline and prepares it
#' for ML training: parses datetime, one-hot encodes categorical microhabitat.
#'
#' @param path Path to CSV file
#' @param is_continuous_microhabitat Logical. TRUE if microhabitat is a continuous variable.
#' @param datetime_format strptime format string for parsing the datetime column.
#' @param includes_index Logical. TRUE if the CSV has a row-index column.
#' @param microhabitat_col Name of the microhabitat column.
#' @param datetime_col Name of the datetime column.
#' @param microhabitat_levels Optional character vector specifying all expected microhabitat
#'   levels. Useful when loading train and test CSVs separately so that one-hot encoded
#'   columns match even if some levels are absent from a split.
#' @param na.strings Character vector of strings to interpret as NA when reading the CSV
#'   (default includes "NA", "N/A", "null", "NULL", "NaN", "").
#' @return A data.frame with parsed datetime and (optionally) one-hot encoded microhabitat.
#' @seealso \code{\link{prepare_dataframe}} for preparing an in-memory data.frame.
#' @examples
#' csv_path <- get_example_data("desert_data_preprocessed.csv")
#' loaded <- load_prepared_csv_data(csv_path)
#' head(loaded[, c("time", "microhabitat", "predicted", "residual")])
#' @export
load_prepared_csv_data <- function(path,
                                   is_continuous_microhabitat = FALSE,
                                   datetime_format = "%Y-%m-%d %H:%M:%S",
                                   includes_index = TRUE,
                                   microhabitat_col = "microhabitat",
                                   datetime_col = "time",
                                   microhabitat_levels = NULL,
                                   na.strings = c("NA", "N/A", "null", "NULL", "NaN", "")) {

  if (includes_index) {
    df <- utils::read.csv(path, row.names = 1, stringsAsFactors = FALSE, na.strings = na.strings)
  } else {
    df <- utils::read.csv(path, stringsAsFactors = FALSE, na.strings = na.strings)
  }

  prepare_dataframe(
    df = df,
    is_continuous_microhabitat = is_continuous_microhabitat,
    datetime_format = datetime_format,
    microhabitat_col = microhabitat_col,
    datetime_col = datetime_col,
    microhabitat_levels = microhabitat_levels,
    complete_cases = TRUE,
    source_label = sprintf("CSV '%s'", path)
  )
}


#' Get feature columns for model training
#'
#' Returns column names suitable for model training by excluding
#' target, datetime, microhabitat, and other non-feature columns.
#'
#' @param df A data.frame
#' @param avoid_cols Character vector of column names to exclude
#' @param target_col Target column name
#' @param microhabitat_col Microhabitat column name
#' @param prediction_col Prediction column name
#' @return Character vector of feature column names
#' @examples
#' data(microclimate_sample)
#' feature_cols <- get_feature_columns(microclimate_sample)
#' print(feature_cols)
#' @export
get_feature_columns <- function(df,
                                avoid_cols = .default_cols$avoid,
                                target_col = .default_cols$target,
                                microhabitat_col = .default_cols$microhabitat,
                                prediction_col = .default_cols$prediction) {
  cols_to_exclude <- unique(c(avoid_cols, target_col, microhabitat_col, prediction_col))
  candidates <- setdiff(names(df), cols_to_exclude)
  # Keep only numeric columns (excludes leftover character metadata columns)
  numeric_cols <- candidates[sapply(df[candidates], is.numeric)]

  # Check non-numeric candidate columns for numeric data coerced by stray strings (Issue 13)
  non_numeric_candidates <- setdiff(candidates, numeric_cols)
  for (col in non_numeric_candidates) {
    vals <- df[[col]]
    if (is.character(vals) || is.factor(vals)) {
      num_vals <- suppressWarnings(as.numeric(as.character(vals)))
      valid_num_count <- sum(!is.na(num_vals))
      non_empty <- sum(!is.na(vals) & trimws(as.character(vals)) != "")
      if (non_empty > 0 && (valid_num_count / non_empty) >= 0.5) {
        bad_idx <- which(!is.na(vals) & is.na(num_vals) & trimws(as.character(vals)) != "")
        bad_samples <- unique(as.character(vals[bad_idx]))
        if (length(bad_samples) > 3) bad_samples <- c(bad_samples[1:3], "...")
        warning(sprintf(
          "Column '%s' was excluded from feature columns because it is %s, but appears to contain mostly numeric values (%.1f%%) with stray non-numeric strings (%s). Check for unparsed missing values (e.g. 'N/A') or typo strings.",
          col, class(vals)[1], 100 * valid_num_count / non_empty,
          paste(sprintf("'%s'", bad_samples), collapse = ", ")
        ), call. = FALSE)
      }
    }
  }

  if (length(numeric_cols) == 0) {
    warning("get_feature_columns: No numeric feature columns found in dataset after excluding target and metadata columns.", call. = FALSE)
  }

  numeric_cols
}

#' Add cyclical time features
#'
#' Adds sine/cosine encoding of Hour (and optionally Month) to a data.frame.
#'
#' @param df A data.frame
#' @param datetime_col Name of the datetime column
#' @param add_month Logical. Whether to also add month cyclical features.
#' @return The data.frame with added Hour_sin, Hour_cos (and optionally Month_sin, Month_cos).
#' @examples
#' data(microclimate_sample)
#' df <- add_cyclical_time(microclimate_sample, datetime_col = "time", add_month = TRUE)
#' head(df[, c("time", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos")])
#' @export
add_cyclical_time <- function(df, datetime_col = "time", add_month = FALSE) {
  if (!(datetime_col %in% names(df))) {
    trimmed_cand <- names(df)[trimws(names(df)) == trimws(datetime_col)]
    case_cand <- names(df)[tolower(names(df)) == tolower(datetime_col)]
    cand <- unique(c(trimmed_cand, case_cand))
    hint <- if (length(cand) > 0) sprintf(" Did you mean '%s'?", cand[1]) else ""
    stop(sprintf(
      "datetime_col '%s' not found in data.frame.%s Available columns: %s",
      datetime_col, hint, paste(names(df), collapse = ", ")
    ), call. = FALSE)
  }
  hours <- as.numeric(format(df[[datetime_col]], "%H"))
  df$Hour_sin <- sin(2 * pi * hours / 24)
  df$Hour_cos <- cos(2 * pi * hours / 24)
  if (add_month) {
    months <- as.numeric(format(df[[datetime_col]], "%m"))
    df$Month_sin <- sin(2 * pi * months / 12)
    df$Month_cos <- cos(2 * pi * months / 12)
  }
  df
}

#' Split data into train, validation, and test sets
#'
#' Splits by shuffled N-day blocks (recommended) or simple chronological split.
#' This is a faithful port of `train_val_test_split` from the Python pipeline.
#'
#' @param data A data.frame with a datetime column
#' @param train_pct Training fraction (0 to 1)
#' @param val_pct Validation fraction (0 to 1)
#' @param block_days Number of days per block for block-shuffle splitting
#' @param use_blocks Logical. If TRUE, use block-shuffle split. If FALSE, chronological.
#' @param datetime_col Name of the datetime column
#' @param seed Random seed
#' @param train_blocks Optional. Pre-defined training block indices.
#' @param val_blocks Optional. Pre-defined validation block indices.
#' @param test_blocks Optional. Pre-defined test block indices.
#' @return A list with elements: train, val, test (data.frames)
#' @examples
#' data(microclimate_sample)
#' splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
#' nrow(splits$train)
#' nrow(splits$val)
#' nrow(splits$test)
#' @export
split_train_val_test <- function(data,
                                 train_pct = 0.75,
                                 val_pct = 0.125,
                                 block_days = 7,
                                 use_blocks = TRUE,
                                 datetime_col = "time",
                                 seed = 123,
                                 train_blocks = NULL,
                                 val_blocks = NULL,
                                 test_blocks = NULL) {

  if (!(datetime_col %in% names(data))) {
    trimmed_cand <- names(data)[trimws(names(data)) == trimws(datetime_col)]
    case_cand <- names(data)[tolower(names(data)) == tolower(datetime_col)]
    cand <- unique(c(trimmed_cand, case_cand))
    hint <- if (length(cand) > 0) sprintf(" Did you mean '%s'?", cand[1]) else ""
    stop(sprintf(
      "datetime_col '%s' not found in data.%s Available columns: %s",
      datetime_col, hint, paste(names(data), collapse = ", ")
    ), call. = FALSE)
  }

  df <- data[order(data[[datetime_col]]), , drop = FALSE]

  if (use_blocks) {
    dates <- as.Date(df[[datetime_col]], tz = "UTC")
    day_index <- as.integer(dates - min(dates)) + 1L
    block <- (day_index - 1L) %/% block_days

    all_blocks <- unique(block)
    if (length(all_blocks) < 3 && is.null(train_blocks)) {
      warning(sprintf("Only %d blocks of %d days. Falling back to chronological split.",
                       length(all_blocks), block_days))
      use_blocks <- FALSE
    }
  }

  if (use_blocks) {
    if (!is.null(train_blocks) && !is.null(val_blocks) && !is.null(test_blocks)) {
      train_df <- df[block %in% train_blocks, , drop = FALSE]
      val_df   <- df[block %in% val_blocks, , drop = FALSE]
      test_df  <- df[block %in% test_blocks, , drop = FALSE]
    } else {
      set.seed(seed)
      all_blocks <- unique(block)
      blocks_shuffled <- sample(all_blocks)

      n_blocks <- length(blocks_shuffled)
      n_val_blocks   <- max(1L, floor(n_blocks * val_pct))
      n_test_blocks  <- max(1L, floor(n_blocks * (1.0 - train_pct - val_pct)))
      n_train_blocks <- n_blocks - n_val_blocks - n_test_blocks
      if (n_train_blocks <= 0) {
        n_train_blocks <- 1L
        n_val_blocks <- 1L
        n_test_blocks <- n_blocks - 2L
      }

      train_blocks <- blocks_shuffled[seq_len(n_train_blocks)]
      val_blocks   <- blocks_shuffled[(n_train_blocks + 1):(n_train_blocks + n_val_blocks)]
      test_blocks  <- blocks_shuffled[(n_train_blocks + n_val_blocks + 1):n_blocks]

      train_df <- df[block %in% train_blocks, , drop = FALSE]
      val_df   <- df[block %in% val_blocks, , drop = FALSE]
      test_df  <- df[block %in% test_blocks, , drop = FALSE]
    }

    # Re-sort within each split
    train_df <- train_df[order(train_df[[datetime_col]]), , drop = FALSE]
    val_df   <- val_df[order(val_df[[datetime_col]]), , drop = FALSE]
    test_df  <- test_df[order(test_df[[datetime_col]]), , drop = FALSE]

  } else {
    n <- nrow(df)
    end_train <- min(n, max(0L, floor(n * train_pct)))
    end_val   <- min(n, max(end_train, floor(n * (val_pct + train_pct))))

    idx_train <- if (end_train >= 1L) seq_len(end_train) else integer(0)
    idx_val   <- if (end_val > end_train) seq.int(end_train + 1L, end_val) else integer(0)
    idx_test  <- if (n > end_val) seq.int(end_val + 1L, n) else integer(0)

    train_df <- df[idx_train, , drop = FALSE]
    val_df   <- df[idx_val, , drop = FALSE]
    test_df  <- df[idx_test, , drop = FALSE]
  }

  list(train = train_df, val = val_df, test = test_df)
}


#' Stratified train/val/test split
#'
#' Splits by shuffled N-day blocks, ensuring balanced representation
#' across a stratification column (e.g., location or site).
#'
#' @param data A data.frame
#' @param train_pct Training fraction
#' @param val_pct Validation fraction
#' @param stratify_col Column name to stratify by
#' @param block_days Days per block
#' @param datetime_col Datetime column name
#' @param seed Random seed
#' @return List with train, val, test data.frames
#' @examples
#' data(microclimate_sample)
#' splits_s <- stratified_split_train_val_test(
#'   microclimate_sample,
#'   stratify_col = "time_series_doc",
#'   train_pct    = 0.6,
#'   val_pct      = 0.2,
#'   block_days   = 1,
#'   seed         = 42
#' )
#' nrow(splits_s$train)
#' nrow(splits_s$val)
#' nrow(splits_s$test)
#' @export
stratified_split_train_val_test <- function(data,
                                            train_pct = 0.75,
                                            val_pct = 0.125,
                                            stratify_col,
                                            block_days = 7,
                                            datetime_col = "time",
                                            seed = 123) {

  if (!(datetime_col %in% names(data))) {
    trimmed_cand <- names(data)[trimws(names(data)) == trimws(datetime_col)]
    case_cand <- names(data)[tolower(names(data)) == tolower(datetime_col)]
    cand <- unique(c(trimmed_cand, case_cand))
    hint <- if (length(cand) > 0) sprintf(" Did you mean '%s'?", cand[1]) else ""
    stop(sprintf(
      "datetime_col '%s' not found in data.%s Available columns: %s",
      datetime_col, hint, paste(names(data), collapse = ", ")
    ), call. = FALSE)
  }
  if (!(stratify_col %in% names(data))) {
    trimmed_cand <- names(data)[trimws(names(data)) == trimws(stratify_col)]
    case_cand <- names(data)[tolower(names(data)) == tolower(stratify_col)]
    cand <- unique(c(trimmed_cand, case_cand))
    hint <- if (length(cand) > 0) sprintf(" Did you mean '%s'?", cand[1]) else ""
    stop(sprintf(
      "stratify_col '%s' not found in data.%s Available columns: %s",
      stratify_col, hint, paste(names(data), collapse = ", ")
    ), call. = FALSE)
  }

  df <- data[order(data[[datetime_col]]), , drop = FALSE]

  set.seed(seed)
  train_rows <- integer(0)
  val_rows   <- integer(0)
  test_rows  <- integer(0)

  for (strat_val in unique(df[[stratify_col]])) {
    # Work with row indices into df so sites never contaminate each other
    group_idx <- which(df[[stratify_col]] == strat_val)
    group_df  <- df[group_idx, , drop = FALSE]

    # Blocks relative to this site's own date range
    dates     <- as.Date(group_df[[datetime_col]], tz = "UTC")
    day_index <- as.integer(dates - min(dates)) + 1L
    block     <- (day_index - 1L) %/% block_days

    blocks_shuffled <- sample(unique(block))
    n_b     <- length(blocks_shuffled)
    if (n_b < 3) {
      warning(sprintf(
        "Stratification group '%s' has only %d block(s) of %d days. Groups with fewer than 3 blocks cannot be partitioned across train/val/test; %s rows are assigned to test.",
        as.character(strat_val), n_b, block_days,
        if (floor(n_b * train_pct) == 0 && floor(n_b * val_pct) == 0) "all" else "some"
      ), call. = FALSE)
    }
    n_train <- floor(n_b * train_pct)
    n_val   <- floor(n_b * val_pct)
    n_test  <- n_b - n_train - n_val

    tb   <- blocks_shuffled[seq_len(n_train)]
    vb   <- if (n_val  > 0) blocks_shuffled[seq(n_train + 1,         n_train + n_val)] else integer(0)
    tesb <- if (n_test > 0) blocks_shuffled[seq(n_train + n_val + 1, n_b            )] else integer(0)

    train_rows <- c(train_rows, group_idx[block %in% tb])
    val_rows   <- c(val_rows,   group_idx[block %in% vb])
    test_rows  <- c(test_rows,  group_idx[block %in% tesb])
  }

  train_df <- df[train_rows, , drop = FALSE]
  val_df   <- df[val_rows,   , drop = FALSE]
  test_df  <- df[test_rows,  , drop = FALSE]

  train_df <- train_df[order(train_df[[datetime_col]]), , drop = FALSE]
  val_df   <- val_df[order(val_df[[datetime_col]]),     , drop = FALSE]
  test_df  <- test_df[order(test_df[[datetime_col]]),   , drop = FALSE]

  list(train = train_df, val = val_df, test = test_df)
}

#' MinMax scale features (fit to training data only)
#'
#' Performs MinMax scaling on feature columns. The scaler is fit only to
#' the training data to avoid data leakage.
#'
#' @param train Training data.frame
#' @param val Validation data.frame
#' @param test Test data.frame
#' @param avoid_cols Columns to exclude from scaling
#' @param target_col Target column name
#' @param microhabitat_col Microhabitat column name
#' @param prediction_col Prediction column name
#' @return List with scaled train, val, test data.frames and scaler info
#' @examples
#' data(microclimate_sample)
#' splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
#' scaled <- lstm_scaling(splits$train, splits$val, splits$test)
#' scaled$scaler$cols
#' @export
lstm_scaling <- function(train, val, test,
                         avoid_cols = .default_cols$avoid,
                         target_col = .default_cols$target,
                         microhabitat_col = .default_cols$microhabitat,
                         prediction_col = .default_cols$prediction) {

  train <- as.data.frame(train)
  val   <- as.data.frame(val)
  test  <- as.data.frame(test)

  cols_to_scale <- get_feature_columns(train, avoid_cols = avoid_cols,
                                       target_col = target_col,
                                       microhabitat_col = microhabitat_col,
                                       prediction_col = prediction_col)

  # Fit scaler on training data
  mins <- sapply(train[cols_to_scale], min, na.rm = TRUE)
  maxs <- sapply(train[cols_to_scale], max, na.rm = TRUE)
  ranges <- maxs - mins
  ranges[ranges == 0] <- 1  # avoid division by zero

  scaler <- list(min = mins, range = ranges, cols = cols_to_scale)

  # Reconcile any missing microhabitat columns in val or test (Issue 11)
  for (nm in c("val", "test")) {
    d <- get(nm)
    missing_cols <- setdiff(cols_to_scale, names(d))
    if (length(missing_cols) > 0) {
      is_micro <- grepl(paste0("^", microhabitat_col, "_"), missing_cols)
      if (any(is_micro)) {
        for (mc in missing_cols[is_micro]) {
          lvl <- sub(paste0("^", microhabitat_col, "_"), "", mc)
          if (microhabitat_col %in% names(d)) {
            d[[mc]] <- as.numeric(as.character(d[[microhabitat_col]]) == lvl)
          } else {
            d[[mc]] <- 0
          }
        }
      }
      missing_remaining <- setdiff(cols_to_scale, names(d))
      if (length(missing_remaining) > 0) {
        stop(sprintf("Feature column(s) '%s' present in training data were not found in %s dataset.",
                     paste(missing_remaining, collapse = ", "), nm), call. = FALSE)
      }
      assign(nm, d)
    }
  }

  # Scale all three datasets
  for (col in cols_to_scale) {
    train[[col]] <- (train[[col]] - scaler$min[col]) / scaler$range[col]
    val[[col]]   <- (val[[col]]   - scaler$min[col]) / scaler$range[col]
    test[[col]]  <- (test[[col]]  - scaler$min[col]) / scaler$range[col]
  }

  list(train = train, val = val, test = test, scaler = scaler)
}

#' Create sliding windows for LSTM input
#'
#' Rearranges time-series data into sliding windows of fixed size.
#' Skips windows that contain temporal gaps larger than max_gap_hours.
#'
#' @param X_mat Matrix of features (n_samples x n_features)
#' @param y_vec Vector of target values
#' @param base_pred_vec Vector of base predictions
#' @param datetime_vec Vector of POSIXct datetimes
#' @param window_size Integer window length
#' @param max_gap_hours Maximum allowed gap in hours between consecutive points. NULL to disable.
#' @return List with X (3D array), y, base_pred, datetime
#' @examples
#' data(microclimate_sample)
#' feat_cols <- c("TAREF", "RH", "VREF", "SOLR")
#' win <- make_windows(
#'   X_mat         = as.matrix(microclimate_sample[1:100, feat_cols]),
#'   y_vec         = microclimate_sample$residual[1:100],
#'   base_pred_vec = microclimate_sample$predicted[1:100],
#'   datetime_vec  = microclimate_sample$time[1:100],
#'   window_size   = 6,
#'   max_gap_hours = 1
#' )
#' dim(win$X)
#' length(win$y)
#' @export
make_windows <- function(X_mat, y_vec, base_pred_vec, datetime_vec,
                         window_size, max_gap_hours = 1) {

  if (!is.numeric(window_size) || length(window_size) != 1L || is.na(window_size) || window_size <= 0) {
    stop("window_size must be a positive integer, got: ", window_size, call. = FALSE)
  }
  window_size <- as.integer(window_size)

  n <- nrow(X_mat)
  if (n < window_size) {
    return(list(
      X = array(numeric(0), dim = c(0, window_size, ncol(X_mat))),
      y = numeric(0),
      base_pred = numeric(0),
      datetime = as.POSIXct(character(0))
    ))
  }

  X_windows <- list()
  y_windows <- c()
  base_pred_windows <- c()
  datetime_windows <- c()

  datetime_num <- as.numeric(datetime_vec)

  for (i in seq_len(n - window_size + 1)) {
    idx <- i:(i + window_size - 1)

    # Check for time gaps
    if (!is.null(max_gap_hours)) {
      time_diffs <- diff(datetime_num[idx]) / 3600  # seconds to hours
      if (any(time_diffs > max_gap_hours)) next
    }

    X_windows[[length(X_windows) + 1]] <- X_mat[idx, , drop = FALSE]
    y_windows <- c(y_windows, y_vec[i + window_size - 1])
    base_pred_windows <- c(base_pred_windows, base_pred_vec[i + window_size - 1])
    datetime_windows <- c(datetime_windows, datetime_num[i + window_size - 1])
  }

  n_windows <- length(X_windows)
  n_features <- ncol(X_mat)

  if (n_windows == 0) {
    return(list(
      X = array(numeric(0), dim = c(0, window_size, n_features)),
      y = numeric(0),
      base_pred = numeric(0),
      datetime = as.POSIXct(character(0))
    ))
  }

  # Stack into 3D array: (n_windows, window_size, n_features)
  X_arr <- array(NA_real_, dim = c(n_windows, window_size, n_features))
  for (w in seq_len(n_windows)) {
    X_arr[w, , ] <- as.matrix(X_windows[[w]])
  }

  list(
    X = X_arr,
    y = y_windows,
    base_pred = base_pred_windows,
    datetime = as.POSIXct(datetime_windows, origin = "1970-01-01", tz = "UTC")
  )
}

#' LSTM-specific preprocessing for one dataset
#'
#' Splits a dataset by time-series site, creates windows for each,
#' and concatenates them.
#'
#' @param data_set A data.frame
#' @param window_size Integer window size
#' @param unique_ts_sites Character vector of unique time-series site names
#' @param ts_names_col Column containing site identifiers
#' @param avoid_cols Columns to exclude from features
#' @param target_col Target column name
#' @param microhabitat_col Microhabitat column name
#' @param prediction_col Prediction column name
#' @param datetime_col Datetime column name
#' @return List with dataset_dict and idx_per_ts
#' @keywords internal
one_dataset_lstm_preprocessing <- function(data_set, window_size, unique_ts_sites,
                                           ts_names_col,
                                           avoid_cols = .default_cols$avoid,
                                           target_col = .default_cols$target,
                                           microhabitat_col = .default_cols$microhabitat,
                                           prediction_col = .default_cols$prediction,
                                           datetime_col = .default_cols$datetime) {

  X_list <- list()
  y_list <- list()
  bp_list <- list()
  dt_list <- list()
  idx_per_ts <- setNames(vector("list", length(unique_ts_sites)), unique_ts_sites)
  for (s in unique_ts_sites) idx_per_ts[[s]] <- integer(0)
  pos <- 0L

  feature_cols <- get_feature_columns(data_set, avoid_cols = avoid_cols,
                                      target_col = target_col,
                                      microhabitat_col = microhabitat_col,
                                      prediction_col = prediction_col)

  for (ts_site in unique_ts_sites) {
    ts_df <- data_set[data_set[[ts_names_col]] == ts_site, , drop = FALSE]
    if (nrow(ts_df) == 0) {
      idx_per_ts[[ts_site]] <- integer(0)
      next
    }

    X_mat <- as.matrix(ts_df[, feature_cols, drop = FALSE])
    y_vec <- ts_df[[target_col]]
    pred_vec <- ts_df[[prediction_col]]
    dt_vec <- ts_df[[datetime_col]]

    win <- make_windows(X_mat, y_vec, pred_vec, dt_vec, window_size)

    if (length(win$y) == 0) {
      idx_per_ts[[ts_site]] <- integer(0)
      next
    }

    n_win <- length(win$y)
    idx_per_ts[[ts_site]] <- seq(pos, pos + n_win - 1L)
    pos <- pos + n_win

    X_list[[length(X_list) + 1]] <- win$X
    y_list <- c(y_list, list(win$y))
    bp_list <- c(bp_list, list(win$base_pred))
    dt_list <- c(dt_list, list(win$datetime))
  }

  if (length(X_list) == 0) {
    n_feat <- length(feature_cols)
    return(list(
      dataset_dict = list(
        X = array(numeric(0), dim = c(0, window_size, n_feat)),
        y = numeric(0), base_pred = numeric(0),
        datetime = as.POSIXct(character(0))
      ),
      idx_per_ts = idx_per_ts
    ))
  }

  # Concatenate along first dimension
  total_windows <- sum(sapply(X_list, function(x) dim(x)[1]))
  n_feat <- dim(X_list[[1]])[3]
  X_all <- array(NA_real_, dim = c(total_windows, window_size, n_feat))
  offset <- 0L
  for (x in X_list) {
    nw <- dim(x)[1]
    X_all[(offset + 1):(offset + nw), , ] <- x
    offset <- offset + nw
  }

  list(
    dataset_dict = list(
      X = X_all,
      y = unlist(y_list),
      base_pred = unlist(bp_list),
      datetime = do.call(c, dt_list)
    ),
    idx_per_ts = idx_per_ts
  )
}

#' LSTM-specific preprocessing for train/val/test
#'
#' Creates windowed datasets for all three splits, tracking per-site indices.
#'
#' @param train Scaled training data.frame
#' @param val Scaled validation data.frame
#' @param test Scaled test data.frame
#' @param window_size Window size for LSTM
#' @param ts_names_col Column with site/time-series identifiers
#' @return List with train_dict, val_dict, test_dict, index_info
#' @examples
#' data(microclimate_sample)
#' splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
#' scaled <- lstm_scaling(splits$train, splits$val, splits$test)
#' lstm_data <- lstm_specific_preprocessing(
#'   scaled$train, scaled$val, scaled$test,
#'   window_size  = 4,
#'   ts_names_col = "time_series_doc"
#' )
#' dim(lstm_data$train_dict$X)
#' @export
lstm_specific_preprocessing <- function(train, val, test, window_size,
                                        ts_names_col = "time_series_doc") {

  for (nm in c("train", "val", "test")) {
    d <- get(nm)
    if (!(ts_names_col %in% names(d))) {
      trimmed_cand <- names(d)[trimws(names(d)) == trimws(ts_names_col)]
      hint <- if (length(trimmed_cand) > 0) sprintf(" Did you mean '%s'?", trimmed_cand[1]) else ""
      stop(sprintf("ts_names_col '%s' not found in %s dataset.%s Available columns: %s",
                   ts_names_col, nm, hint, paste(names(d), collapse = ", ")), call. = FALSE)
    }
  }

  unique_sites <- unique(c(train[[ts_names_col]], val[[ts_names_col]], test[[ts_names_col]]))

  train_res <- one_dataset_lstm_preprocessing(train, window_size, unique_sites, ts_names_col)
  val_res   <- one_dataset_lstm_preprocessing(val, window_size, unique_sites, ts_names_col)
  test_res  <- one_dataset_lstm_preprocessing(test, window_size, unique_sites, ts_names_col)

  index_info <- list(
    datasets      = unique_sites,
    train_indices = train_res$idx_per_ts,
    val_indices   = val_res$idx_per_ts,
    test_indices  = test_res$idx_per_ts
  )

  list(
    train_dict = train_res$dataset_dict,
    val_dict   = val_res$dataset_dict,
    test_dict  = test_res$dataset_dict,
    index_info = index_info
  )
}

#' Align RF test set with LSTM test set
#'
#' Filters the point-based test dataset to only include rows that were
#' successfully processed as window endpoints by the LSTM windowing.
#'
#' @param test_dataset Original test data.frame
#' @param lstm_test_dict LSTM test dictionary (with datetime element)
#' @param ts_index_info Index info from lstm_specific_preprocessing
#' @param site_name_col Site name column
#' @param datetime_col Datetime column
#' @return Filtered test data.frame aligned to LSTM endpoints
#' @examples
#' data(microclimate_sample)
#' splits <- split_train_val_test(microclimate_sample, train_pct = 0.6, val_pct = 0.2, block_days = 2, seed = 42)
#' scaled <- lstm_scaling(splits$train, splits$val, splits$test)
#' prep <- lstm_specific_preprocessing(scaled$train, scaled$val, scaled$test,
#'                                     window_size = 2, ts_names_col = "time_series_doc")
#' aligned <- align_test_sets(splits$test, prep$test_dict, prep$index_info, "time_series_doc")
#' nrow(aligned)
#' @export
align_test_sets <- function(test_dataset, lstm_test_dict, ts_index_info,
                            site_name_col, datetime_col = "time") {

  if (!(site_name_col %in% names(test_dataset))) {
    trimmed_cand <- names(test_dataset)[trimws(names(test_dataset)) == trimws(site_name_col)]
    hint <- if (length(trimmed_cand) > 0) sprintf(" Did you mean '%s'?", trimmed_cand[1]) else ""
    stop(sprintf("site_name_col '%s' not found in test_dataset.%s Available columns: %s",
                 site_name_col, hint, paste(names(test_dataset), collapse = ", ") ), call. = FALSE)
  }
  if (!(datetime_col %in% names(test_dataset))) {
    trimmed_cand <- names(test_dataset)[trimws(names(test_dataset)) == trimws(datetime_col)]
    hint <- if (length(trimmed_cand) > 0) sprintf(" Did you mean '%s'?", trimmed_cand[1]) else ""
    stop(sprintf("datetime_col '%s' not found in test_dataset.%s Available columns: %s",
                 datetime_col, hint, paste(names(test_dataset), collapse = ", ")), call. = FALSE)
  }

  # Reconstruct site names for each LSTM window
  test_sites <- character(length(lstm_test_dict$datetime))
  for (i in seq_along(ts_index_info$datasets)) {
    site_name <- ts_index_info$datasets[i]
    test_idx <- if (!is.null(names(ts_index_info$test_indices))) {
      ts_index_info$test_indices[[site_name]]
    } else if (i <= length(ts_index_info$test_indices)) {
      ts_index_info$test_indices[[i]]
    } else {
      integer(0)
    }
    if (length(test_idx) > 0) {
      test_sites[test_idx + 1L] <- site_name  # +1 for R 1-indexing
    }
  }

  # Create whitelist
  lstm_keys <- data.frame(
    dt   = lstm_test_dict$datetime,
    site = test_sites,
    .order = seq_along(lstm_test_dict$datetime),
    stringsAsFactors = FALSE
  )
  names(lstm_keys)[1:2] <- c(datetime_col, site_name_col)

  # Check and handle duplicates in test_dataset to prevent many-to-many fan-out
  dup_mask <- duplicated(test_dataset[c(datetime_col, site_name_col)])
  if (any(dup_mask)) {
    warning(sprintf(
      "Found %d duplicate (%s, %s) timestamp(s) in test_dataset; deduplicating to avoid merge fan-out.",
      sum(dup_mask), datetime_col, site_name_col
    ), call. = FALSE)
    test_dataset <- test_dataset[!dup_mask, , drop = FALSE]
  }

  # Merge (inner join preserving exact LSTM order)
  aligned <- merge(lstm_keys, test_dataset, by = c(datetime_col, site_name_col), sort = FALSE)
  aligned <- aligned[order(aligned$.order), , drop = FALSE]
  aligned$.order <- NULL
  aligned
}
