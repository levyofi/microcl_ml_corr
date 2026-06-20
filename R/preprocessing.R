# ---- microclCorr: Preprocessing ----
# Port of preprocessing.py from the Python pipeline

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
#' @return A data.frame with parsed datetime and (optionally) one-hot encoded microhabitat.
#' @export
load_prepared_csv_data <- function(path,
                                   is_continuous_microhabitat = FALSE,
                                   datetime_format = "%Y-%m-%d %H:%M:%S",
                                   includes_index = TRUE,
                                   microhabitat_col = "microhabitat",
                                   datetime_col = "time") {

  if (includes_index) {
    df <- read.csv(path, row.names = 1, stringsAsFactors = FALSE)
  } else {
    df <- read.csv(path, stringsAsFactors = FALSE)
  }

  # One-hot encode categorical microhabitat
  if (!is_continuous_microhabitat && microhabitat_col %in% names(df)) {
    orig_micro <- df[[microhabitat_col]]
    df[[microhabitat_col]] <- NULL
    levels_micro <- unique(orig_micro)
    for (lvl in levels_micro) {
      df[[paste0(microhabitat_col, "_", lvl)]] <- as.numeric(orig_micro == lvl)
    }
    df[[microhabitat_col]] <- orig_micro
  }

  # Fix midnight timestamps that lack time component
  time_vals <- df[[datetime_col]]
  no_colon <- !grepl(":", time_vals, fixed = TRUE)
  if (datetime_format == "%Y-%m-%d %H:%M:%S") {
    time_vals[no_colon] <- paste0(time_vals[no_colon], " 0:00:00")
  } else if (datetime_format == "%d/%m/%Y %H:%M") {
    time_vals[no_colon] <- paste0(time_vals[no_colon], " 0:00")
  }

  df[[datetime_col]] <- as.POSIXct(time_vals, format = datetime_format, tz = "UTC")
  df <- df[complete.cases(df), , drop = FALSE]

  return(df)
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
#' @export
get_feature_columns <- function(df,
                                avoid_cols = .default_cols$avoid,
                                target_col = .default_cols$target,
                                microhabitat_col = .default_cols$microhabitat,
                                prediction_col = .default_cols$prediction) {
  cols_to_exclude <- unique(c(avoid_cols, target_col, microhabitat_col, prediction_col))
  setdiff(names(df), cols_to_exclude)
}

#' Add cyclical time features
#'
#' Adds sine/cosine encoding of Hour (and optionally Month) to a data.frame.
#'
#' @param df A data.frame
#' @param datetime_col Name of the datetime column
#' @param add_month Logical. Whether to also add month cyclical features.
#' @return The data.frame with added Hour_sin, Hour_cos (and optionally Month_sin, Month_cos).
#' @export
add_cyclical_time <- function(df, datetime_col = "time", add_month = FALSE) {
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
    end_train <- floor(n * train_pct)
    end_val   <- floor(n * (val_pct + train_pct))

    train_df <- df[seq_len(end_train), , drop = FALSE]
    val_df   <- df[(end_train + 1):end_val, , drop = FALSE]
    test_df  <- df[(end_val + 1):n, , drop = FALSE]
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
#' @export
stratified_split_train_val_test <- function(data,
                                            train_pct = 0.75,
                                            val_pct = 0.125,
                                            stratify_col,
                                            block_days = 7,
                                            datetime_col = "time",
                                            seed = 123) {

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
#' @export
make_windows <- function(X_mat, y_vec, base_pred_vec, datetime_vec,
                         window_size, max_gap_hours = 1) {

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
  idx_per_ts <- list()
  pos <- 0L

  feature_cols <- get_feature_columns(data_set, avoid_cols = avoid_cols,
                                      target_col = target_col,
                                      microhabitat_col = microhabitat_col,
                                      prediction_col = prediction_col)

  for (ts_site in unique_ts_sites) {
    ts_df <- data_set[data_set[[ts_names_col]] == ts_site, , drop = FALSE]
    if (nrow(ts_df) == 0) next

    X_mat <- as.matrix(ts_df[, feature_cols, drop = FALSE])
    y_vec <- ts_df[[target_col]]
    pred_vec <- ts_df[[prediction_col]]
    dt_vec <- ts_df[[datetime_col]]

    win <- make_windows(X_mat, y_vec, pred_vec, dt_vec, window_size)

    if (length(win$y) == 0) next

    n_win <- length(win$y)
    idx_per_ts[[length(idx_per_ts) + 1]] <- seq(pos, pos + n_win - 1L)
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
      idx_per_ts = list()
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
#' @export
lstm_specific_preprocessing <- function(train, val, test, window_size,
                                        ts_names_col = "time_series_doc") {

  unique_sites <- unique(train[[ts_names_col]])

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
#' @export
align_test_sets <- function(test_dataset, lstm_test_dict, ts_index_info,
                            site_name_col, datetime_col = "time") {

  # Reconstruct site names for each LSTM window
  test_sites <- character(length(lstm_test_dict$datetime))
  for (i in seq_along(ts_index_info$datasets)) {
    site_name <- ts_index_info$datasets[i]
    test_idx  <- ts_index_info$test_indices[[i]]
    if (length(test_idx) > 0) {
      test_sites[test_idx + 1L] <- site_name  # +1 for R 1-indexing
    }
  }

  # Create whitelist
  lstm_keys <- data.frame(
    dt   = lstm_test_dict$datetime,
    site = test_sites,
    stringsAsFactors = FALSE
  )
  names(lstm_keys) <- c(datetime_col, site_name_col)

  # Merge (inner join preserving LSTM order)
  aligned <- merge(lstm_keys, test_dataset, by = c(datetime_col, site_name_col))
  aligned
}
