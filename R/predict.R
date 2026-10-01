# ---- microclCorr: Prediction / Correction ----

#' Apply a trained correction model to new data
#'
#' Takes a trained model (Random Forest or LSTM) and new NicheMapR microclimate
#' simulation data, predicts residual error corrections, and returns corrected
#' predictions alongside base predictions and timestamps.
#'
#' @details
#' \subsection{What to do before calling \code{correct_predictions}}{
#' \enumerate{
#'   \item \strong{Prepare new data in original units (unscaled)}:
#'     Ensure \code{new_data} is a \code{data.frame} containing the base model
#'     predictions (\code{prediction_col}, default \code{"predicted"}), timestamps
#'     (\code{datetime_col}, default \code{"time"}), and all predictor variables
#'     (\code{feature_cols}). If loaded from CSV, use \code{\link{load_prepared_csv_data}};
#'     if already in-memory, use \code{\link{prepare_dataframe}}.
#'     \emph{Important:} Do NOT pre-scale \code{new_data} for LSTM; pass raw unscaled
#'     values because \code{correct_predictions} scales them internally using the saved \code{scaler}.
#'   \item \strong{Add engineered time features if used during training}:
#'     If cyclical time features (\code{Hour_sin}, \code{Hour_cos}, \code{Month_sin}, etc.)
#'     were included in your training features, call \code{\link{add_cyclical_time}} on \code{new_data} first.
#'   \item \strong{Ensure matching microhabitat levels}:
#'     If categorical microhabitat was one-hot encoded, provide \code{microhabitat_levels}
#'     to \code{\link{prepare_dataframe}} (or \code{\link{load_prepared_csv_data}}) or retain the raw \code{microhabitat} column.
#'     Any one-hot levels present in training but absent in \code{new_data} will be
#'     automatically synthesized as 0 with an informative message.
#'   \item \strong{Requirements by Model Type}:
#'     \itemize{
#'       \item \strong{Random Forest (\code{model_type = "rf"})}: Requires \code{model}
#'         and \code{feature_cols}. Does not require \code{scaler} or \code{window_size}.
#'       \item \strong{LSTM (\code{model_type = "lstm"})}: Requires \code{model},
#'         \code{scaler} (the scaler fit on training data from \code{\link{lstm_scaling}} or
#'         \code{\link{load_correction_model}}), the same \code{window_size} used during training,
#'         and \code{feature_cols}. \code{new_data} must be ordered chronologically so
#'         sliding windows can be created.
#'     }
#' }
#' }
#'
#' @param model Trained model (\code{ranger} or \code{keras3}).
#' @param new_data Data.frame with features and base predictions in original (unscaled) units.
#' @param model_type Character: \code{"rf"} for Random Forest or \code{"lstm"} for LSTM.
#' @param scaler Scaler object from \code{\link{lstm_scaling}} or \code{\link{load_correction_model}}
#'   (required for LSTM; ignored for RF).
#' @param feature_cols Character vector of feature column names used during training.
#'   If \code{NULL}, inferred using \code{\link{get_feature_columns}}.
#' @param prediction_col Name of base prediction column (default \code{"predicted"}).
#' @param window_size Integer window size used during LSTM training (default 2; required for LSTM).
#' @param datetime_col Name of the datetime column (default \code{"time"}).
#' @param microhabitat_col Microhabitat column name (default \code{"microhabitat"}).
#' @return Data.frame with columns:
#'   \item{datetime}{Timestamps matching prediction points}
#'   \item{base_prediction}{Original uncorrected predictions from NicheMapR}
#'   \item{correction}{Predicted residual error corrections}
#'   \item{corrected_prediction}{Final corrected microclimate predictions (\code{base_prediction + correction})}
#'
#' @examples
#' # =========================================================================
#' # Example 1: Random Forest (RF) Correction Pipeline
#' # =========================================================================
#' data(microclimate_sample)
#'
#' # Step 1: Define features used during training
#' feat_cols <- c("TAREF", "RH", "VREF", "SOLR")
#'
#' # Step 2: Train RF model on training subset
#' train_data <- microclimate_sample[1:300, ]
#' rf <- train_rf(
#'   train_X   = train_data[, feat_cols],
#'   train_y   = train_data$residual,
#'   num_trees = 10,
#'   tune      = FALSE
#' )
#'
#' # Step 3: Simulate another CSV containing new NicheMapR microclimate predictions
#' # (In real use cases, this is your new/target NicheMapR output CSV file)
#' new_csv <- tempfile(fileext = ".csv")
#' write.csv(microclimate_sample[301:500, ], new_csv, row.names = TRUE)
#'
#' # Step 4: Load the new CSV using load_prepared_csv_data()
#' new_data_rf <- load_prepared_csv_data(new_csv)
#'
#' # Step 5: Apply model to correct predictions from the loaded CSV
#' corrected_rf <- correct_predictions(
#'   model        = rf,
#'   new_data     = new_data_rf,
#'   model_type   = "rf",
#'   feature_cols = feat_cols
#' )
#' head(corrected_rf)
#'
#' \dontrun{
#' # =========================================================================
#' # Example 2: LSTM Correction Pipeline with Loaded CSV
#' # =========================================================================
#' if (check_lstm_environment(error = FALSE)) {
#'   # Step 1: Split and scale training data, saving the fitted scaler
#'   splits <- split_train_val_test(microclimate_sample, block_days = 2, seed = 42)
#'   scaled <- lstm_scaling(splits$train, splits$val, splits$test)
#'   saved_scaler <- scaled$scaler
#'   win_size <- 4
#'
#'   # Step 2: Prepare windowed training tensors and train LSTM
#'   prep <- lstm_specific_preprocessing(
#'     scaled$train, scaled$val, scaled$test,
#'     window_size  = win_size,
#'     ts_names_col = "time_series_doc"
#'   )
#'   lstm_model <- train_lstm(
#'     train_X    = prep$train_dict$X,
#'     train_y    = prep$train_dict$y,
#'     val_X      = prep$val_dict$X,
#'     val_y      = prep$val_dict$y,
#'     epochs     = 5,
#'     seed       = 42
#'   )
#'
#'   # Step 3: Simulate another CSV containing new NicheMapR microclimate predictions
#'   new_csv_lstm <- tempfile(fileext = ".csv")
#'   write.csv(splits$test, new_csv_lstm, row.names = TRUE)
#'
#'   # Step 4: Load the new CSV using load_prepared_csv_data() (kept unscaled!)
#'   new_data_lstm <- load_prepared_csv_data(new_csv_lstm)
#'
#'   # Step 5: Apply model to correct predictions using the saved training scaler
#'   corrected_lstm <- correct_predictions(
#'     model        = lstm_model,
#'     new_data     = new_data_lstm,
#'     model_type   = "lstm",
#'     scaler       = saved_scaler,
#'     window_size  = win_size,
#'     feature_cols = saved_scaler$cols
#'   )
#'   head(corrected_lstm)
#' }
#' }
#' @export
correct_predictions <- function(model, new_data,
                                model_type = c("rf", "lstm"),
                                scaler = NULL,
                                feature_cols = NULL,
                                prediction_col = "predicted",
                                window_size = 2,
                                datetime_col = "time",
                                microhabitat_col = "microhabitat") {
  model_type <- match.arg(model_type)

  # Validate prediction_col and datetime_col (Issue 12)
  if (!(prediction_col %in% names(new_data))) {
    trimmed_cand <- names(new_data)[trimws(names(new_data)) == trimws(prediction_col)]
    hint <- if (length(trimmed_cand) > 0) sprintf(" Did you mean '%s'? (Check whitespace)", trimmed_cand[1]) else ""
    stop(sprintf(
      "prediction_col '%s' not found in 'new_data'.%s Available columns: %s",
      prediction_col, hint, paste(names(new_data), collapse = ", ")
    ), call. = FALSE)
  }
  if (!(datetime_col %in% names(new_data))) {
    trimmed_cand <- names(new_data)[trimws(names(new_data)) == trimws(datetime_col)]
    hint <- if (length(trimmed_cand) > 0) sprintf(" Did you mean '%s'? (Check whitespace)", trimmed_cand[1]) else ""
    stop(sprintf(
      "datetime_col '%s' not found in 'new_data'.%s Available columns: %s",
      datetime_col, hint, paste(names(new_data), collapse = ", ")
    ), call. = FALSE)
  }

  if (is.null(feature_cols)) {
    feature_cols <- get_feature_columns(new_data, microhabitat_col = microhabitat_col, prediction_col = prediction_col)
  }

  # Validate and reconcile feature_cols (Issue 11 & Issue 12)
  missing_cols <- setdiff(feature_cols, names(new_data))
  if (length(missing_cols) > 0) {
    # Check if any missing columns are one-hot microhabitat columns
    is_micro <- grepl(paste0("^", microhabitat_col, "_"), missing_cols) | grepl("^microhabitat_", missing_cols)
    if (any(is_micro)) {
      micro_missing <- missing_cols[is_micro]
      for (mcol in micro_missing) {
        prefix <- if (grepl(paste0("^", microhabitat_col, "_"), mcol)) paste0("^", microhabitat_col, "_") else "^microhabitat_"
        lvl <- sub(prefix, "", mcol)
        if (microhabitat_col %in% names(new_data)) {
          new_data[[mcol]] <- as.numeric(as.character(new_data[[microhabitat_col]]) == lvl)
        } else {
          new_data[[mcol]] <- 0
        }
      }
      message(sprintf(
        "Synthesized missing one-hot microhabitat feature column(s) (%s) in 'new_data' with 0 for absent levels.",
        paste(micro_missing, collapse = ", ")
      ))
      missing_cols <- setdiff(feature_cols, names(new_data))
    }

    if (length(missing_cols) > 0) {
      trimmed_cand <- names(new_data)[trimws(names(new_data)) %in% trimws(missing_cols)]
      hint <- if (length(trimmed_cand) > 0) {
        sprintf(" Possible whitespace mismatch with: %s.", paste(trimmed_cand, collapse = ", "))
      } else {
        ""
      }
      stop(sprintf(
        "The following required feature columns specified in 'feature_cols' are missing from 'new_data': %s.%s\nAvailable columns: %s",
        paste(missing_cols, collapse = ", "),
        hint,
        paste(names(new_data), collapse = ", ")
      ), call. = FALSE)
    }
  }

  if (model_type == "rf") {
    if (!isNamespaceLoaded("ranger")) {
      loadNamespace("ranger")
    }
    X <- new_data[, feature_cols, drop = FALSE]
    pred_res <- stats::predict(model, data = as.data.frame(X))$predictions
    base <- new_data[[prediction_col]]

    result <- data.frame(
      datetime = new_data[[datetime_col]],
      base_prediction = base,
      correction = pred_res,
      corrected_prediction = base + pred_res,
      stringsAsFactors = FALSE
    )

  } else {
    check_keras3()
    # LSTM: need to scale and window
    scaled_data <- new_data
    if (!is.null(scaler)) {
      for (col in scaler$cols) {
        if (col %in% names(scaled_data)) {
          scaled_data[[col]] <- (scaled_data[[col]] - scaler$min[col]) / scaler$range[col]
        }
      }
    }

    X_mat <- as.matrix(scaled_data[, feature_cols, drop = FALSE])
    y_vec <- rep(0, nrow(scaled_data))  # placeholder
    base_vec <- scaled_data[[prediction_col]]
    dt_vec <- scaled_data[[datetime_col]]

    win <- make_windows(X_mat, y_vec, base_vec, dt_vec, window_size)

    if (length(win$y) == 0) {
      warning("No valid windows could be created from the data")
      return(data.frame())
    }

    pred_res <- as.numeric(model |> keras3::predict_on_batch(win$X))
    # Note: base_pred from windowing is the SCALED version, we need unscaled
    # The base predictions are not scaled (prediction_col is in avoid_cols)
    # So win$base_pred already has the correct values
    base <- win$base_pred

    result <- data.frame(
      datetime = win$datetime,
      base_prediction = base,
      correction = pred_res,
      corrected_prediction = base + pred_res,
      stringsAsFactors = FALSE
    )
  }

  result
}
