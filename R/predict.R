# ---- microclCorr: Prediction / Correction ----

#' Apply a trained correction model to new data
#'
#' Takes a trained model and new NicheMapR output data, predicts the
#' residual correction, and returns corrected predictions.
#'
#' @param model Trained model (ranger or keras)
#' @param new_data Data.frame with features and base predictions
#' @param model_type Character: "rf" or "lstm"
#' @param scaler Scaler object from lstm_scaling (required for LSTM)
#' @param feature_cols Feature column names
#' @param prediction_col Name of base prediction column
#' @param window_size Window size (for LSTM only)
#' @param datetime_col Datetime column name (for LSTM windowing)
#' @return Data.frame with base_prediction, correction, corrected_prediction
#' @export
correct_predictions <- function(model, new_data,
                                model_type = c("rf", "lstm"),
                                scaler = NULL,
                                feature_cols = NULL,
                                prediction_col = "predicted",
                                window_size = 2,
                                datetime_col = "time") {
  model_type <- match.arg(model_type)

  if (is.null(feature_cols)) {
    feature_cols <- get_feature_columns(new_data)
  }

  if (model_type == "rf") {
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
