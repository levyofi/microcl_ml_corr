# ---- microclCorr: Shared Utilities ----

#' Default column names used throughout the package
#' @keywords internal
.default_cols <- list(
  datetime    = "time",
  target      = "residual",
  prediction  = "predicted",
  microhabitat = "microhabitat",
  ts_names    = "time_series_doc",
  avoid       = c("time_series_doc", "time_series_site", "TIME", "time",
                   "location", "site_id")
)

#' Save a correction model to disk
#'
#' RF models are saved entirely in the .rds file. Keras (LSTM) models are
#' saved to a temporary .keras file, read as raw bytes, and embedded into the 
#' .rds file so they can be reloaded seamlessly as a single file by any user.
#'
#' @param model Trained model (ranger or keras)
#' @param scaler List with min/max from scaling
#' @param feature_cols Character vector of feature column names
#' @param path File path to save to (.rds extension)
#' @export
save_correction_model <- function(model, scaler, feature_cols, path) {
  is_rf      <- inherits(model, "ranger")
  model_type <- if (is_rf) "rf" else "lstm"

  keras_bytes <- NULL
  if (!is_rf) {
    keras_tmp <- tempfile(fileext = ".keras")
    keras3::save_model(model, keras_tmp)
    keras_bytes <- readBin(keras_tmp, "raw", file.info(keras_tmp)$size)
    unlink(keras_tmp)
    model <- NULL # don't embed the Python pointer
  }

  obj <- list(
    model        = model,
    keras_bytes  = keras_bytes,
    scaler       = scaler,
    feature_cols = feature_cols,
    model_type   = model_type
  )
  saveRDS(obj, path)
  invisible(path)
}

#' Load a correction model from disk
#'
#' @param path File path to the .rds model
#' @return List with model, scaler, feature_cols, model_type
#' @export
load_correction_model <- function(path) {
  obj <- readRDS(path)
  if (identical(obj$model_type, "lstm")) {
    keras_path <- sub("\\.rds$", ".keras", path)
    if (file.exists(keras_path)) {
      obj$model <- keras3::load_model(keras_path)
    } else if (!is.null(obj$keras_bytes)) {
      keras_tmp <- tempfile(fileext = ".keras")
      writeBin(obj$keras_bytes, keras_tmp)
      obj$model <- keras3::load_model(keras_tmp)
      unlink(keras_tmp)
    } else {
      # Fallback for models saved with serialize_keras_object
      tryCatch({
        obj$model <- keras3::deserialize_keras_object(obj$model)
      }, error = function(e) {
        stop("Failed to load LSTM model. Dead python pointer and no side-by-side .keras file found.")
      })
    }
  }
  obj
}
