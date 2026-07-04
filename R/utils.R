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
#' @param model Trained model (ranger or keras)
#' @param scaler List with min/max from scaling
#' @param feature_cols Character vector of feature column names
#' @param path File path to save to (will create .rds file)
#' @export
save_correction_model <- function(model, scaler, feature_cols, path) {
  obj <- list(
    model        = model,
    scaler       = scaler,
    feature_cols = feature_cols,
    model_type   = if (inherits(model, "ranger")) "rf" else "lstm"
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
  # If it's an LSTM, load the .keras file
  if (obj$model_type == "lstm" || (is.character(obj$model) && length(obj$model) > 0)) {
    keras_path <- sub("\\.rds$", ".keras", path)
    if (file.exists(keras_path)) {
      obj$model <- keras3::load_model(keras_path)
    }
  }
  obj
}

#' Setup Tensorflow environment
#' @export
setup_tensorflow <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    install.packages("reticulate")
  }
  
  if (Sys.getenv("KERAS_HOME") == "") {
    Sys.setenv(KERAS_HOME = normalizePath("."))
  }
  
  env_name <- "microcl_env"
  if (!reticulate::virtualenv_exists(env_name)) {
    reticulate::virtualenv_create(env_name, packages = c("tensorflow", "keras"))
  }
  reticulate::use_virtualenv(env_name, required = TRUE)
  
  # Ensure python works
  tryCatch({
    system2(reticulate::py_exe(), args = c("-c", "import tensorflow; print('ok')"), stdout = FALSE, stderr = FALSE)
  }, error = function(e) {})
}

