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
  readRDS(path)
}
