# ---- microclCorr: Evaluation ----
# Port of evaluate_residual_model from helpers.py

#' Evaluate correction quality
#'
#' Computes RMSE and R^2 for both the base (uncorrected) predictions
#' and the ML-corrected predictions.
#'
#' @param model Trained model (ranger or keras)
#' @param X Test features (data.frame for RF, 3D array for LSTM)
#' @param y Test targets (residuals)
#' @param base_prediction Base model predictions
#' @param model_type Character: "rf" or "lstm"
#' @return A list with rmse_base, rmse_corr, r2_base, r2_corr
#' @examples
#' data(microclimate_sample)
#' feat_cols <- c("TAREF", "RH", "VREF", "SOLR")
#' rf <- train_rf(microclimate_sample[1:300, feat_cols],
#'                microclimate_sample$residual[1:300],
#'                num_trees = 5, tune = FALSE)
#' metrics <- evaluate_correction(
#'   model           = rf,
#'   X               = microclimate_sample[301:504, feat_cols],
#'   y               = microclimate_sample$residual[301:504],
#'   base_prediction = microclimate_sample$predicted[301:504],
#'   model_type      = "rf"
#' )
#' metrics$rmse_base
#' metrics$rmse_corr
#' @export
evaluate_correction <- function(model, X, y, base_prediction,
                                model_type = c("rf", "lstm")) {
  model_type <- match.arg(model_type)

  # Predict residuals
  if (model_type == "rf") {
    if (!isNamespaceLoaded("ranger")) {
      loadNamespace("ranger")
    }
    pred_res <- stats::predict(model, data = as.data.frame(X))$predictions
  } else {
    check_keras3()
    pred_res <- as.numeric(model |> keras3::predict_on_batch(X))
  }

  # True values and corrected predictions
  measured  <- base_prediction + y
  corrected <- base_prediction + pred_res

  # RMSE
  rmse_base <- sqrt(mean((measured - base_prediction)^2))
  rmse_corr <- sqrt(mean((measured - corrected)^2))

  # R^2
  ss_tot <- sum((measured - mean(measured))^2)
  if (isTRUE(all.equal(ss_tot, 0)) || ss_tot == 0) {
    warning("Zero variance in measured values: R^2 is undefined, returning NA.", call. = FALSE)
    r2_base <- NA_real_
    r2_corr <- NA_real_
  } else {
    r2_base <- 1 - sum((measured - base_prediction)^2) / ss_tot
    r2_corr <- 1 - sum((measured - corrected)^2) / ss_tot
  }

  list(
    rmse_base = rmse_base,
    rmse_corr = rmse_corr,
    r2_base   = r2_base,
    r2_corr   = r2_corr
  )
}

#' Print evaluation metrics in a formatted way
#'
#' @param metrics List from evaluate_correction
#' @param model_name Optional model name for display
#' @keywords internal
print_metrics <- function(metrics, model_name = "") {
  if (nchar(model_name) > 0) {
    cat(sprintf("\n=== %s ===\n", model_name))
  }
  cat(sprintf("  RMSE base:      %.4f\n", metrics$rmse_base))
  cat(sprintf("  RMSE corrected: %.4f\n", metrics$rmse_corr))
  cat(sprintf("  R^2 base:        %.4f\n", metrics$r2_base))
  cat(sprintf("  R^2 corrected:   %.4f\n", metrics$r2_corr))
  cat(sprintf("  Improvement:    %.1f%%\n",
              (1 - metrics$rmse_corr / metrics$rmse_base) * 100))
}
