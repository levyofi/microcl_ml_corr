# ---- microclCorr: Random Forest Training ----
# Port of RF logic from train.py using ranger

#' Train a Random Forest model for residual correction
#'
#' Trains a ranger Random Forest to predict residuals (measured - predicted).
#' Optionally performs hyperparameter tuning via out-of-bag error.
#'
#' @param train_X Data.frame or matrix of training features
#' @param train_y Numeric vector of training targets (residuals)
#' @param num_trees Number of trees (default 500)
#' @param tune Logical. If TRUE, try multiple hyperparameter combinations.
#' @param n_combinations Number of random HP combinations to try when tune=TRUE
#' @param max_depth_options Integer vector of max.depth values to search
#' @param min_node_size_options Integer vector of min.node.size values
#' @param mtry_options Values for mtry (NULL means use defaults)
#' @param val_X Optional validation features (for dedicated val set tuning)
#' @param val_y Optional validation targets
#' @param seed Random seed
#' @return A fitted ranger model
#' @export
train_rf <- function(train_X, train_y,
                     num_trees = 500,
                     tune = TRUE,
                     n_combinations = 5,
                     max_depth_options = c(10, 20, 30, 0),
                     min_node_size_options = c(2, 5, 10),
                     mtry_options = NULL,
                     val_X = NULL, val_y = NULL,
                     seed = 123) {

  train_X <- as.data.frame(train_X)

  if (is.null(mtry_options)) {
    p <- ncol(train_X)
    mtry_options <- unique(c(
      floor(sqrt(p)),
      floor(p / 3),
      p
    ))
  }

  if (!tune) {
    # Train with defaults
    set.seed(seed)
    model <- ranger::ranger(
      x = train_X, y = train_y,
      num.trees = num_trees,
      seed = seed
    )
    return(model)
  }

  # Hyperparameter tuning via grid search
  # If val set provided, use it; otherwise use OOB error
  grid <- expand.grid(
    max_depth = max_depth_options,
    min_node_size = min_node_size_options,
    mtry = mtry_options,
    stringsAsFactors = FALSE
  )

  # Sample n_combinations from the grid
  if (nrow(grid) > n_combinations) {
    set.seed(seed)
    grid <- grid[sample(nrow(grid), n_combinations), , drop = FALSE]
  }

  best_mse <- Inf
  best_model <- NULL

  for (i in seq_len(nrow(grid))) {
    md <- grid$max_depth[i]
    if (md == 0) md <- NULL  # 0 means unlimited

    set.seed(seed)
    model <- ranger::ranger(
      x = train_X, y = train_y,
      num.trees = 200,  # fewer trees for search (like Python N_ESTIMATORS_SEARCH)
      max.depth = md,
      min.node.size = grid$min_node_size[i],
      mtry = grid$mtry[i],
      seed = seed
    )

    if (!is.null(val_X) && !is.null(val_y)) {
      preds <- stats::predict(model, data = as.data.frame(val_X))$predictions
      mse <- mean((preds - val_y)^2)
    } else {
      # Use OOB prediction error
      mse <- model$prediction.error
    }

    if (mse < best_mse) {
      best_mse <- mse
      best_model <- model
      best_params <- grid[i, , drop = FALSE]
    }
  }

  message(sprintf("RF HPO: Best MSE = %.4f | max_depth=%s, min_node_size=%d, mtry=%d",
                  best_mse,
                  ifelse(is.null(best_params$max_depth) || best_params$max_depth == 0,
                         "NULL", as.character(best_params$max_depth)),
                  best_params$min_node_size, best_params$mtry))

  # Retrain with best params and full num_trees
  md_final <- best_params$max_depth
  if (!is.null(md_final) && md_final == 0) md_final <- NULL

  set.seed(seed)
  final_model <- ranger::ranger(
    x = train_X, y = train_y,
    num.trees = num_trees,
    max.depth = md_final,
    min.node.size = best_params$min_node_size,
    mtry = best_params$mtry,
    seed = seed
  )

  final_model
}
