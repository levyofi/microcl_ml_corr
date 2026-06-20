# ---- microclCorr: LSTM Training ----
# Port of LSTM logic from train.py using keras3

#' Build a stacked LSTM model
#'
#' Creates a Sequential keras model with stacked LSTM layers,
#' dropout, and a single dense output for residual prediction.
#'
#' @param input_shape Numeric vector c(window_size, n_features)
#' @param n_units Number of LSTM units per layer
#' @param n_layers Number of stacked LSTM layers
#' @param dropout Dropout rate (0 to 1)
#' @param lr Learning rate (NULL to use Adam default)
#' @return A compiled keras model
#' @export
build_lstm <- function(input_shape,
                       n_units = 64,
                       n_layers = 2,
                       dropout = 0.1,
                       lr = 0.001) {

  model <- keras3::keras_model_sequential()

  for (i in seq_len(n_layers)) {
    return_seq <- (i < n_layers)  # return sequences for all but last LSTM
    if (i == 1) {
      model |> keras3::layer_lstm(
        units = n_units,
        return_sequences = return_seq,
        input_shape = input_shape
      )
    } else {
      model |> keras3::layer_lstm(
        units = n_units,
        return_sequences = return_seq
      )
    }
  }

  model |>
    keras3::layer_dropout(rate = dropout) |>
    keras3::layer_dense(units = 1, activation = "linear")

  if (!is.null(lr)) {
    optimizer <- keras3::optimizer_adam(learning_rate = lr)
  } else {
    optimizer <- "adam"
  }

  model |> keras3::compile(
    loss = "mean_squared_error",
    optimizer = optimizer,
    metrics = list("mse")
  )

  model
}

#' Train an LSTM model for residual correction
#'
#' Builds and trains a stacked LSTM on windowed time-series data.
#' Uses early stopping on validation loss.
#'
#' @param train_X 3D array (n_windows, window_size, n_features)
#' @param train_y Numeric vector of targets
#' @param val_X 3D array of validation features
#' @param val_y Numeric vector of validation targets
#' @param n_units LSTM units per layer
#' @param n_layers Number of LSTM layers
#' @param dropout Dropout rate
#' @param lr Learning rate
#' @param epochs Max training epochs
#' @param batch_size Training batch size
#' @param patience Early stopping patience
#' @param seed Random seed for reproducibility
#' @return A trained keras model
#' @export
train_lstm <- function(train_X, train_y,
                       val_X, val_y,
                       n_units = 64,
                       n_layers = 2,
                       dropout = 0.1,
                       lr = 0.001,
                       epochs = 100,
                       batch_size = 32,
                       patience = 10,
                       seed = 42) {

  # Set seeds for reproducibility
  tensorflow::tf$random$set_seed(as.integer(seed))
  set.seed(seed)

  input_shape <- c(dim(train_X)[2], dim(train_X)[3])

  model <- build_lstm(
    input_shape = input_shape,
    n_units = n_units,
    n_layers = n_layers,
    dropout = dropout,
    lr = lr
  )

  early_stop <- keras3::callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    restore_best_weights = TRUE
  )

  model |> keras3::fit(
    x = train_X,
    y = train_y,
    validation_data = list(val_X, val_y),
    epochs = as.integer(epochs),
    batch_size = as.integer(batch_size),
    callbacks = list(early_stop),
    verbose = 1
  )

  model
}

#' LSTM Hyperparameter Tuning
#'
#' Performs a simple random search over LSTM hyperparameters.
#'
#' @param train_X 3D training features array
#' @param train_y Training targets
#' @param val_X 3D validation features array
#' @param val_y Validation targets
#' @param n_trials Number of random HP combinations to try
#' @param units_range Min/max for n_units (searched in steps of 32)
#' @param layers_range Min/max for n_layers
#' @param dropout_range Min/max for dropout
#' @param lr_range Min/max for learning rate (log-uniform)
#' @param epochs Max epochs per trial
#' @param batch_size Batch size
#' @param patience Early stopping patience
#' @param seed Random seed
#' @return List with best_model and best_params
#' @keywords internal
lstm_hypertuning <- function(train_X, train_y, val_X, val_y,
                             n_trials = 5,
                             units_range = c(32, 512),
                             layers_range = c(1, 3),
                             dropout_range = c(0, 0.3),
                             lr_range = c(1e-4, 1e-2),
                             epochs = 100,
                             batch_size = 32,
                             patience = 10,
                             seed = 123) {

  set.seed(seed)

  best_val_loss <- Inf
  best_model <- NULL
  best_params <- NULL

  units_choices <- seq(units_range[1], units_range[2], by = 32)
  layers_choices <- seq(layers_range[1], layers_range[2])
  dropout_choices <- seq(dropout_range[1], dropout_range[2], by = 0.05)

  for (trial in seq_len(n_trials)) {
    params <- list(
      n_units  = sample(units_choices, 1),
      n_layers = sample(layers_choices, 1),
      dropout  = sample(dropout_choices, 1),
      lr       = exp(runif(1, log(lr_range[1]), log(lr_range[2])))
    )

    message(sprintf("Trial %d/%d: units=%d, layers=%d, dropout=%.2f, lr=%.6f",
                    trial, n_trials, params$n_units, params$n_layers,
                    params$dropout, params$lr))

    model <- train_lstm(
      train_X, train_y, val_X, val_y,
      n_units = params$n_units,
      n_layers = params$n_layers,
      dropout = params$dropout,
      lr = params$lr,
      epochs = epochs,
      batch_size = batch_size,
      patience = patience,
      seed = seed + trial
    )

    # Evaluate on validation set
    val_preds <- as.numeric(model |> keras3::predict_on_batch(val_X))
    val_mse <- mean((val_preds - val_y)^2)

    message(sprintf("  -> Val MSE = %.4f", val_mse))

    if (val_mse < best_val_loss) {
      best_val_loss <- val_mse
      best_model <- model
      best_params <- params
    }
  }

  message(sprintf("Best trial: units=%d, layers=%d, dropout=%.2f, lr=%.6f, val_mse=%.4f",
                  best_params$n_units, best_params$n_layers,
                  best_params$dropout, best_params$lr, best_val_loss))

  list(model = best_model, params = best_params, val_mse = best_val_loss)
}
