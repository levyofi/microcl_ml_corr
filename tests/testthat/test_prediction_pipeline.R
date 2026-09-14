test_that("correct_predictions and model save/load works for Random Forest", {
  set.seed(42)
  # create a random dataset for test
  train_X <- data.frame(temp_env = rnorm(20, 15, 2))
  train_y <- rnorm(20, 0, 1)
  
  # train a RF model on the random dataset
  rf_model <- train_rf(train_X, train_y, num_trees = 10, tune = FALSE, seed = 42)
  
  # create a random dataset for test
  df <- data.frame(
    time = as.POSIXct(c("2024-01-01 00:00:00", "2024-01-01 01:00:00"), tz = "UTC"),
    predicted = c(20, 25),
    temp_env = c(15, 18)
  )
  
  # test correct_predictions function
  res <- correct_predictions(rf_model, df, model_type = "rf", feature_cols = "temp_env")
  expect_equal(nrow(res), 2)
  expect_named(res, c("datetime", "base_prediction", "correction", "corrected_prediction"))
  expect_equal(res$corrected_prediction, res$base_prediction + res$correction)
  
  tmp_path <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp_path), add = TRUE)
  
  # save the trained model
  save_correction_model(rf_model, scaler = NULL, feature_cols = "temp_env", path = tmp_path)
  # load the model again
  loaded <- load_correction_model(tmp_path)
  
  # Check that the loaded model is a ranger model and has the same feature columns
  expect_equal(loaded$model_type, "rf")
  expect_equal(loaded$feature_cols, "temp_env")
})
