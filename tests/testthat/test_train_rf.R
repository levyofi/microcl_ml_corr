test_that("train_rf fits a ranger model on dummy microclimate data", {
  set.seed(42)
  # create random data for test
  train_X <- data.frame(
    temp = rnorm(30, mean = 20, sd = 5),
    solar = rnorm(30, mean = 500, sd = 100)
  )
  train_y <- rnorm(30, mean = 0, sd = 1)
  
  # train a rf model on the random dataset
  model <- train_rf(train_X, train_y, num_trees = 10, tune = FALSE, seed = 42)
  
  # Check that the trained model is a ranger model and has 10 trees
  expect_s3_class(model, "ranger")
  expect_equal(model$num.trees, 10)
  
  # Check that the model can predict on the training data
  preds <- stats::predict(model, data = train_X)$predictions
  expect_equal(length(preds), 30)
})
