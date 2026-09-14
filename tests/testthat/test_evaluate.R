test_that("evaluate_correction calculates exact RMSE = 0 and R2 = 1 when predictions are perfect", {
  # When target residual y is constant, Random Forest fits it perfectly (pred_res == y)
  X <- data.frame(temp = c(10, 15, 20, 25, 30))
  y <- c(2, 2, 2, 2, 2)
  base_pred <- c(20, 22, 24, 26, 28)
  
  rf_model <- train_rf(X, y, num_trees = 5, tune = FALSE, seed = 42)
  
  res <- evaluate_correction(rf_model, X, y, base_pred, model_type = "rf")
  
  expect_type(res, "list")
  expect_named(res, c("rmse_base", "rmse_corr", "r2_base", "r2_corr"))
  expect_equal(res$rmse_corr, 0, tolerance = 1e-6)
  expect_equal(res$r2_corr, 1, tolerance = 1e-6)
  expect_equal(res$rmse_base, 2, tolerance = 1e-6)
})
