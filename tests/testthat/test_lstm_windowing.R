test_that("make_windows creates valid 3D tensors for LSTM", {
  n <- 10
  X_mat <- matrix(1:20, nrow = n, ncol = 2)
  y_vec <- 1:n
  base_pred_vec <- 20 + 1:n
  datetime_vec <- seq(as.POSIXct("2024-01-01 00:00:00", tz = "UTC"), by = "hour", length.out = n)
  
  win <- make_windows(X_mat, y_vec, base_pred_vec, datetime_vec, window_size = 3, max_gap_hours = 1)
  
  expect_type(win, "list")
  expect_equal(dim(win$X), c(8, 3, 2))
  expect_equal(length(win$y), 8)
  expect_equal(length(win$base_pred), 8)
})

test_that("make_windows validates window_size argument", {
  X_mat <- matrix(1:10, nrow = 5, ncol = 2)
  dt <- seq(as.POSIXct("2024-01-01", tz = "UTC"), by = "hour", length.out = 5)
  expect_error(make_windows(X_mat, 1:5, 1:5, dt, window_size = 0), "positive integer")
  expect_error(make_windows(X_mat, 1:5, 1:5, dt, window_size = -2), "positive integer")
})

