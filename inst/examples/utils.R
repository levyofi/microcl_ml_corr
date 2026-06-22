# inst/examples/utils.R
# Shared helper functions used by scenario scripts.
# Source this file at the top of any scenario that loads pre-defined splits.

# load_splits_from_csv -----------------------------------------------------------
# Scenarios 4-8 use a pre-defined CSV file that assigns every row in the dataset
# to one of three roles: "train" (used to fit the model), "val" (used to tune
# settings during training), or "test" (held out to measure final accuracy).
# This function attaches those assignments to the data by matching on timestamp
# and logger ID, then splits the data into three separate tables.
load_splits_from_csv <- function(data, splits_csv, site_col, datetime_col = "time") {
  sp  <- read.csv(splits_csv, stringsAsFactors = FALSE)

  # Standardise timestamp format so rows can be matched across the two files
  fmt <- function(x) format(as.POSIXct(x, tz = "UTC"), "%Y-%m-%d %H:%M:%S", tz = "UTC")
  data$.t <- fmt(data[[datetime_col]])
  sp$.t   <- fmt(sp[[datetime_col]])

  # Join split labels onto the data rows
  m <- merge(data, sp[, c(".t", site_col, "split")], by = c(".t", site_col), all.x = TRUE)
  m <- m[order(m[[datetime_col]]), ]
  m$.t <- NULL
  cols <- setdiff(names(m), "split")   # keep all columns except the temporary "split" label

  list(
    train = m[!is.na(m$split) & m$split == "train", cols],
    val   = m[!is.na(m$split) & m$split == "val",   cols],
    test  = m[!is.na(m$split) & m$split == "test",  cols]
  )
}

# find_min_training_days ---------------------------------------------------------
# Answers the question: "How many days of logger data do I actually need?"
#
# The function trains RF and/or LSTM at progressively smaller training sizes
# (e.g. 1, 2, 3, 7, 14, 21, 28, 35, 42 days) and finds the minimum number of
# days where accuracy stays within `tolerance` of the full-data result.
#
# For example, tolerance = 0.10 means "find the fewest days where RMSE is at
# most 10% worse than training on all available data".
#
# Arguments:
#   splits        — list(train, val, test) from split_train_val_test()
#   lstm_2h       — output of lstm_specific_preprocessing() with window_size=2
#   feature_cols  — character vector from get_feature_columns()
#   rf_model      — trained RF model (from train_rf()) used as full-data reference
#   lstm_model    — trained LSTM model (from train_lstm()) as full-data reference
#   lstm_params   — list(n_units, n_layers, dropout, lr) from lstm_hypertuning()
#   rf_test       — aligned RF test set from align_test_sets()
#   X_test_lstm, y_test_lstm, base_test_lstm — aligned LSTM test arrays
#   site_col      — column name identifying the logger
#   tolerance     — acceptable fraction above full-data RMSE (default 0.10 = 10%)
#   training_days — vector of training sizes to test (in days)
#   n_runs        — number of random repetitions per size (for variance estimate)
#   seed          — base random seed
#
# Returns a list with:
#   $results  — data frame with RMSE at every training size and run
#   $summary  — mean RMSE ± SD per model per training size
#   $min_days — named vector: minimum days needed per model at the given tolerance
#   $plot     — ggplot learning curve (print it or ggsave it)
find_min_training_days <- function(splits, lstm_2h, feature_cols,
                                   rf_model, lstm_model, lstm_params,
                                   rf_test, X_test_lstm, y_test_lstm, base_test_lstm,
                                   site_col,
                                   tolerance     = 0.10,
                                   training_days = c(1, 2, 3, 7, 14, 21, 28, 35, 42),
                                   n_runs        = 5,
                                   seed          = 123) {
  library(ggplot2)

  # Reference RMSE: performance with all available training data
  ref_rf   <- evaluate_correction(rf_model, rf_test[, feature_cols],
                                   rf_test$residual, rf_test$predicted,
                                   model_type = "rf")$rmse_corr
  ref_lstm <- evaluate_correction(lstm_model, X_test_lstm, y_test_lstm,
                                   base_test_lstm, model_type = "lstm")$rmse_corr

  # Training rows sorted by time (first N days = first N*24 rows)
  train_sorted <- splits$train[order(splits$train$time), ]

  results <- list()

  for (n_days in training_days) {
    n_hours    <- n_days * 24
    rf_partial <- train_sorted[seq_len(min(nrow(train_sorted), n_hours)), ]
    lstm_n     <- min(length(lstm_2h$train_dict$y), n_hours)
    X_partial  <- lstm_2h$train_dict$X[seq_len(lstm_n), , , drop = FALSE]
    y_partial  <- lstm_2h$train_dict$y[seq_len(lstm_n)]

    for (run in seq_len(n_runs) - 1L) {
      # RF: train with different seed each run
      rf_m <- train_rf(rf_partial[, feature_cols], rf_partial$residual,
                        tune = FALSE, seed = run)
      m_rf <- evaluate_correction(rf_m, rf_test[, feature_cols],
                                   rf_test$residual, rf_test$predicted,
                                   model_type = "rf")
      results[[length(results) + 1]] <- data.frame(
        model = "RF", n_days = n_days, run = run,
        rmse_corr = m_rf$rmse_corr, rmse_base = m_rf$rmse_base)

      # LSTM: train with tuned architecture, different seed each run
      lstm_m <- train_lstm(X_partial, y_partial,
                            lstm_2h$val_dict$X, lstm_2h$val_dict$y,
                            n_units    = lstm_params$n_units,
                            n_layers   = lstm_params$n_layers,
                            dropout    = lstm_params$dropout,
                            lr         = lstm_params$lr,
                            epochs = 40, batch_size = 32, patience = 10,
                            seed   = run)
      m_lstm <- evaluate_correction(lstm_m, X_test_lstm, y_test_lstm,
                                     base_test_lstm, model_type = "lstm")
      results[[length(results) + 1]] <- data.frame(
        model = "LSTM_2h", n_days = n_days, run = run,
        rmse_corr = m_lstm$rmse_corr, rmse_base = m_lstm$rmse_base)
    }
  }

  results_df <- do.call(rbind, results)

  # Summary: mean ± SD per model per training size
  summ    <- aggregate(cbind(rmse_corr, rmse_base) ~ model + n_days, results_df, mean)
  summ_sd <- aggregate(rmse_corr ~ model + n_days, results_df, sd)
  summ$sd_corr <- summ_sd$rmse_corr

  # Minimum days within tolerance of full-data RMSE
  thresholds <- c(RF = ref_rf * (1 + tolerance), LSTM_2h = ref_lstm * (1 + tolerance))
  min_days <- sapply(c("RF", "LSTM_2h"), function(m) {
    sub <- summ[summ$model == m, ]
    passing <- sub$n_days[sub$rmse_corr <= thresholds[m]]
    if (length(passing) == 0) max(training_days) else min(passing)
  })
  names(min_days) <- c("RF", "LSTM_2h")

  # Learning curve plot
  ref_lines <- data.frame(
    model     = c("RF", "LSTM_2h"),
    threshold = c(thresholds["RF"], thresholds["LSTM_2h"])
  )
  p <- ggplot(summ, aes(x = n_days, y = rmse_corr, color = model, fill = model)) +
    geom_ribbon(aes(ymin = rmse_corr - sd_corr, ymax = rmse_corr + sd_corr),
                alpha = 0.15, color = NA) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 1.5) +
    geom_hline(aes(yintercept = threshold, color = model),
               data = ref_lines, linetype = "dashed", linewidth = 0.7, show.legend = FALSE) +
    scale_color_manual(values = c("RF" = "#10b981", "LSTM_2h" = "#3b82f6"),
                       labels = c("RF" = "Random Forest", "LSTM_2h" = "LSTM (2h)")) +
    scale_fill_manual(values  = c("RF" = "#10b981", "LSTM_2h" = "#3b82f6"),
                      labels  = c("RF" = "Random Forest", "LSTM_2h" = "LSTM (2h)")) +
    scale_x_continuous(breaks = training_days) +
    labs(title    = sprintf("Minimum Training Data Search  (tolerance = %.0f%%)", tolerance * 100),
         subtitle = sprintf("RF: %d days | LSTM: %d days to reach within %.0f%% of full-data RMSE",
                            min_days["RF"], min_days["LSTM_2h"], tolerance * 100),
         x     = "Training Data Size (Days)",
         y     = "Test RMSE (°C)  — lower is better",
         color = "Model", fill = "Model") +
    theme_minimal(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, color = "#555555"),
          legend.position = "bottom", panel.grid.minor = element_blank())

  list(results = results_df, summary = summ, min_days = min_days, plot = p)
}

# results_row --------------------------------------------------------------------
# Convenience function that turns the output of evaluate_correction() into a
# single-row data frame, also computing the percentage improvement over the
# uncorrected NicheMapR baseline.
results_row <- function(model_name, site, metrics) {
  data.frame(
    model           = model_name,
    site            = site,
    rmse_base       = metrics$rmse_base,   # NicheMapR error before correction
    rmse_corr       = metrics$rmse_corr,   # model error after correction
    improvement_pct = (metrics$rmse_base - metrics$rmse_corr) / metrics$rmse_base * 100,
    stringsAsFactors = FALSE
  )
}
