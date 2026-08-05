# inst/examples/utils.R
# Shared helper functions used by scenario scripts.
# Source this file at the top of any scenario that loads pre-defined splits.

# setup_tensorflow ---------------------------------------------------------------
# Finds a Python environment with TensorFlow installed (searches the reticulate
# uv cache) and sets RETICULATE_PYTHON before reticulate binds to Python.
# Call this BEFORE library(reticulate) and py_require("tensorflow").
#
# Usage (at top of every scenario script, replacing the manual env var):
#   source(system.file("examples", "utils.R", package = "microclCorr"))
#   setup_tensorflow()
#   library(reticulate); py_require("tensorflow")
setup_tensorflow <- function() {
  if (nchar(Sys.getenv("RETICULATE_PYTHON")) > 0) return(invisible(NULL))

  cache_root <- file.path(path.expand("~"), "Library", "Caches",
                           "org.R-project.R", "R", "reticulate", "uv",
                           "cache", "archive-v0")
  if (!dir.exists(cache_root)) return(invisible(NULL))

  # Find all python/python3 binaries in the reticulate uv cache
  all_files  <- list.files(cache_root, recursive = TRUE, full.names = TRUE)
  candidates <- all_files[grepl("/bin/python3?$", all_files)]

  for (py in candidates) {
    if (!file.access(py, 1) == 0) next   # not executable
    has_tf <- tryCatch({
      res <- suppressWarnings(
        system2(py, c("-c", "import tensorflow; print('ok')"),
                stdout = TRUE, stderr = FALSE))
      any(grepl("ok", res))
    }, error = function(e) FALSE)
    if (has_tf) {
      Sys.setenv(RETICULATE_PYTHON = py)
      message("setup_tensorflow: using ", py)
      return(invisible(py))
    }
  }
  invisible(NULL)  # reticulate will find TF via py_require on its own
}

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

# logger_temp_stats --------------------------------------------------------------
# Summarise measured temperatures for one logger's data subset as daily
# aggregates: average daily mean, average daily min, average daily max,
# each reported as mean ± SD across days.
# Returns a single-row data frame labelled by `label`.
logger_temp_stats <- function(data_subset, label) {
  meas <- data_subset$predicted + data_subset$residual
  date <- as.Date(data_subset$time)

  daily_mean <- tapply(meas, date, mean, na.rm = TRUE)
  daily_min  <- tapply(meas, date, min,  na.rm = TRUE)
  daily_max  <- tapply(meas, date, max,  na.rm = TRUE)

  data.frame(
    logger         = label,
    daily_mean_avg = round(mean(daily_mean), 2),
    daily_mean_sd  = round(sd(daily_mean),   2),
    daily_min_avg  = round(mean(daily_min),  2),
    daily_min_sd   = round(sd(daily_min),    2),
    daily_max_avg  = round(mean(daily_max),  2),
    daily_max_sd   = round(sd(daily_max),    2)
  )
}

# make_pred_plot -----------------------------------------------------------------
# Build a ggplot showing observed, NicheMapR, RF-corrected, and LSTM-corrected
# temperature lines for a single panel.
#
# Arguments:
#   df           — data frame with columns: time, measured, base, rf, lstm
#   title_str    — plot title
#   show_legend  — whether to draw the colour legend (set FALSE for panels 2+
#                  in a multi-panel grid to avoid repetition)
#   linewidth_obs — line width for the Observed series
# make_residual_hist -------------------------------------------------------------
# Overlay histograms of hourly residuals (measured − predicted) for NicheMapR
# (before correction), RF, and LSTM (after correction).
#
# Arguments:
#   full_df     — data frame with columns: measured, base, rf, lstm
#                 (lstm column is optional; omit or set has_lstm = FALSE)
#   title_str   — plot title
#   has_lstm    — whether to include the LSTM series (default TRUE)
#
# Residuals are defined as measured − model, so positive = model under-predicts.
make_residual_hist <- function(full_df, title_str, has_lstm = TRUE,
                               xlim = NULL, show_strip = TRUE, show_legend = TRUE) {
  required <- c("measured", "base", "rf")
  missing  <- setdiff(required, names(full_df))
  if (length(missing) > 0)
    stop("make_residual_hist: full_df missing columns: ",
         paste(missing, collapse = ", "))
  if (has_lstm && !"lstm" %in% names(full_df))
    stop("make_residual_hist: has_lstm=TRUE but 'lstm' column not found in full_df")

  check_nas <- c("measured", "base", "rf")
  if (has_lstm) check_nas <- c(check_nas, "lstm")
  for (col in check_nas) {
    n_na <- sum(is.na(full_df[[col]]))
    if (n_na > 0)
      warning(sprintf("make_residual_hist: %d NA(s) in column '%s' — dropping",
                      n_na, col))
  }
  full_df <- full_df[complete.cases(full_df[, check_nas]), ]

  res_base <- full_df$measured - full_df$base
  res_rf   <- full_df$measured - full_df$rf

  rows <- list(
    data.frame(residual = res_base, model = "NicheMapR (before)"),
    data.frame(residual = res_rf,   model = "RF (after)")
  )

  if (has_lstm && "lstm" %in% names(full_df)) {
    rows[[3]] <- data.frame(residual = full_df$measured - full_df$lstm,
                            model    = "LSTM (after)")
  }

  df_long <- do.call(rbind, rows)
  df_long$model <- factor(df_long$model,
    levels = c("NicheMapR (before)", "RF (after)", "LSTM (after)"))

  cols <- c("NicheMapR (before)" = "#ef4444",
            "RF (after)"         = "#10b981",
            "LSTM (after)"       = "#3b82f6")

  if (is.null(xlim)) xlim <- range(df_long$residual, na.rm = TRUE)

  p <- ggplot2::ggplot(df_long, ggplot2::aes(x = residual, fill = model)) +
    ggplot2::geom_histogram(bins = 50, position = "identity", colour = "white", alpha = 0.5) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed",
                        linewidth = 0.7, colour = "#333333") +
    ggplot2::coord_cartesian(xlim = xlim) +
    ggplot2::scale_fill_manual(values = cols) +
    ggplot2::labs(
      title = title_str,
      x     = expression("Prediction errors (" * degree * "C)"),
      y     = "Count",
      fill  = NULL) +
    ggplot2::theme_minimal(base_size = 18) +
    ggplot2::theme(
      aspect.ratio       = 1,
      plot.title         = ggplot2::element_text(face = "bold", hjust = 0.5, size = 20),
      axis.title         = ggplot2::element_text(size = 18),
      axis.text          = ggplot2::element_text(size = 16),
      legend.position    = "bottom",
      legend.text        = ggplot2::element_text(size = 16),
      panel.grid.major   = ggplot2::element_blank(),
      panel.grid.minor   = ggplot2::element_blank(),
      panel.border       = ggplot2::element_rect(colour = "black", fill = NA,
                                                  linewidth = 0.8))
  
  if (!show_legend) {
    p <- p + ggplot2::theme(legend.position = "none")
  }
  
  return(p)
}

make_pred_plot <- function(df, title_str, show_legend = TRUE, linewidth_obs = 0.9,
                           has_lstm = TRUE) {
  p <- ggplot2::ggplot(df, ggplot2::aes(x = time)) +
    ggplot2::geom_line(ggplot2::aes(y = measured, color = "Observed"),
                       linewidth = linewidth_obs) +
    ggplot2::geom_line(ggplot2::aes(y = base,     color = "NicheMapR"),
                       linetype = "dashed", linewidth = 0.8)
  if (has_lstm && "lstm" %in% names(df))
    p <- p + ggplot2::geom_line(ggplot2::aes(y = lstm, color = "LSTM Corrected"),
                                 linewidth = 0.8)
  p <- p +
    ggplot2::geom_line(ggplot2::aes(y = rf,       color = "RF Corrected"),
                       linetype = "dotted", linewidth = 0.8) +
    ggplot2::scale_color_manual(
      values = c("Observed"       = "#111111",
                 "NicheMapR"      = "#ef4444",
                 "LSTM Corrected" = "#3b82f6",
                 "RF Corrected"   = "#10b981")) +
    ggplot2::labs(title = title_str, x = NULL, y = "Temperature (°C)", color = NULL) +
    ggplot2::theme_minimal(base_size = 16) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(face = "bold", hjust = 0.5, size = 18),
      legend.position  = "top",
      legend.text      = ggplot2::element_text(size = 16),
      axis.title       = ggplot2::element_text(size = 16),
      axis.text        = ggplot2::element_text(size = 14),
      panel.grid.minor = ggplot2::element_blank())
}

# build_pred_df ------------------------------------------------------------------
# Assemble the four-column prediction data frame from model objects and test data.
# Returns a data frame sorted by time with columns: time, measured, base, rf, lstm.
#
# Arguments:
#   rf_test        — aligned RF test set (data frame)
#   feature_cols   — character vector of predictor column names
#   rf_model       — trained ranger model
#   base_test_lstm — NicheMapR predictions aligned to LSTM test windows
#   lstm_model     — trained Keras LSTM model
#   X_test_lstm    — 3-D array of LSTM test windows
build_pred_df <- function(rf_test, feature_cols, rf_model,
                           base_test_lstm, lstm_model, X_test_lstm) {
  rf_preds   <- rf_test$predicted +
                ranger:::predict.ranger(rf_model, data = as.data.frame(rf_test[, feature_cols]))$predictions
  lstm_preds <- base_test_lstm +
                as.numeric(predict(lstm_model, X_test_lstm, verbose = 0)[, 1])
  # Preserve the original row order — predictions are positionally aligned to
  # rf_test rows. Callers should sort per site after filtering; sorting here
  # across multiple sites would break the positional alignment.
  data.frame(
    time     = rf_test$time,
    measured = rf_test$predicted + rf_test$residual,
    base     = rf_test$predicted,
    rf       = rf_preds,
    lstm     = lstm_preds
  )
}

# compute_daily_stats ------------------------------------------------------------
# From a prediction data frame (columns: time, measured, base, rf, and
# optionally lstm), compute summary statistics of daily errors.
#
# For each model (base, rf, lstm) and each daily aggregate (min, mean, max):
#   - Compute the per-day absolute error and signed error
#   - Return avg ± SD of RMSE across days and avg ± SD of ME across days
#
# Returns a data frame with one row per model, columns:
#   model,
#   rmse_mean_avg, rmse_mean_sd,   (RMSE of daily means across days)
#   rmse_min_avg,  rmse_min_sd,    (RMSE of daily mins  across days)
#   rmse_max_avg,  rmse_max_sd,    (RMSE of daily maxes across days)
#   me_mean_avg,   me_mean_sd,     (ME   of daily means across days)
#   me_min_avg,    me_min_sd,      (ME   of daily mins  across days)
#   me_max_avg,    me_max_sd       (ME   of daily maxes across days)
compute_daily_stats <- function(full_df) {
  full_df$date <- as.Date(full_df$time)
  daily <- do.call(rbind, lapply(split(full_df, full_df$date), function(d) {
    row <- data.frame(
      date      = d$date[1],
      meas_min  = min(d$measured),  meas_mean = mean(d$measured), meas_max = max(d$measured),
      base_min  = min(d$base),      base_mean = mean(d$base),     base_max = max(d$base),
      rf_min    = min(d$rf),        rf_mean   = mean(d$rf),       rf_max   = max(d$rf)
    )
    if ("lstm" %in% names(d)) {
      row$lstm_min  <- min(d$lstm)
      row$lstm_mean <- mean(d$lstm)
      row$lstm_max  <- max(d$lstm)
    }
    row
  }))

  has_lstm <- all(c("lstm_min", "lstm_mean", "lstm_max") %in% names(daily))
  models   <- if (has_lstm) c("base", "rf", "lstm") else c("base", "rf")

  rows <- lapply(models, function(m) {
    err_mean <- daily[[paste0(m, "_mean")]] - daily$meas_mean
    err_min  <- daily[[paste0(m, "_min")]]  - daily$meas_min
    err_max  <- daily[[paste0(m, "_max")]]  - daily$meas_max
    data.frame(
      model         = m,
      rmse_mean_avg = round(sqrt(mean(err_mean^2)), 2),
      rmse_mean_sd  = round(sd(abs(err_mean)),       2),
      rmse_min_avg  = round(sqrt(mean(err_min^2)),   2),
      rmse_min_sd   = round(sd(abs(err_min)),         2),
      rmse_max_avg  = round(sqrt(mean(err_max^2)),   2),
      rmse_max_sd   = round(sd(abs(err_max)),         2),
      me_mean_avg   = round(mean(err_mean),           2),
      me_mean_sd    = round(sd(err_mean),             2),
      me_min_avg    = round(mean(err_min),            2),
      me_min_sd     = round(sd(err_min),              2),
      me_max_avg    = round(mean(err_max),            2),
      me_max_sd     = round(sd(err_max),              2)
    )
  })
  do.call(rbind, rows)
}

# print_daily_stats --------------------------------------------------------------
# Pretty-print the output of compute_daily_stats() for one microhabitat/logger.
# Prints two tables: RMSE and ME, each avg ± SD across days.
print_daily_stats <- function(ds, label) {
  cat(sprintf("\n  %s\n", label))
  cat(sprintf("  %-8s | %20s | %20s | %20s\n",
              "Model", "Daily Mean RMSE", "Daily Min RMSE", "Daily Max RMSE"))
  for (i in seq_len(nrow(ds))) {
    r <- ds[i, ]
    cat(sprintf("  %-8s | %8.2f ± %-8.2f | %8.2f ± %-8.2f | %8.2f ± %-8.2f\n",
                r$model,
                r$rmse_mean_avg, r$rmse_mean_sd,
                r$rmse_min_avg,  r$rmse_min_sd,
                r$rmse_max_avg,  r$rmse_max_sd))
  }
  cat(sprintf("  %-8s | %20s | %20s | %20s\n",
              "Model", "Daily Mean ME", "Daily Min ME", "Daily Max ME"))
  for (i in seq_len(nrow(ds))) {
    r <- ds[i, ]
    cat(sprintf("  %-8s | %8.2f ± %-8.2f | %8.2f ± %-8.2f | %8.2f ± %-8.2f\n",
                r$model,
                r$me_mean_avg, r$me_mean_sd,
                r$me_min_avg,  r$me_min_sd,
                r$me_max_avg,  r$me_max_sd))
  }
}
