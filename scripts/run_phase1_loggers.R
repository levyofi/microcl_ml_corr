#!/usr/bin/env Rscript
# ===========================================================
# Phase 1 Runner: Single-Logger ML Correction Pipeline (R)
#
# This script runs the full Phase 1 pipeline for individual
# temperature loggers, matching the Python pipeline_v2/main.py:
#   1. Load pre-aligned CSV data
#   2. Split into train / validation / test
#   3. Train RF and LSTM models
#   4. Evaluate and save results
#   5. Compare with Python pipeline_v2 results
# ===========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(ranger)
})

# --- Load the microclCorr package ---
args <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("^--file=", args, value = TRUE)
if (length(file_flag) > 0) {
  pkg_root <- normalizePath(dirname(dirname(sub("^--file=", "", file_flag))))
} else {
  pkg_root <- normalizePath("..")
}
suppressMessages(devtools::load_all(pkg_root, quiet = TRUE))

# ===========================================================
# CONFIGURATION
# ===========================================================
PROJECT_ROOT <- normalizePath(file.path(pkg_root, ".."))

# Data paths
HAROD_PATH   <- file.path(PROJECT_ROOT, "data/experiments_data/Harod_dataset.csv")
BEACH_PATH   <- file.path(PROJECT_ROOT, "data/experiments_data/Beach_data_preprocessed.csv")

# Python results for comparison
PYTHON_RESULTS_DIR <- file.path(PROJECT_ROOT, "pipeline_v2/outputs/phase_1/replicates")

# Output directory
OUTPUT_DIR <- file.path(PROJECT_ROOT, "microcl_ml_corr/outputs/phase_1")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Training config (matching Python pipeline defaults)
TRAIN_PCT   <- 0.75
VAL_PCT     <- 0.125
BLOCK_DAYS  <- 7
SEED        <- 123
N_RUNS      <- 5
WINDOW_SIZES <- c(1, 2, 24)

# Training sizes in days (matching Python pipeline_v2)
TRAINING_DAYS <- c(1, 2, 3, 7, 14, 21, 28, 35, 42)

# LSTM training config
LSTM_EPOCHS  <- 100
LSTM_PATIENCE <- 10
LSTM_BATCH    <- 32

# ===========================================================
# MAIN PIPELINE FUNCTION (one logger)
# ===========================================================
run_single_logger <- function(data, site_name, site_name_col,
                              window_sizes = WINDOW_SIZES,
                              n_runs = N_RUNS,
                              output_dir = OUTPUT_DIR,
                              use_lstm = TRUE) {

  cat(sprintf("\n========================================\n"))
  cat(sprintf("Processing logger: %s\n", site_name))
  cat(sprintf("Dataset size: %d rows\n", nrow(data)))
  cat(sprintf("========================================\n"))

  # 1. Split data
  splits <- split_train_val_test(
    data, train_pct = TRAIN_PCT, val_pct = VAL_PCT,
    block_days = BLOCK_DAYS, use_blocks = TRUE, seed = SEED
  )
  cat(sprintf("Split: train=%d, val=%d, test=%d\n",
              nrow(splits$train), nrow(splits$val), nrow(splits$test)))

  # 2. Scale features (for LSTM)
  scaled <- lstm_scaling(splits$train, splits$val, splits$test)
  feature_cols <- get_feature_columns(splits$train)
  cat(sprintf("Features: %s\n", paste(feature_cols, collapse = ", ")))

  # 3. Results collector
  results <- list()

  # --- 4. RF: Hyperparameter tuning + training at multiple sizes ---
  cat("\n--- Random Forest ---\n")

  rf_train_X <- splits$train[, feature_cols, drop = FALSE]
  rf_train_y <- splits$train$residual
  rf_val_X   <- splits$val[, feature_cols, drop = FALSE]
  rf_val_y   <- splits$val$residual

  # HPO on full data
  rf_full <- train_rf(
    rf_train_X, rf_train_y,
    tune = TRUE, n_combinations = 5,
    val_X = rf_val_X, val_y = rf_val_y,
    seed = SEED
  )

  # Get RF best params for partial training
  rf_params <- list(
    max.depth = rf_full$max.depth,
    min.node.size = rf_full$min.node.size,
    mtry = rf_full$mtry
  )

  # Evaluate at different training sizes
  for (n_days in TRAINING_DAYS) {
    n_hours <- n_days * 24
    # Get partial data for each time series in training set
    ts_sites <- unique(splits$train[[site_name_col]])
    partial_rows <- list()
    for (ts in ts_sites) {
      ts_data <- splits$train[splits$train[[site_name_col]] == ts, , drop = FALSE]
      ts_data <- ts_data[order(ts_data$time), , drop = FALSE]
      k <- min(nrow(ts_data), n_hours)
      partial_rows[[length(partial_rows) + 1]] <- ts_data[seq_len(k), , drop = FALSE]
    }
    rf_partial_train <- do.call(rbind, partial_rows)
    train_size <- nrow(rf_partial_train)

    if (train_size == 0) next

    rf_partial_X <- rf_partial_train[, feature_cols, drop = FALSE]
    rf_partial_y <- rf_partial_train$residual

    # RF test aligned with LSTM test (use full test set for RF when no LSTM alignment)
    rf_test_X <- splits$test[, feature_cols, drop = FALSE]
    rf_test_y <- splits$test$residual
    rf_test_base <- splits$test$predicted

    for (run_id in seq_len(n_runs)) {
      set.seed(SEED + run_id - 1)
      rf_model <- ranger::ranger(
        x = rf_partial_X, y = rf_partial_y,
        num.trees = 500,
        max.depth = rf_params$max.depth,
        min.node.size = rf_params$min.node.size,
        mtry = rf_params$mtry,
        seed = run_id - 1
      )

      metrics <- evaluate_correction(rf_model, rf_test_X, rf_test_y, rf_test_base,
                                     model_type = "rf")

      results[[length(results) + 1]] <- data.frame(
        model = "RF",
        window_size = NA,
        perc = n_days,
        train_size = train_size,
        ts_name = "ALL",
        rmse_corr = metrics$rmse_corr,
        rmse_base = metrics$rmse_base,
        run = run_id - 1,
        stringsAsFactors = FALSE
      )
    }

    cat(sprintf("  RF: %d days (%d pts), mean RMSE_corr=%.3f (base=%.3f)\n",
                n_days, train_size,
                mean(sapply(tail(results, n_runs), function(x) x$rmse_corr)),
                metrics$rmse_base))
  }

  # --- 5. LSTM: HPO + training at multiple sizes ---
  if (use_lstm) {
    cat("\n--- LSTM ---\n")

    for (ws in window_sizes) {
      cat(sprintf("\n  Window size: %dh\n", ws))

      lstm_data <- lstm_specific_preprocessing(
        scaled$train, scaled$val, scaled$test,
        window_size = ws, ts_names_col = site_name_col
      )

      # HPO
      hpo_result <- tryCatch({
        lstm_hypertuning(
          lstm_data$train_dict$X, lstm_data$train_dict$y,
          lstm_data$val_dict$X, lstm_data$val_dict$y,
          n_trials = 5, epochs = LSTM_EPOCHS,
          batch_size = LSTM_BATCH, patience = LSTM_PATIENCE,
          seed = SEED
        )
      }, error = function(e) {
        cat(sprintf("    [ERROR] LSTM HPO failed: %s\n", e$message))
        NULL
      })

      if (is.null(hpo_result)) next

      best_params <- hpo_result$params

      # Train at different sizes
      for (n_days in TRAINING_DAYS) {
        n_hours <- n_days * 24

        # Subsample training windows
        total_train <- length(lstm_data$train_dict$y)
        k <- min(total_train, n_hours)
        if (k == 0) next

        train_idx <- seq_len(k)
        X_partial <- lstm_data$train_dict$X[train_idx, , , drop = FALSE]
        y_partial <- lstm_data$train_dict$y[train_idx]

        for (run_id in seq_len(n_runs)) {
          model <- tryCatch({
            train_lstm(
              X_partial, y_partial,
              lstm_data$val_dict$X, lstm_data$val_dict$y,
              n_units = best_params$n_units,
              n_layers = best_params$n_layers,
              dropout = best_params$dropout,
              lr = best_params$lr,
              epochs = LSTM_EPOCHS,
              batch_size = LSTM_BATCH,
              patience = LSTM_PATIENCE,
              seed = run_id - 1
            )
          }, error = function(e) {
            cat(sprintf("    [ERROR] LSTM training failed: %s\n", e$message))
            NULL
          })

          if (is.null(model)) next

          metrics <- evaluate_correction(
            model, lstm_data$test_dict$X, lstm_data$test_dict$y,
            lstm_data$test_dict$base_pred, model_type = "lstm"
          )

          results[[length(results) + 1]] <- data.frame(
            model = sprintf("LSTM_%dh", ws),
            window_size = ws,
            perc = n_days,
            train_size = k,
            ts_name = "ALL",
            rmse_corr = metrics$rmse_corr,
            rmse_base = metrics$rmse_base,
            run = run_id - 1,
            stringsAsFactors = FALSE
          )
        }

        cat(sprintf("    LSTM_%dh: %d days (%d pts), mean RMSE_corr=%.3f\n",
                    ws, n_days, k,
                    mean(sapply(tail(results, n_runs), function(x) x$rmse_corr))))
      }
    }
  }

  # --- 6. Save results ---
  results_df <- do.call(rbind, results)

  safe_name <- gsub("[/ .]+", "_", gsub("\\.csv$", "", site_name))
  site_output_dir <- file.path(output_dir, safe_name)
  dir.create(site_output_dir, recursive = TRUE, showWarnings = FALSE)

  results_path <- file.path(site_output_dir, "model_performance_results.csv")
  write.csv(results_df, results_path, row.names = FALSE)
  cat(sprintf("\nResults saved to: %s\n", results_path))

  results_df
}

# ===========================================================
# COMPARISON FUNCTION
# ===========================================================
compare_with_python <- function(r_results_dir, python_results_dir, output_path) {

  cat("\n============================================\n")
  cat("COMPARING R vs PYTHON RESULTS\n")
  cat("============================================\n")

  # Collect R results
  r_files <- list.files(r_results_dir, pattern = "model_performance_results\\.csv",
                        recursive = TRUE, full.names = TRUE)
  r_all <- do.call(rbind, lapply(r_files, function(f) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    df$site <- basename(dirname(f))
    df$pipeline <- "R"
    df
  }))

  # Collect Python results (matching sites only)
  r_sites <- unique(r_all$site)
  py_files <- list.files(python_results_dir, pattern = "model_performance_results\\.csv",
                         recursive = TRUE, full.names = TRUE)
  py_all <- do.call(rbind, lapply(py_files, function(f) {
    site <- basename(dirname(f))
    if (site %in% r_sites) {
      df <- read.csv(f, stringsAsFactors = FALSE)
      df$site <- site
      df$pipeline <- "Python"
      df
    }
  }))

  if (is.null(py_all) || nrow(py_all) == 0) {
    cat("No matching Python results found.\n")
    return(invisible(NULL))
  }

  # Compare: group by model, perc (days), and compute mean RMSE
  combined <- rbind(r_all, py_all)
  combined <- combined[combined$ts_name == "ALL", ]

  comparison <- combined %>%
    group_by(pipeline, model, perc, site) %>%
    summarize(
      mean_rmse_corr = mean(rmse_corr, na.rm = TRUE),
      sd_rmse_corr = sd(rmse_corr, na.rm = TRUE),
      mean_rmse_base = mean(rmse_base, na.rm = TRUE),
      n_runs = n(),
      .groups = "drop"
    )

  # Print summary table
  cat("\n--- Per-site, per-model summary ---\n")
  for (s in unique(comparison$site)) {
    cat(sprintf("\n=== Site: %s ===\n", s))
    site_comp <- comparison[comparison$site == s, ]

    # Show RF comparison at max training days
    max_day <- max(site_comp$perc[site_comp$model == "RF"], na.rm = TRUE)
    rf_comp <- site_comp[site_comp$model == "RF" & site_comp$perc == max_day, ]
    if (nrow(rf_comp) > 0) {
      cat(sprintf("  RF (full data, %d days):\n", max_day))
      for (p in unique(rf_comp$pipeline)) {
        row <- rf_comp[rf_comp$pipeline == p, ]
        cat(sprintf("    %s: RMSE=%.3f ± %.3f (base=%.3f)\n",
                    p, row$mean_rmse_corr, row$sd_rmse_corr, row$mean_rmse_base))
      }
    }

    # Show LSTM_2h comparison
    lstm_comp <- site_comp[site_comp$model == "LSTM_2h" & site_comp$perc == max_day, ]
    if (nrow(lstm_comp) > 0) {
      cat(sprintf("  LSTM_2h (full data, %d days):\n", max_day))
      for (p in unique(lstm_comp$pipeline)) {
        row <- lstm_comp[lstm_comp$pipeline == p, ]
        cat(sprintf("    %s: RMSE=%.3f ± %.3f\n",
                    p, row$mean_rmse_corr, row$sd_rmse_corr))
      }
    }
  }

  # Save comparison
  write.csv(comparison, output_path, row.names = FALSE)
  cat(sprintf("\nComparison saved to: %s\n", output_path))

  comparison
}

# ===========================================================
# MAIN: Run on selected loggers
# ===========================================================
main <- function() {
  cat("=== microclCorr Phase 1 Pipeline ===\n\n")

  # --- Harod Valley loggers (3 loggers) ---
  cat("Loading Harod Valley data...\n")
  harod_data <- load_prepared_csv_data(
    HAROD_PATH,
    is_continuous_microhabitat = FALSE,
    datetime_format = "%d/%m/%Y %H:%M",
    includes_index = TRUE
  )

  harod_sites <- unique(harod_data$time_series_doc)
  cat(sprintf("Found %d Harod loggers: %s\n", length(harod_sites),
              paste(harod_sites, collapse = ", ")))

  # Run each logger individually (Phase 1)
  for (site in harod_sites) {
    site_data <- harod_data[harod_data$time_series_doc == site, , drop = FALSE]

    # Start with RF only for speed; add LSTM later
    run_single_logger(
      data = site_data,
      site_name = site,
      site_name_col = "time_series_doc",
      window_sizes = c(2),  # Just 2h window for initial comparison
      n_runs = 3,           # 3 runs to start (vs 5 in Python)
      use_lstm = FALSE       # Start with RF only for initial validation
    )
  }

  # --- Comparison with Python results ---
  compare_with_python(
    r_results_dir = OUTPUT_DIR,
    python_results_dir = PYTHON_RESULTS_DIR,
    output_path = file.path(OUTPUT_DIR, "r_vs_python_comparison.csv")
  )

  cat("\n=== Phase 1 Complete ===\n")
}

# Run if executed as script
if (sys.nframe() == 0L || !interactive()) {
  main()
}
