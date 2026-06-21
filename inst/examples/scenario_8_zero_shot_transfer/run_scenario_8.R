# =============================================================================
# Scenario 8: Zero-Shot Spatial Transfer
# =============================================================================
# Goal: Test whether a model trained on NEARBY sites can correct NicheMapR
#       predictions at a completely new location where NO logger data exists.
#
# This is the practical deployment scenario: a researcher wants to correct
# NicheMapR at a new field site before any sensors have been installed.
#
# For each beach location in turn, we pretend it is "new" (unseen) and train
# using only the OTHER locations. We then compare four strategies:
#
#   A) Zero-Shot (Nearby Sites) — train on the other two locations only.
#      This is the true zero-shot scenario: the target site was never seen.
#
#   B) Specialized (Local Data) — train on local data only (upper bound).
#      This is how well we can do if we have local measurements.
#
#   C) Pooled (All Sites) — train on all three locations including the target.
#      This is the best-case pooled scenario from Scenario 4.
#
#   D) Pooled (Downsampled to N) — train on the same NUMBER of rows as B,
#      but drawn randomly from all three locations (not just the local one).
#      This isolates whether the gain from pooling comes from spatial diversity
#      or simply from having more data.
#
# Compare with: Scenario 2 (single logger, ~1,405 train rows)
# Note: strategy B uses all Ashkelon loggers combined (~4,631 rows), which
#       is already ~3× more data than the single logger in Scenario 2.
# =============================================================================

library(microclCorr)
library(ggplot2)
source(system.file("examples", "utils.R", package = "microclCorr"))

# ── Settings ──────────────────────────────────────────────────────────────────
SEED         <- 42
SITE_COL     <- "time_series_site"
N_DOWNSAMPLE <- 10   # number of random subsamples to average for strategy D

DATA_PATH    <- system.file("extdata", "Beach_data_preprocessed.csv", package = "microclCorr")
SPLITS_PATH  <- system.file("extdata", "beach_splits.csv",            package = "microclCorr")
RESULTS_DIR  <- file.path("inst", "examples", "scenario_8_zero_shot_transfer", "results")
SCENARIO_DIR <- file.path("inst", "examples", "scenario_8_zero_shot_transfer")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Scenario 8: Zero-Shot Spatial Transfer ===\n")

# ── Step 1: Load data ─────────────────────────────────────────────────────────
data <- load_prepared_csv_data(DATA_PATH,
                               is_continuous_microhabitat = FALSE,
                               datetime_format = "%Y-%m-%d %H:%M:%S",
                               includes_index  = TRUE)
if ("microhabitat_sun" %in% names(data)) data$microhabitat_sun <- NULL

# ── Step 2: Split into train / validation / test ───────────────────────────────
splits <- load_splits_from_csv(data, SPLITS_PATH, SITE_COL)

# ── Step 3: Select predictor columns ──────────────────────────────────────────
feature_cols <- get_feature_columns(splits$train)

# ── Steps 4–7: Loop — treat each location as the unseen target in turn ────────
results <- list()

for (target in c("Ashkelon", "Range_24", "Rosh_HaNikra")) {
  cat(sprintf("\n── Target location (held out): %s ──\n", target))

  # Test rows = the target location only (these rows are NEVER used in training
  # for strategies A and D)
  test_loc  <- splits$test[splits$test$location == target, ]
  y_test    <- test_loc$residual    # actual residuals (what we want to predict)
  base_test <- test_loc$predicted   # NicheMapR raw prediction (no correction)
  rmse_raw  <- sqrt(mean(y_test^2)) # baseline: error if we apply NO correction

  # ── Strategy A: Zero-Shot — train on the other two locations ────────────────
  train_other <- splits$train[splits$train$location != target, ]
  rf_zs <- train_rf(train_other[, feature_cols], train_other$residual, seed = SEED)
  m_zs  <- evaluate_correction(rf_zs, test_loc[, feature_cols], y_test, base_test,
                                model_type = "rf")
  results[[length(results) + 1]] <- data.frame(
    target = target, strategy = "A: Zero-Shot (Nearby Sites)",
    train_size = nrow(train_other), rmse_base = rmse_raw, rmse_corr = m_zs$rmse_corr)

  # ── Strategy B: Specialized — train on local data only (best case for local) ──
  train_local <- splits$train[splits$train$location == target, ]
  rf_loc <- train_rf(train_local[, feature_cols], train_local$residual, seed = SEED)
  m_loc  <- evaluate_correction(rf_loc, test_loc[, feature_cols], y_test, base_test,
                                 model_type = "rf")
  results[[length(results) + 1]] <- data.frame(
    target = target, strategy = "B: Specialized (Local Data)",
    train_size = nrow(train_local), rmse_base = rmse_raw, rmse_corr = m_loc$rmse_corr)

  # ── Strategy C: Pooled — train on all three locations (includes the target) ──
  rf_all <- train_rf(splits$train[, feature_cols], splits$train$residual, seed = SEED)
  m_all  <- evaluate_correction(rf_all, test_loc[, feature_cols], y_test, base_test,
                                 model_type = "rf")
  results[[length(results) + 1]] <- data.frame(
    target = target, strategy = "C: Pooled (All Sites)",
    train_size = nrow(splits$train), rmse_base = rmse_raw, rmse_corr = m_all$rmse_corr)

  # ── Strategy D: Downsampled pooled — same N as local, but mixed from all sites ──
  # By matching the training set size to strategy B, we can check whether the
  # difference between B and C comes from MORE DATA or from DIVERSE LOCATIONS.
  N_local  <- nrow(train_local)
  ds_rmses <- vapply(seq_len(N_DOWNSAMPLE), function(s) {
    set.seed(s)
    idx   <- sample(nrow(splits$train), N_local)   # random subset of size N_local
    rf_ds <- train_rf(splits$train[idx, feature_cols],
                       splits$train$residual[idx], seed = s)
    evaluate_correction(rf_ds, test_loc[, feature_cols], y_test, base_test,
                         model_type = "rf")$rmse_corr
  }, numeric(1))

  results[[length(results) + 1]] <- data.frame(
    target = target, strategy = "D: Pooled (Downsampled to N)",
    train_size = N_local,
    rmse_base  = rmse_raw,
    rmse_corr  = mean(ds_rmses))   # average over N_DOWNSAMPLE random draws
}

results_df <- do.call(rbind, results)
results_df$improvement_pct <- (results_df$rmse_base - results_df$rmse_corr) /
                               results_df$rmse_base * 100

# ── Step 8: Save results ──────────────────────────────────────────────────────
write.csv(results_df,
          file.path(RESULTS_DIR, "zero_shot_results.csv"),
          row.names = FALSE)

# ── Plot: compare strategies side by side per location ────────────────────────
results_df$strategy <- factor(results_df$strategy,
  levels = c("B: Specialized (Local Data)", "C: Pooled (All Sites)",
             "D: Pooled (Downsampled to N)", "A: Zero-Shot (Nearby Sites)"))

p <- ggplot(results_df, aes(x = target, y = rmse_corr, fill = strategy)) +
  geom_bar(stat = "identity", position = position_dodge(0.85), width = 0.75) +
  # Red dashed line = uncorrected NicheMapR error (our starting point)
  geom_hline(aes(yintercept = rmse_base), linetype = "dashed",
             color = "#ef4444", linewidth = 0.7) +
  scale_fill_manual(values = c(
    "B: Specialized (Local Data)"    = "#10b981",
    "C: Pooled (All Sites)"          = "#3b82f6",
    "D: Pooled (Downsampled to N)"   = "#8b5cf6",
    "A: Zero-Shot (Nearby Sites)"    = "#f59e0b")) +
  labs(title    = "Zero-Shot Transfer: Can we correct an unseen location?",
       subtitle = "Red dashed line = uncorrected NicheMapR error",
       x        = "Target Location (held out during training)",
       y        = "Corrected RMSE (°C)",
       fill     = "Training Strategy") +
  theme_minimal(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "#666666"),
        legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2))

ggsave(file.path(SCENARIO_DIR, "zero_shot_transfer.png"),
       p, width = 10, height = 6, dpi = 300)

cat("\nResults:\n")
print(results_df[, c("target", "strategy", "train_size", "rmse_corr", "improvement_pct")])
cat("=== Scenario 8 complete ===\n")
