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
