# ---- microclCorr: Shared Utilities ----

#' Default column names used throughout the package
#' @keywords internal
.default_cols <- list(
  datetime    = "time",
  target      = "residual",
  prediction  = "predicted",
  microhabitat = "microhabitat",
  ts_names    = "time_series_doc",
  avoid       = c("time_series_doc", "time_series_site", "TIME", "time",
                   "location", "site_id")
)

#' Save a correction model to disk
#'
#' @param model Trained model (ranger or keras)
#' @param scaler List with min/max from scaling
#' @param feature_cols Character vector of feature column names
#' @param path File path to save to (will create .rds file)
#' @export
save_correction_model <- function(model, scaler, feature_cols, path) {
  obj <- list(
    model        = model,
    scaler       = scaler,
    feature_cols = feature_cols,
    model_type   = if (inherits(model, "ranger")) "rf" else "lstm"
  )
  saveRDS(obj, path)
  invisible(path)
}

#' Load a correction model from disk
#'
#' @param path File path to the .rds model
#' @return List with model, scaler, feature_cols, model_type
#' @export
load_correction_model <- function(path) {
  obj <- readRDS(path)
  # If it's an LSTM, load the .keras file
  if (obj$model_type == "lstm" || (is.character(obj$model) && length(obj$model) > 0)) {
    keras_path <- sub("\\.rds$", ".keras", path)
    if (file.exists(keras_path)) {
      obj$model <- keras3::load_model(keras_path)
    }
  }
  obj
}

#' Setup Tensorflow environment
#' @export
setup_tensorflow <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required to set up TensorFlow. Please install it using install.packages('reticulate').")
  }
  
  if (Sys.getenv("KERAS_HOME") == "") {
    Sys.setenv(KERAS_HOME = normalizePath("."))
  }
  
  env_name <- "microcl_env"
  if (!reticulate::virtualenv_exists(env_name)) {
    reticulate::virtualenv_create(env_name, packages = c("tensorflow", "keras"))
  }
  reticulate::use_virtualenv(env_name, required = TRUE)
  
  # Ensure python works
  tryCatch({
    system2(reticulate::py_exe(), args = c("-c", "import tensorflow; print('ok')"), stdout = FALSE, stderr = FALSE)
  }, error = function(e) {})
}

#' Get path to an example dataset (downloading on demand if needed)
#'
#' Retrieves the path to an example dataset. If the dataset exists locally
#' in the package installation or local directory, its local path is returned.
#' Otherwise, it is downloaded on demand from the repository and cached.
#'
#' @param filename Character. Name of the dataset file (e.g. \code{"Harod_dataset.csv"},
#'   \code{"Beach_data_preprocessed.csv"}, \code{"desert_data_preprocessed.csv"},
#'   \code{"beach_splits.csv"}, \code{"desert_splits.csv"}).
#' @param dest_dir Character. Directory where downloaded files should be cached.
#'   Defaults to \code{tools::R_user_dir("microclCorr", "data")}.
#' @param base_url Character. Base URL for on-demand downloads.
#' @param force Logical. If \code{TRUE}, force re-download even if the file exists.
#' @return Character. Absolute file path to the local dataset.
#' @export
#' @examples
#' \dontrun{
#' data_path <- get_example_data("Harod_dataset.csv")
#' df <- read.csv(data_path)
#' }
get_example_data <- function(filename,
                             dest_dir = NULL,
                             base_url = "https://anonymous.4open.science/r/microcl_ml_corr-3E14/inst/extdata/",
                             force = FALSE) {
  # 1. Check if the file is bundled inside inst/extdata in installed package
  pkg_file <- system.file("extdata", filename, package = "microclCorr")
  if (!force && nzchar(pkg_file) && file.exists(pkg_file)) {
    return(normalizePath(pkg_file))
  }

  # Check local development paths relative to current working directory
  candidates <- c(
    file.path("inst", "extdata", filename),
    file.path("..", "extdata", filename),
    file.path("..", "..", "extdata", filename),
    file.path("extdata", filename)
  )
  for (cand in candidates) {
    if (!force && file.exists(cand)) {
      return(normalizePath(cand))
    }
  }

  # 2. Determine cache destination directory
  if (is.null(dest_dir)) {
    dest_dir <- tryCatch(
      tools::R_user_dir("microclCorr", "data"),
      error = function(e) file.path(tempdir(), "microclCorr_data")
    )
  }

  target_file <- file.path(dest_dir, filename)
  if (!force && file.exists(target_file) && file.size(target_file) > 0) {
    return(normalizePath(target_file))
  }

  # 3. Download on demand from anonymous repository
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  }

  clean_base <- sub("/+$", "", base_url)
  download_url <- paste0(clean_base, "/", filename)

  message("Downloading example dataset '", filename, "' from repository...")

  err <- tryCatch({
    utils::download.file(
      url      = download_url,
      destfile = target_file,
      mode     = "wb",
      quiet    = FALSE
    )
    NULL
  }, error = function(e) e)

  if (!is.null(err) || !file.exists(target_file) || file.size(target_file) == 0) {
    if (file.exists(target_file)) unlink(target_file)
    stop("Failed to download '", filename, "' from: ", download_url,
         if (!is.null(err)) paste0("\nError: ", err$message) else "")
  }

  message("Dataset cached at: ", normalizePath(target_file))
  normalizePath(target_file)
}


