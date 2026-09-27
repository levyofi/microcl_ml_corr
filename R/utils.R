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

#' Check if keras3 is installed
#' @keywords internal
check_keras3 <- function() {
  if (!requireNamespace("keras3", quietly = TRUE)) {
    stop(
      "Package 'keras3' is required to build, train, evaluate, or load LSTM models.\n",
      "Please install it using: install.packages('keras3')\n",
      "and configure Keras with: keras3::install_keras()",
      call. = FALSE
    )
  }
}

#' Check if tensorflow is installed
#' @keywords internal
check_tensorflow <- function() {
  if (!requireNamespace("tensorflow", quietly = TRUE)) {
    stop(
      "Package 'tensorflow' is required for TensorFlow operations.\n",
      "Please install it using: install.packages('tensorflow')",
      call. = FALSE
    )
  }
}

#' Save a correction model to disk
#'
#' @param model Trained model (ranger or keras)
#' @param scaler List with min/max from scaling
#' @param feature_cols Character vector of feature column names
#' @param path File path to save to (will create .rds file)
#' @examples
#' data(microclimate_sample)
#' feature_cols <- c("TAREF", "RH", "VREF", "SOLR")
#' rf <- train_rf(microclimate_sample[, feature_cols], microclimate_sample$residual, num_trees = 5, tune = FALSE)
#' tmp <- tempfile(fileext = ".rds")
#' save_correction_model(rf, scaler = NULL, feature_cols = feature_cols, path = tmp)
#' unlink(tmp)
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
#' @examples
#' data(microclimate_sample)
#' feature_cols <- c("TAREF", "RH", "VREF", "SOLR")
#' rf <- train_rf(microclimate_sample[, feature_cols], microclimate_sample$residual, num_trees = 5, tune = FALSE)
#' tmp <- tempfile(fileext = ".rds")
#' save_correction_model(rf, scaler = NULL, feature_cols = feature_cols, path = tmp)
#' loaded <- load_correction_model(tmp)
#' unlink(tmp)
#' loaded$model_type
#' @export
load_correction_model <- function(path) {
  obj <- readRDS(path)
  # If it's an LSTM, load the .keras file
  if (obj$model_type == "lstm" || (is.character(obj$model) && length(obj$model) > 0)) {
    keras_path <- sub("\\.rds$", ".keras", path)
    if (file.exists(keras_path)) {
      check_keras3()
      obj$model <- keras3::load_model(keras_path)
    }
  }
  obj
}

#' Setup Tensorflow environment
#'
#' Finds or creates a Python environment with TensorFlow and Keras configured,
#' and sets environment variables before reticulate binds to Python.
#'
#' @return Invisible file path to the Python executable, or NULL.
#' @examples
#' \dontrun{
#' if (requireNamespace("reticulate", quietly = TRUE) &&
#'     requireNamespace("tensorflow", quietly = TRUE)) {
#'   setup_tensorflow()
#' }
#' }
#' @export
setup_tensorflow <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required to configure Python and TensorFlow.\n",
         "Please install it using: install.packages('reticulate')", call. = FALSE)
  }
  if (!requireNamespace("tensorflow", quietly = TRUE)) {
    stop("Package 'tensorflow' is required to use TensorFlow in R.\n",
         "Please install it using: install.packages('tensorflow')", call. = FALSE)
  }

  if (Sys.getenv("KERAS_HOME") == "") {
    k_dir <- tryCatch(tools::R_user_dir("microclCorr", "config"),
                      error = function(e) file.path(tempdir(), "keras_home"))
    if (!dir.exists(k_dir)) dir.create(k_dir, recursive = TRUE, showWarnings = FALSE)
    Sys.setenv(KERAS_HOME = k_dir)
  }

  if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
    return(invisible(Sys.getenv("RETICULATE_PYTHON")))
  }

  # 1. Check reticulate uv cache if present (common on macOS/Linux with reticulate >= 1.35)
  cache_root <- file.path(path.expand("~"), "Library", "Caches",
                          "org.R-project.R", "R", "reticulate", "uv",
                          "cache", "archive-v0")
  if (dir.exists(cache_root)) {
    all_files  <- list.files(cache_root, recursive = TRUE, full.names = TRUE)
    candidates <- all_files[grepl("/bin/python3?$", all_files)]
    for (py in candidates) {
      if (file.access(py, 1) == 0) {
        has_tf <- tryCatch({
          res <- suppressWarnings(
            system2(py, args = c("-c", "\"import tensorflow\""),
                    stdout = FALSE, stderr = FALSE))
          identical(res, 0L)
        }, error = function(e) FALSE)
        if (has_tf) {
          Sys.setenv(RETICULATE_PYTHON = py)
          message("setup_tensorflow: using ", py)
          return(invisible(py))
        }
      }
    }
  }

  # 2. Check or create dedicated virtual environment 'microcl_env'
  env_name <- "microcl_env"
  if (!reticulate::virtualenv_exists(env_name)) {
    reticulate::virtualenv_create(env_name, packages = c("tensorflow", "keras"))
  }
  reticulate::use_virtualenv(env_name, required = TRUE)

  py_path <- tryCatch(reticulate::py_exe(), error = function(e) NULL)
  invisible(py_path)
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


