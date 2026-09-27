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

.onLoad <- function(libname, pkgname) {
  # Block reticulate and keras3 from automatically downloading/installing packages via uv or ephemeral venvs
  if (Sys.getenv("RETICULATE_AUTOCONFIGURE") == "") {
    Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  }
  if (Sys.getenv("RETICULATE_USE_MANAGED_VENV") == "") {
    Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")
  }
}

#' Check if keras3 and required backend are installed without auto-installing packages
#' @param check_backend Logical. Whether to verify the Python backend is available.
#' @keywords internal
check_keras3 <- function(check_backend = TRUE) {
  # Explicitly ensure auto-configuration and auto-downloading are disabled
  Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")

  if (!requireNamespace("keras3", quietly = TRUE) ||
      !requireNamespace("reticulate", quietly = TRUE) ||
      !requireNamespace("tensorflow", quietly = TRUE)) {
    stop(
      "LSTM models require a configured TensorFlow environment.\n",
      "Please run setup_tensorflow() first to configure your TensorFlow environment.",
      call. = FALSE
    )
  }

  if (check_backend) {
    has_backend <- tryCatch({
      if (!reticulate::py_available(initialize = TRUE)) return(FALSE)
      reticulate::py_module_available("keras") || reticulate::py_module_available("tensorflow")
    }, error = function(e) FALSE)

    if (!has_backend) {
      stop(
        "A Python environment with 'keras' or 'tensorflow' is required for LSTM operations, but none is currently configured.\n",
        "Please run setup_tensorflow() first to configure your TensorFlow environment.",
        call. = FALSE
      )
    }
  }
}

#' Check if tensorflow is installed
#' @keywords internal
check_tensorflow <- function() {
  if (!requireNamespace("tensorflow", quietly = TRUE) ||
      !requireNamespace("reticulate", quietly = TRUE) ||
      !requireNamespace("keras3", quietly = TRUE)) {
    stop(
      "LSTM models require a configured TensorFlow environment.\n",
      "Please run setup_tensorflow() first to configure your TensorFlow environment.",
      call. = FALSE
    )
  }
}

#' Save a correction model to disk
#'
#' Saves a trained correction model (Random Forest or LSTM) and its metadata
#' to a single \code{.rds} file. Random Forest models are serialized natively.
#' LSTM (Keras) models are saved to a temporary file, serialized as raw binary
#' bytes, and embedded directly inside the \code{.rds} file, avoiding dead
#' Python pointers and providing a self-contained bundle across R sessions.
#'
#' @param model Trained model (ranger or keras)
#' @param scaler List with min/max from scaling
#' @param feature_cols Character vector of feature column names
#' @param path File path to save to (will create .rds file)
#' @return Invisible file path to the saved model file.
#' @examples
#' data(microclimate_sample)
#' feature_cols <- c("TAREF", "RH", "VREF", "SOLR")
#' rf <- train_rf(microclimate_sample[, feature_cols], microclimate_sample$residual, num_trees = 5, tune = FALSE)
#' tmp <- tempfile(fileext = ".rds")
#' save_correction_model(rf, scaler = NULL, feature_cols = feature_cols, path = tmp)
#' unlink(tmp)
#' @export
save_correction_model <- function(model, scaler, feature_cols, path) {
  is_rf      <- inherits(model, "ranger")
  model_type <- if (is_rf) "rf" else "lstm"

  keras_bytes <- NULL
  if (!is_rf) {
    check_keras3()
    keras_tmp <- tempfile(fileext = ".keras")
    keras3::save_model(model, keras_tmp)
    keras_bytes <- readBin(keras_tmp, "raw", file.info(keras_tmp)$size)
    unlink(keras_tmp)
    model <- NULL # don't embed the Python pointer
  }

  obj <- list(
    model        = model,
    keras_bytes  = keras_bytes,
    scaler       = scaler,
    feature_cols = feature_cols,
    model_type   = model_type
  )
  saveRDS(obj, path)
  invisible(path)
}

#' Load a correction model from disk
#'
#' Loads a model bundle previously saved with \code{\link{save_correction_model}}.
#' Supports both Random Forest models and LSTM models serialized as raw binary
#' bytes or sidecar \code{.keras} files.
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
  if (identical(obj$model_type, "lstm") || (!is.null(obj$keras_bytes)) ||
      (is.character(obj$model) && length(obj$model) > 0)) {
    check_keras3()
    keras_path <- sub("\\.rds$", ".keras", path)
    if (file.exists(keras_path)) {
      obj$model <- keras3::load_model(keras_path)
    } else if (!is.null(obj$keras_bytes)) {
      keras_tmp <- tempfile(fileext = ".keras")
      writeBin(obj$keras_bytes, keras_tmp)
      obj$model <- keras3::load_model(keras_tmp)
      unlink(keras_tmp)
    } else {
      # Fallback for models saved with serialize_keras_object
      tryCatch({
        obj$model <- keras3::deserialize_keras_object(obj$model)
      }, error = function(e) {
        stop("Failed to load LSTM model. Dead python pointer and no side-by-side .keras file or embedded model bytes found.", call. = FALSE)
      })
    }
  }
  obj
}

#' Setup Tensorflow environment
#'
#' Configures a Python environment with TensorFlow and Keras. Installs any
#' missing required R packages (reticulate, tensorflow, keras3) and creates/configures
#' a dedicated virtual environment with TensorFlow and Keras if no existing
#' environment is found.
#'
#' @param envname Name of the virtual environment to use or create (default: "microcl_env").
#' @param install_if_missing Logical. If TRUE (default), automatically installs missing
#'   required R packages and creates/installs the Python virtual environment if needed.
#' @return Invisible file path to the Python executable, or NULL.
#' @examples
#' \dontrun{
#'   setup_tensorflow()
#' }
#' @export
setup_tensorflow <- function(envname = "microcl_env", install_if_missing = TRUE) {
  # Block reticulate and keras3 from automatically downloading/installing packages via uv or ephemeral venvs
  Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")

  missing_r_pkgs <- c()
  if (!requireNamespace("reticulate", quietly = TRUE)) missing_r_pkgs <- c(missing_r_pkgs, "reticulate")
  if (!requireNamespace("tensorflow", quietly = TRUE)) missing_r_pkgs <- c(missing_r_pkgs, "tensorflow")
  if (!requireNamespace("keras3", quietly = TRUE))     missing_r_pkgs <- c(missing_r_pkgs, "keras3")

  if (length(missing_r_pkgs) > 0) {
    if (isTRUE(install_if_missing)) {
      message("setup_tensorflow: installing missing R packages (", paste(missing_r_pkgs, collapse = ", "), ")...")
      utils::install.packages(missing_r_pkgs, repos = "https://cloud.r-project.org")
    } else {
      stop(
        "The following R packages are required for LSTM operations but are not installed: ",
        paste(missing_r_pkgs, collapse = ", "), ".\n",
        "Please install them manually using:\n",
        "  install.packages(c(", paste(sprintf("'%s'", missing_r_pkgs), collapse = ", "), "))",
        call. = FALSE
      )
    }
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

  # 1. Check existing virtual environment (envname)
  if (reticulate::virtualenv_exists(envname)) {
    py <- tryCatch(reticulate::virtualenv_python(envname), error = function(e) NULL)
    if (!is.null(py) && file.exists(py)) {
      has_tf <- tryCatch({
        res <- suppressWarnings(
          system2(py, args = c("-c", "\"import tensorflow\""),
                  stdout = FALSE, stderr = FALSE))
        identical(res, 0L)
      }, error = function(e) FALSE)
      if (has_tf) {
        Sys.setenv(RETICULATE_PYTHON = py)
        message("setup_tensorflow: using virtualenv '", envname, "' (", py, ")")
        return(invisible(py))
      } else if (isTRUE(install_if_missing)) {
        message("setup_tensorflow: installing tensorflow into virtual environment '", envname, "'...")
        tryCatch({
          reticulate::virtualenv_install(envname, packages = c("tensorflow", "keras"))
          Sys.setenv(RETICULATE_PYTHON = py)
          message("setup_tensorflow: successfully configured virtualenv '", envname, "' (", py, ")")
          return(invisible(py))
        }, error = function(e) NULL)
      }
    }
  }

  # 2. Check conda environments if conda is available
  conda_envs <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
  if (!is.null(conda_envs) && nrow(conda_envs) > 0) {
    for (i in seq_len(nrow(conda_envs))) {
      py <- conda_envs$python[i]
      if (file.exists(py) && file.access(py, 1) == 0) {
        has_tf <- tryCatch({
          res <- suppressWarnings(
            system2(py, args = c("-c", "\"import tensorflow\""),
                    stdout = FALSE, stderr = FALSE))
          identical(res, 0L)
        }, error = function(e) FALSE)
        if (has_tf) {
          Sys.setenv(RETICULATE_PYTHON = py)
          message("setup_tensorflow: using conda environment '", conda_envs$name[i], "' (", py, ")")
          return(invisible(py))
        }
      }
    }
  }

  # 3. Check reticulate cache if present
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
          message("setup_tensorflow: using cached python ", py)
          return(invisible(py))
        }
      }
    }
  }

  # 4. If no existing environment with TensorFlow was found, install or stop with instructions
  if (isTRUE(install_if_missing)) {
    message("setup_tensorflow: creating virtual environment '", envname, "' with tensorflow and keras...")
    reticulate::virtualenv_create(envname, packages = c("tensorflow", "keras"))
    py <- reticulate::virtualenv_python(envname)
    Sys.setenv(RETICULATE_PYTHON = py)
    message("setup_tensorflow: successfully configured virtualenv '", envname, "' (", py, ")")
    return(invisible(py))
  }

  stop(
    "No existing Python environment with 'tensorflow' was found.\n",
    "Please run setup_tensorflow() with install_if_missing = TRUE or configure manually:\n",
    "  1. If you already have an environment with TensorFlow/Keras:\n",
    "     Sys.setenv(RETICULATE_PYTHON = '/path/to/python')\n",
    "     or in R: reticulate::use_condaenv('your_env')\n",
    "     or in R: reticulate::use_virtualenv('your_env')\n\n",
    "  2. To create/install TensorFlow in a virtual environment manually:\n",
    "     reticulate::virtualenv_create('", envname, "', packages = c('tensorflow', 'keras'))\n",
    "     reticulate::use_virtualenv('", envname, "')\n",
    "     or in shell terminal: pip install tensorflow keras\n",
    call. = FALSE
  )
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


