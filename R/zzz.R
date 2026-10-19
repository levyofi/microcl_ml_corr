.onLoad <- functiom(libname, pkgname){
  reticulate:py_require(
    packages = c("tensorflow", "numpy", "keras")
    )
  }
