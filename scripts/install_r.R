#!/usr/bin/env Rscript
# Install the unchanged official source archive, without any network access.
file_arg <- grep("^--file=", commandArgs(), value=TRUE)
root <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
lib <- file.path(root, ".r-library")
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
stopifnot(getRversion() >= "4.1.0")
for (p in c("jsonlite", "ggplot2", "plotly")) {
  if (!requireNamespace(p, quietly=TRUE)) stop("Missing environment dependency: ", p)
}
if (requireNamespace("MTest", quietly=TRUE) && as.character(packageVersion("MTest")) == "1.0.4") {
  message("MTest 1.0.4 already installed; no changes.")
} else {
  install.packages(file.path(root, "vendor/MTest_1.0.4.tar.gz"),
                   repos=NULL, type="source", lib=lib)
  stopifnot(read.dcf(file.path(lib, "MTest/DESCRIPTION"))[1,"Version"] == "1.0.4")
  message("Installed official MTest 1.0.4.")
}
