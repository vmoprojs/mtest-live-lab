#!/usr/bin/env Rscript
# Official MTest bridge. No reimplementation of the test or its bootstrap.
args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
local_lib <- file.path(root, ".r-library")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))

main <- function() {
  if (!requireNamespace("MTest", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE))
    stop("Faltan MTest o jsonlite. Ejecute Rscript scripts/install_r.R")
  if (getRversion() < "4.1.0" || as.character(utils::packageVersion("MTest")) != "1.0.4")
    stop("Se requiere R >= 4.1 y MTest 1.0.4.")
  if (identical(args, "--probe")) {
    cat(jsonlite::toJSON(list(R = R.version.string,
      MTest = as.character(utils::packageVersion("MTest")),
      jsonlite = as.character(utils::packageVersion("jsonlite")),
      signature = paste(deparse(args(MTest::MTest)), collapse=" ")),
      auto_unbox = TRUE))
    return(invisible(NULL))
  }
  if (length(args) != 14 || any(!startsWith(args[seq(1, 14, 2)], "--")))
    stop("Uso: --csv archivo --response nombre --predictors JSON --nboot B --nsam N --threshold c --seed s")
  opts <- as.list(args[seq(2, 14, 2)])
  names(opts) <- sub("^--", "", args[seq(1, 14, 2)])
  required <- c("csv", "response", "predictors", "nboot", "nsam", "threshold", "seed")
  if (!setequal(names(opts), required)) stop("Argumentos incompletos o desconocidos.")
  predictors <- jsonlite::fromJSON(opts$predictors)
  d <- read.csv(opts$csv, check.names = FALSE, fileEncoding = "UTF-8", colClasses="character")
  if (anyDuplicated(names(d)) || !all(c(opts$response, predictors) %in% names(d))) stop("Columnas inválidas.")
  selected <- d[, c(opts$response, predictors), drop=FALSE]
  selected[] <- lapply(selected, as.numeric)
  d[, c(opts$response, predictors)] <- selected
  if (!all(vapply(selected, is.numeric, logical(1))) || any(!is.finite(as.matrix(selected))))
    stop("Datos no numéricos o no finitos.")
  if (any(vapply(selected, function(x) length(unique(x)) < 2, logical(1)))) stop("Variable constante.")
  # Internal aliases prevent formula parsing of arbitrary uploaded column names.
  safe_names <- sprintf("x%03d", seq_along(predictors))
  names(selected) <- c("response", safe_names)
  modelo <- lm(reformulate(safe_names, response="response"), data=selected, na.action=na.fail)
  design <- model.matrix(modelo)
  if (qr(design)$rank < ncol(design)) stop("Matriz singular.")
  nboot <- as.integer(opts$nboot); nsam <- as.integer(opts$nsam)
  threshold <- as.numeric(opts$threshold); seed <- as.integer(opts$seed)
  if (is.na(nboot) || nboot < 20 || nboot > 500 || is.na(nsam) || nsam < 20 || nsam > 2000 ||
      !is.finite(threshold) || threshold <= 0 || threshold >= 1 || is.na(seed) || seed < 0)
    stop("Parámetros fuera de rango.")
  start <- proc.time()[["elapsed"]]
  result <- MTest::MTest(object=modelo, nboot=nboot, nsam=nsam,
                        seed=seed, valor_vif=threshold, trace=FALSE)
  elapsed <- proc.time()[["elapsed"]] - start
  required_result <- c("pval_vif", "pval_klein", "Bvals", "vif.tot", "R.tot")
  if (!all(required_result %in% names(result))) stop("API MTest incompatible.")
  stopifnot(identical(colnames(result$Bvals), c("global", safe_names)))
  payload <- list(engine="R", version=as.character(utils::packageVersion("MTest")),
    runtime=R.version.string, predictors=unname(predictors), n=nrow(d), p=length(predictors),
    design_columns=c("(Intercept)", predictors), intercept=TRUE, nboot=nboot, nsam=nsam,
    threshold=threshold, seed=seed, package_seconds=elapsed,
    # Echo the parsed matrix: Python checks all values, not just a hash of the file.
    input_hex=matrix(sprintf("%a", as.matrix(d[, c(opts$response, predictors), drop=FALSE])), nrow=nrow(d)),
    pval_vif=unname(result$pval_vif[safe_names]),
    pval_klein=unname(result$pval_klein[safe_names]),
    Bvals=unname(result$Bvals),
    vif.tot=unname(result$vif.tot[safe_names]),
    R.tot=unname(result$R.tot[c("global", safe_names)]))
  cat(jsonlite::toJSON(payload, auto_unbox=TRUE, digits=NA, na="null", matrix="rowmajor"))
}
tryCatch(main(), error=function(e) {
  message("MTEST_R_ERROR: ", conditionMessage(e))
  quit(status=1)
})
