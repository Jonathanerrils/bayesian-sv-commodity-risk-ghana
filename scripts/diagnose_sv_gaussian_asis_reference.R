#!/usr/bin/env Rscript

# Gaussian SV ASIS reference sampler for cocoa/gold/oil.
# Diagnostic only: stochvol cannot natively represent the production
# Half-Cauchy(0.5) prior on sigma_eta. We therefore sample with the closest
# admissible inverse-gamma reference prior on sigma^2 and save per-draw
# importance weights that map the sigma prior back toward the production
# Half-Cauchy target. The Python summarizer diagnoses whether that reweighting
# is statistically stable enough to be informative.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) stop(paste("Missing value for", flag))
  args[[idx + 1]]
}

commodity <- get_arg("--commodity")
window <- as.integer(get_arg("--window", "1000"))
outdir <- get_arg("--output-dir", "diagnostics/asis_gaussian")
if (is.null(commodity) || !(commodity %in% c("cocoa", "gold", "oil"))) {
  stop("--commodity must be one of: cocoa, gold, oil")
}
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(stochvol)
  library(coda)
})

read_returns <- function(commodity, n) {
  if (commodity == "oil") {
    path <- file.path("data", "raw", "brent_oil_primary.csv")
    raw <- read.csv(path, header = FALSE, stringsAsFactors = FALSE)
    names(raw)[1:2] <- c("Date", "Close")
  } else if (commodity == "gold") {
    path <- file.path("data", "raw", "gold_primary.csv")
    raw <- read.csv(path, header = FALSE, stringsAsFactors = FALSE)
    names(raw)[1:2] <- c("Date", "Close")
  } else {
    path <- file.path("data", "raw", "cocoa_primary.csv")
    raw <- read.csv(path, header = FALSE, stringsAsFactors = FALSE)
    names(raw)[1:2] <- c("Date", "Close")
  }
  keep <- grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", raw$Date)
  raw <- raw[keep, c("Date", "Close")]
  raw$Date <- as.Date(raw$Date)
  raw$Close <- suppressWarnings(as.numeric(raw$Close))
  raw <- raw[is.finite(raw$Close) & raw$Close > 0, ]
  raw <- raw[order(raw$Date), ]
  r <- diff(log(raw$Close))
  r <- r[is.finite(r)]
  if (length(r) < n) stop("series does not contain requested window")
  as.numeric(r[seq_len(n)])
}

y <- read_returns(commodity, window)
y <- as.numeric(y - mean(y))

# Reference prior on sigma^2 chosen as in the earlier leverage ASIS diagnostic.
sigma2_shape <- 2.1
sigma2_target_median <- 0.25
sigma2_scale <- sigma2_target_median * qgamma(
  0.5, shape = sigma2_shape, rate = 1
)

priors <- specify_priors(
  mu = sv_normal(mean = -10, sd = 3),
  phi = sv_beta(shape1 = 20, shape2 = 1.5),
  sigma2 = sv_inverse_gamma(shape = sigma2_shape, scale = sigma2_scale),
  nu = sv_infinity(),
  rho = sv_constant(0),
  latent0_variance = "stationary"
)

set.seed(42L)
fit <- svsample(
  y,
  draws = 10000,
  burnin = 5000,
  priorspec = priors,
  thinpara = 1,
  thinlatent = 10000,
  keeptime = "last",
  quiet = TRUE,
  parallel = "snow",
  n_chains = 4,
  n_cpus = 4,
  expert = list(interweave = TRUE, correct_model_misspecification = TRUE)
)

par_draws <- para(fit, chain = "all")
if (coda::nchain(par_draws) != 4L) stop("expected four MCMC chains")

# Densities for exact-prior importance correction on sigma only.
# target sigma prior: HalfCauchy(scale=0.5)
# reference sigma prior induced by sigma^2 ~ InvGamma(a,b):
# p_sigma(s) = p_sigma2(s^2) * 2s.
log_target_sigma <- function(s) {
  log(2 / (pi * 0.5 * (1 + (s / 0.5)^2)))
}
log_ref_sigma <- function(s) {
  x <- s^2
  a <- sigma2_shape
  b <- sigma2_scale
  # InvGamma(a, scale=b)
  logp_x <- a * log(b) - lgamma(a) - (a + 1) * log(x) - b / x
  logp_x + log(2 * s)
}

wanted <- c("mu", "phi", "sigma")
for (i in seq_len(coda::nchain(par_draws))) {
  mat <- as.matrix(par_draws[[i]])
  missing <- setdiff(wanted, colnames(mat))
  if (length(missing)) stop(paste("missing sampled parameters:", paste(missing, collapse = ", ")))
  out <- as.data.frame(mat[, wanted, drop = FALSE])
  out$log_prior_ratio_sigma <- log_target_sigma(out$sigma) - log_ref_sigma(out$sigma)
  write.csv(
    out,
    file.path(outdir, sprintf("%s_chain_%d.csv", commodity, i)),
    row.names = FALSE
  )
}

metadata <- data.frame(
  commodity = commodity,
  window = window,
  draws_per_chain = 10000L,
  burnin = 5000L,
  n_chains = 4L,
  interweave = TRUE,
  correct_model_misspecification = TRUE,
  stochvol_version = as.character(packageVersion("stochvol")),
  sigma2_reference_prior = sprintf(
    "InvGamma(shape=%.3f,scale=%.12g)", sigma2_shape, sigma2_scale
  ),
  production_sigma_prior = "HalfCauchy(scale=0.5) on sigma_eta",
  prior_match_exact_native = FALSE,
  importance_reweighting_target = "exact production HalfCauchy sigma prior",
  reference_only = TRUE,
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(outdir, sprintf("%s_metadata.csv", commodity)), row.names = FALSE)
cat(sprintf("Completed Gaussian stochvol ASIS reference for %s.\n", commodity))
