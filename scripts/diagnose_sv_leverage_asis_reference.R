#!/usr/bin/env Rscript

# Reference ASIS diagnostic for Gaussian stochastic volatility with leverage.
#
# This uses the established leverage sampler in the CRAN package `stochvol`.
# It is intentionally diagnostic-only.  The sampler can match our priors for
# mu, phi, rho and stationary h0 exactly, but it cannot represent our
# Half-Cauchy(0.5) prior on sigma_eta.  For sigma_eta^2 we therefore use an
# inverse-gamma reference prior with shape 0.5 and scale chosen so that the
# median of sigma_eta^2 is approximately 0.25, matching the Half-Cauchy scale's
# median after squaring.  This retains a heavy right tail but is not identical.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) stop(paste("Missing value for", flag))
  args[[idx + 1]]
}

case <- get_arg("--case")
window <- as.integer(get_arg("--window", "1000"))
outdir <- get_arg("--output-dir", "diagnostics/asis_reference")
if (is.null(case) || !(case %in% c("oil", "synthetic"))) {
  stop("--case must be one of: oil, synthetic")
}
if (!is.finite(window) || window < 100) stop("invalid --window")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(stochvol)
  library(coda)
})

truth <- c(mu = -9.0, phi = 0.97, sigma = 0.22, rho = -0.50)

simulate_leverage <- function(n, mu, phi, sigma, rho, seed = 20260917L) {
  set.seed(seed)
  eta <- rnorm(n)
  xi <- rnorm(n)
  h <- numeric(n + 1L)
  h[[1]] <- mu + sigma / sqrt(1 - phi^2) * rnorm(1)
  y <- numeric(n)
  orth <- sqrt(1 - rho^2)
  for (t in seq_len(n)) {
    eps <- rho * eta[[t]] + orth * xi[[t]]
    y[[t]] <- exp(h[[t]] / 2) * eps
    h[[t + 1L]] <- mu + phi * (h[[t]] - mu) + sigma * eta[[t]]
  }
  y
}

load_oil_returns <- function(n) {
  path <- file.path("data", "raw", "brent_oil_primary.csv")
  raw <- read.csv(path, header = FALSE, stringsAsFactors = FALSE)
  if (ncol(raw) < 2) stop("unexpected Brent oil file format")
  names(raw)[1:2] <- c("Date", "Close")
  keep <- grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", raw$Date)
  raw <- raw[keep, c("Date", "Close")]
  raw$Date <- as.Date(raw$Date)
  raw$Close <- suppressWarnings(as.numeric(raw$Close))
  raw <- raw[raw$Date >= as.Date("2003-01-01") &
             raw$Date <= as.Date("2026-06-22") &
             is.finite(raw$Close) & raw$Close > 0, ]
  raw <- raw[order(raw$Date), ]
  r <- diff(log(raw$Close))
  r <- r[is.finite(r)]
  if (length(r) < n) stop("oil series does not contain requested window")
  as.numeric(r[seq_len(n)])
}

if (case == "synthetic") {
  y <- simulate_leverage(window, truth[["mu"]], truth[["phi"]],
                         truth[["sigma"]], truth[["rho"]])
} else {
  y <- load_oil_returns(window)
}
y <- as.numeric(y - mean(y))

# Half-Cauchy(0.5) on sigma has median sigma=0.5, hence median sigma^2=0.25.
# If X ~ InvGamma(a=0.5, scale=b), 1/X ~ Gamma(a=0.5, rate=b).
# Choosing b = 0.25 * qgamma(0.5, shape=0.5, rate=1) matches that median.
sigma2_scale <- 0.25 * qgamma(0.5, shape = 0.5, rate = 1)

priors <- specify_priors(
  mu = sv_normal(mean = -10, sd = 3),
  phi = sv_beta(shape1 = 20, shape2 = 1.5),
  sigma2 = sv_inverse_gamma(shape = 0.5, scale = sigma2_scale),
  nu = sv_infinity(),
  rho = sv_beta(shape1 = 1, shape2 = 1),
  latent0_variance = "stationary"
)

# The general leverage sampler uses ASIS when interweave=TRUE.  We also request
# correction for the auxiliary-mixture approximation so this reference run is
# not knowingly relying on the uncorrected approximation.
set.seed(if (case == "synthetic") 20260917L else 42L)
fit <- svlsample(
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

wanted <- c("mu", "phi", "sigma", "rho")
for (i in seq_len(coda::nchain(par_draws))) {
  mat <- as.matrix(par_draws[[i]])
  missing <- setdiff(wanted, colnames(mat))
  if (length(missing)) stop(paste("missing sampled parameters:", paste(missing, collapse = ", ")))
  write.csv(mat[, wanted, drop = FALSE],
            file.path(outdir, sprintf("%s_chain_%d.csv", case, i)),
            row.names = FALSE)
}

metadata <- data.frame(
  case = case,
  window = window,
  draws_per_chain = 10000L,
  burnin = 5000L,
  n_chains = 4L,
  interweave = TRUE,
  correct_model_misspecification = TRUE,
  stochvol_version = as.character(packageVersion("stochvol")),
  sigma2_reference_prior = sprintf("InvGamma(shape=0.5,scale=%.12g)", sigma2_scale),
  sigma2_reference_prior_median = 0.25,
  production_sigma_prior = "HalfCauchy(scale=0.5) on sigma_eta",
  prior_match_exact = FALSE,
  truth_mu = if (case == "synthetic") truth[["mu"]] else NA_real_,
  truth_phi = if (case == "synthetic") truth[["phi"]] else NA_real_,
  truth_sigma = if (case == "synthetic") truth[["sigma"]] else NA_real_,
  truth_rho = if (case == "synthetic") truth[["rho"]] else NA_real_,
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(outdir, sprintf("%s_metadata.csv", case)), row.names = FALSE)

cat(sprintf("Completed stochvol ASIS reference diagnostic for %s.\n", case))
