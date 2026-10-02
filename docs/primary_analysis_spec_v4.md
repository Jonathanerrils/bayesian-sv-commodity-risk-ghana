# Corrected primary analysis specification v4

Status: **frozen before the corrected full W=1000 primary results are inspected**.

This document defines the corrected experiment after the validity audit.  It is
not an attempt to reproduce the July 2026 numerical conclusions.  If the final
results disagree with the existing manuscript, the manuscript must change.

## 1. Data and evaluation calendar

- Primary commodities: cocoa, gold and Brent crude oil.
- Primary sample end: 2026-06-22, using the repository's committed cleaning rules.
- Returns are daily log returns.
- Primary rolling estimation window: **W=1000 retained return observations**.
- Every model is evaluated against the same fixed target-date calendar for a
  commodity.  Estimation failures remain explicit rows on that calendar.
- No failed date is silently removed from a model's native record.

## 2. SV model family

All SV models use the state timing

    r_t = m + exp(h_t/2) * epsilon_t
    h_{t+1} = mu + phi (h_t - mu) + sigma_eta * eta_t

with |phi| < 1 and sigma_eta > 0.

The latent path is sampled through independent standard-normal innovations and
an explicit stationary initial state.  A rolling fit to T observed returns
contains h_0,...,h_T, so posterior h_T is the state for the next forecast date.

### SV-Gaussian

- epsilon_t ~ N(0,1)
- epsilon_t independent of eta_t.

### SV-t

- epsilon_t is Student-t with nu > 2 and is scaled to unit variance.
- epsilon_t independent of eta_t.

### SV-Leverage

- corr(epsilon_t, eta_t) = rho.
- Equivalently, epsilon_t = rho eta_t + sqrt(1-rho^2) xi_t,
  with xi_t ~ N(0,1) independent of eta_t.
- This is **forward leverage timing**: r_t is correlated with the volatility
  innovation driving h_{t+1}, not with the innovation that created h_t.

### Heavy-tailed leverage specification

The implementation currently labelled `SV-t-Leverage` uses

    epsilon_t = rho eta_t + sqrt(1-rho^2) xi_t

where xi_t is an independent, unit-variance Student-t innovation.  Therefore
rho remains the linear correlation between epsilon_t and eta_t, but the
*marginal* epsilon_t distribution is a normal--Student-t convolution and is not
literally Student-t for rho != 0.

The final manuscript must describe this specification exactly.  It must not
claim a marginal Student-t return innovation unless the model is replaced by a
joint heavy-tailed construction before the primary run is frozen.

## 3. SV priors and MCMC acceptance

- mu ~ Normal(-10, 3)
- transformed persistence prior: phi = 2*phi_raw - 1,
  phi_raw ~ Beta(20, 1.5)
- sigma_eta ~ Half-Cauchy(0.5)
- nu = 2 + Exponential(0.1) when applicable
- rho ~ Uniform(-1,1) when applicable

Primary adaptive refit policy:

1. 4 chains, 1000 tune + 1000 retained draws per chain;
2. if the strict gate fails, 4 chains, 2000 tune + 2000 retained draws per chain.

A refit is accepted only if all are true:

- max structural-parameter R-hat < 1.01;
- minimum structural-parameter bulk ESS > 400;
- zero divergences.

Structural diagnostic variables are mu, phi, sigma_eta, nu and rho where
present.  A weak trace is never used for forecasting.

## 4. Rolling SV forecast/filter schedule

- Structural parameters are scheduled for refit at global forecast indices
  0, 42, 84, ... .
- Between successful structural refits, posterior particles are filtered after
  every realised return.
- Forecast first from h_t; then condition on r_t and advance to h_{t+1}.
- Primary posterior predictive sample size: **20,000 draws per forecast date**.
- Random seeds are deterministic and depend on the fixed global forecast index,
  so canonical and distributed execution are reproducible.

If a scheduled refit fails after both MCMC attempts:

- the whole corresponding 42-day block is recorded as unavailable for that SV
  model;
- rows remain on the evaluation calendar with `estimation_failed=True` and
  NaN risk forecasts;
- no opportunistic next-day MCMC retry is allowed;
- the next independent scheduled refit is attempted at the next 42-day boundary.

Forecast availability/failure rate is a primary reliability outcome.

## 5. Benchmarks

Primary benchmark set:

1. GARCH(1,1), Gaussian innovations;
2. GARCH(1,1), standardized Student-t innovations;
3. asymmetric EGARCH(1,1), Gaussian innovations;
4. asymmetric EGARCH(1,1), standardized Student-t innovations;
5. exact-transition Ornstein-Uhlenbeck model on log prices;
6. 1000-observation historical simulation.

GARCH forecasts use the same 1000-return rolling window as the SV models.
Student-t benchmark innovations are unit-variance standardized.

EGARCH higher-order robustness fits use the actual AR characteristic-root
stability condition, not the sufficient-but-overrestrictive sum(|beta|)<1
shortcut.

## 6. OU benchmark

The OU model is

    dX_t = kappa(theta-X_t)dt + sigma dW_t

and is fitted by the exact Gaussian transition likelihood.

- OU transition pairs are aligned to the exact price transitions underlying the
  retained return observations.
- The primary interpretation is trading-time with dt=1/252 per retained return
  transition.
- Numerical optimizer convergence and economic mean-reversion strength are
  separate diagnostics.
- A small positive kappa is retained as a valid forecast; it is not an
  estimation failure merely because its half-life is long.

An irregular-calendar-time OU treatment may be reported as sensitivity, but it
must not replace the primary specification after results are seen.

## 7. VaR and ES convention

- VaR and ES are reported as positive loss magnitudes.
- A VaR violation occurs when realised return < -VaR.
- Forecast ES must be at least as large as the corresponding VaR in the loss
  tail; model implementations are checked accordingly.

Primary confidence/tail probabilities: alpha = 0.01 and 0.05.

## 8. Backtests

Primary statistical tests at each alpha:

1. Kupiec unconditional coverage;
2. Christoffersen conditional coverage;
3. Acerbi-Szekely Test 2 for Expected Shortfall.

Christoffersen independence is retained as a separate diagnostic rather than
counted as a fourth primary test.

Acerbi-Szekely reporting:

- uses observation-specific ES_t forecasts;
- preserves full numerical precision internally;
- uses an add-one Monte Carlo p-value so finite simulation cannot report p=0;
- reports Monte Carlo uncertainty around the simulated p-value;
- a multiplicity-adjusted ES decision is not forced when Monte Carlo uncertainty
  overlaps the corrected threshold.

Failure to reject an adequacy/calibration null is described as **non-rejection**,
not proof that the model is correct.

## 9. Multiplicity

Both of the following are reported regardless of which is more favourable:

### Per-test comparison-cell family

10 models x 3 commodities x 2 alpha levels = **m=60** for each primary test
family separately.  Corrected alpha = 0.05/60.

### Global primary-decision family

10 models x 3 commodities x 2 alpha levels x 3 primary tests = **m=180**.
Corrected alpha = 0.05/180.

Raw p-values remain available alongside all adjusted results.

## 10. Missing forecasts and comparison samples

Three distinct quantities must be reported:

1. **availability** on the full fixed evaluation calendar;
2. **native-sample** backtests using each model's valid forecasts;
3. **common-date** comparisons using the intersection of valid forecast dates
   across the complete model family for that commodity.

The common-date table is for like-for-like statistical comparison only.  It
must never be presented without availability/failure rates because intersection
filtering can hide operational instability of a model.

## 11. Primary production gate

The full W=1000 experiment is launched only after production-scale diagnostic
fits have exercised:

- all four SV families;
- all three commodities;
- at least one previously difficult cocoa Gaussian window;
- oil under both a simpler and a rich SV specification.

The diagnostic threshold is identical to the production acceptance threshold.
Timeout/cancellation is an infrastructure outcome, not a convergence result.

## 12. Robustness analyses

After the primary W=1000 experiment and before final manuscript conclusions:

- W=750;
- W=1250;
- cocoa with the three pre-flagged 2024 dates excluded;
- alternative ICCO cocoa data;
- WTI oil alternative;
- SV prior sensitivity;
- GARCH/EGARCH order sensitivity including (1,2) and (2,1) where applicable.

These checks may modify the strength of conclusions but must not be used to
select a replacement primary result after seeing W=1000.

## 13. Manuscript rule

The existing abstract, highlights, Results, Discussion and Conclusion are
legacy/provisional until the corrected primary and robustness experiments have
been completed and audited.  No numerical claim is protected.  The manuscript
must follow the corrected evidence even if every prior headline conclusion is
reversed.
