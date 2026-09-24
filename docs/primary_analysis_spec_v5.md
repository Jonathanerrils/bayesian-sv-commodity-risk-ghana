# Corrected primary analysis specification v5

Status: **frozen before corrected full W=1000 primary results are inspected**.

This specification supersedes v4 for the primary production run. It incorporates the completed leverage-model diagnostic programme and does not preserve any legacy model merely to reproduce the July 2026 manuscript. If corrected results disagree with the existing paper, the paper must change.

## 1. Data and evaluation calendar

- Primary commodities: cocoa, gold and Brent crude oil.
- Primary sample end: 2026-06-22 under the repository's committed cleaning rules.
- Daily log returns.
- Primary rolling estimation window: **W=1000 retained returns**.
- Every model remains on the same fixed target-date calendar for a commodity.
- Estimation failures remain explicit; failed dates are never silently removed.

## 2. Frozen primary model family

The primary comparison contains **8 models**.

### Stochastic-volatility models

1. `SV-Gaussian`
2. `SV-t`

Both use

    r_t = m + exp(h_t/2) * epsilon_t
    h_{t+1} = mu + phi (h_t - mu) + sigma_eta * eta_t

with |phi| < 1 and sigma_eta > 0. The latent path is innovation-noncentred with a stationary Gaussian initial state. A fit to T observed returns contains h_0,...,h_T, so h_T is the next forecast-date state.

`SV-Gaussian` uses epsilon_t ~ N(0,1). `SV-t` uses a Student-t innovation with nu > 2, scaled to unit variance. In both primary SV models epsilon_t and eta_t are independent.

### Benchmark models

3. GARCH(1,1), Gaussian innovations
4. GARCH(1,1), standardized Student-t innovations
5. asymmetric EGARCH(1,1), Gaussian innovations
6. asymmetric EGARCH(1,1), standardized Student-t innovations
7. exact-transition Ornstein-Uhlenbeck model on log prices
8. 1000-observation historical simulation

## 3. Leverage-family decision

Leverage-based SV models are **excluded from the primary family** before corrected primary-result inspection.

The decision follows the pre-production diagnostic programme rather than model ranking:

- the difficult oil W=1000 `SV-t` fit passed the strict convergence gate;
- plain `SV-Leverage` on the same oil window failed because leverage inference mixed inadequately;
- the coherent bivariate-Student-t leverage candidate failed the oil gate and also failed the strict synthetic gate;
- a specialist ASIS leverage reference sampler recovered synthetic rho reasonably but still failed the strict synthetic mixing gate and mixed substantially worse on oil;
- convergence thresholds were never relaxed.

Leverage implementations and diagnostic scripts remain in the repository only for audit reproduction and sensitivity work. They are not eligible for primary production, common-date comparison, multiplicity counting, or headline model claims.

## 4. SV priors and MCMC gate

- mu ~ Normal(-10, 3)
- phi = 2*phi_raw - 1, phi_raw ~ Beta(20, 1.5)
- sigma_eta ~ Half-Cauchy(0.5)
- nu = 2 + Exponential(0.1) for `SV-t`

Adaptive scheduled refits:

1. 4 chains, 1000 tune + 1000 retained draws per chain;
2. if needed, 4 chains, 2000 tune + 2000 retained draws per chain.

A refit is accepted only when all are true:

- max structural R-hat < 1.01;
- minimum structural bulk ESS > 400;
- zero divergences.

A weak trace is never used for forecasting.

## 5. Rolling forecast/filter schedule

- Structural refits at global forecast indices 0, 42, 84, ... .
- Between successful refits, posterior particles are filtered after every realised return.
- Forecast from h_t first, then condition on r_t and advance to h_{t+1}.
- 20,000 posterior predictive draws per forecast date.
- Deterministic random seeds tied to global forecast index.

If a scheduled refit fails after both MCMC attempts, the complete 42-day block is unavailable. No next-day retry is permitted. The next attempt occurs at the next scheduled boundary.

## 6. Checkpoint/resume safety

Primary production uses versioned fail-closed SV checkpoints.

- State sidecars contain an explicit checkpoint schema version and exact SV model version.
- Missing, malformed, stale, or mismatched sidecars are rejected rather than guessed compatible.
- CSV Boolean fields are parsed explicitly; strings such as `"False"` can never become true through generic Python truthiness.
- CSV and state-sidecar positions must match exactly for resume.
- Old unversioned partial SV checkpoints require a clean restart.

The model/pipeline version change in v5 deliberately separates all new production output from legacy checkpoints.

## 7. OU benchmark

The OU process

    dX_t = kappa(theta-X_t)dt + sigma dW_t

uses the exact Gaussian transition likelihood and exact one-step return variance. Price transition pairs are aligned to the retained return transitions. Primary time is trading time, dt=1/252 per retained transition. Small positive kappa is not treated as estimation failure merely because mean reversion is slow.

## 8. VaR and ES convention

- VaR and ES are positive loss magnitudes.
- A VaR violation occurs when realised return < -VaR.
- ES must be at least VaR in the corresponding loss tail.
- Primary alpha levels: 0.01 and 0.05.

## 9. Primary backtests

At each alpha:

1. Kupiec unconditional coverage;
2. Christoffersen conditional coverage;
3. Acerbi-Szekely Test 2 for Expected Shortfall.

Christoffersen independence remains a separate diagnostic, not a fourth primary test.

Acerbi-Szekely Test 2 uses observation-specific ES forecasts, full numerical precision, an add-one Monte Carlo p-value, and Monte Carlo uncertainty. No adjusted decision is forced when Monte Carlo uncertainty overlaps the corrected threshold. Non-rejection is not described as proof of adequacy.

## 10. Frozen multiplicity families

The v5 primary family contains 8 models x 3 commodities x 2 alpha levels.

### Per-test family

8 x 3 x 2 = **m=48** for each named primary test. Corrected alpha = 0.05/48.

### Global-primary family

8 x 3 x 2 x 3 primary tests = **m=144**. Corrected alpha = 0.05/144.

Both schemes and raw p-values are reported regardless of which is more favourable.

## 11. Missing forecasts and comparison samples

Report all three:

1. availability on the full fixed calendar;
2. native-valid-date backtests for each model;
3. common-date comparisons using the intersection of valid dates across the full **8-model primary family** for that commodity.

Common-date results must never be presented without availability because intersection filtering can conceal model instability.

## 12. Primary production gate

Before launching the complete W=1000 experiment:

- ordinary CI must pass;
- checkpoint/resume boundary tests must pass;
- the production matrix must contain only `SV-Gaussian` and `SV-t` as SV variants;
- representative W=1000 diagnostics must confirm the retained SV families on cocoa, gold and oil, including the previously difficult cocoa Gaussian case and the validated oil Student-t case;
- no leverage production job may be present.

Timeout or cancellation is an infrastructure outcome, not statistical nonconvergence.

## 13. Robustness programme

After the frozen W=1000 primary run:

- W=750;
- W=1250;
- cocoa excluding the three pre-flagged 2024 dates;
- alternative ICCO cocoa data;
- WTI oil alternative;
- SV prior sensitivity;
- GARCH/EGARCH order sensitivity including (1,2) and (2,1) where applicable;
- leverage-family diagnostics may be reported separately as sensitivity/audit evidence, but cannot replace a primary model after results are seen.

## 14. Manuscript rule

The existing abstract, highlights, Results, Discussion and Conclusion remain legacy/provisional until the corrected primary and robustness experiments are completed and audited. No numerical claim is protected. The manuscript follows the corrected evidence even if prior headline conclusions are reversed.
