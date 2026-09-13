# Bayesian Latent Stochastic Volatility Models for Commodity Price Risk in West Africa

Code, data, and manuscript for a comparison of Bayesian stochastic-volatility models with Gaussian GARCH, Student-t GARCH, asymmetric Gaussian EGARCH, asymmetric Student-t EGARCH, Ornstein-Uhlenbeck, and Historical Simulation benchmarks for one-day Value-at-Risk (VaR) and Expected Shortfall (ES) forecasting on cocoa, gold, and Brent crude oil.

## Audit status — September 2026

An independent code audit identified validity problems in the original (`v1`) rolling SV results. The July 2026 checkpoint CSVs, tables, figures, and manuscript results are therefore **legacy/provisional artifacts and must not be treated as confirmed results** until the corrected pipeline has been rerun.

The main issues were:

1. SV structural parameters were refitted every 42 trading days, but the latent volatility state was not updated between refits, producing mechanically flat VaR/ES blocks.
2. `rho` was estimated in the leverage variants but omitted from one-step predictive simulation.
3. Student-t innovations were not variance-standardized and the prior did not actually enforce `nu > 2`.
4. the Acerbi-Szekely Test 2 implementation used mean ES rather than each day's `ES_t` forecast.
5. the EGARCH benchmark omitted the asymmetric `o=1` term described in the paper.
6. `--window` was logged but not threaded into production model calls, and checkpoint names did not identify the run configuration.
7. the benchmark set gave Student-t tails to SV models but only Gaussian innovations to GARCH-family models, confounding latent-volatility gains with innovation-distribution gains.

The `audit-fix/sv-validity-repair` line of work addresses these issues. Corrected results should be written under `checkpoints/v2/` and `outputs/v2/`; the existing top-level checkpoint/result files are retained only as an audit trail.

## Corrected v2 design

The repaired SV runner separates **parameter learning** from **state filtering**. MCMC still re-estimates structural parameters every 42 trading days by default, but after every one-day forecast the newly observed return is used to filter the latent volatility state before the next forecast. Leverage is propagated through the predictive shock, Student-t innovations are constrained to finite variance and standardized to unit variance, and each VaR/ES pair is estimated from a shared posterior predictive sample.

The benchmark family now crosses volatility dynamics and innovation laws: GARCH-Normal, GARCH-t, asymmetric EGARCH-Normal, and asymmetric EGARCH-t. This makes comparisons with SV-t materially fairer because heavy-tailed innovations are no longer exclusive to the SV family.

Partial SV runs now save a filter-state sidecar together with the CSV checkpoint. If the two become inconsistent, the runner refuses an inexact resume rather than silently changing the refit schedule.

## Repository structure

```text
paper/          Manuscript source and the legacy compiled paper
src/            Data loading, models, filtering, backtests, production runner
data/raw/       Raw commodity price and validation series
checkpoints/    Legacy v1 forecasts plus configuration-scoped v2 forecasts
outputs/        Legacy tables plus configuration-scoped v2 tables
tests/          Regression/unit tests for the validity fixes
.github/        CI workflow
```

## Environment

The v2 repair deliberately pins PyMC 5 because PyMC 5 is incompatible with ArviZ 1.x. Install with:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## Running corrected forecasts

Benchmark models only:

```bash
cd src
python production_runner.py --benchmark-only --window 1000
```

One commodity's SV models:

```bash
python production_runner.py --commodity gold --sv-only \
  --window 1000 --refit-every 42 --predictive-draws 20000
```

Window sensitivity now creates separate checkpoint namespaces, so for example:

```bash
python production_runner.py --window 750
python production_runner.py --window 1000
python production_runner.py --window 1250
```

cannot silently reuse one another's forecasts.

The GARCH implementation also accepts alternative `(p, q)` orders directly through `rolling_var_es(..., p=..., q=...)`, with stationarity checks applied across all fitted ARCH/GARCH coefficients. This supports the planned GARCH(1,2) and GARCH(2,1) robustness runs.

## Robustness checks still required

The pre-specified robustness work remains outstanding until rerun on the repaired pipeline: rolling-window sensitivity (750/1,000/1,250 days), exclusion of the three flagged 2024 cocoa dates, ICCO in place of the primary cocoa series, WTI in place of Brent, prior sensitivity, and GARCH-order sensitivity. These checks should be completed before the manuscript's headline conclusions are restored.

## Author

Jonathan, independent researcher, Ghana.
