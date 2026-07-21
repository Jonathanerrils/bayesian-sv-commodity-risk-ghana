# Bayesian Latent Stochastic Volatility Models for Commodity Price Risk in West Africa

Code, data pipeline, and paper for a study comparing Bayesian latent stochastic volatility models against standard benchmarks (GARCH, EGARCH, Ornstein-Uhlenbeck, Historical Simulation) for Value-at-Risk and Expected Shortfall forecasting on cocoa, gold, and Brent crude oil — three commodities central to Ghana's economy.

The core finding: no single model wins everywhere. Student-t stochastic volatility models achieve a clean sweep of every backtest for both cocoa and gold, with a leverage extension offering a further, smaller improvement for cocoa specifically. Oil resists every specification tested, including the richest one, at the 99% VaR confidence level. The full reasoning is in the paper.

## Repository structure

```
paper/          Full paper (LaTeX source + compiled PDF), Elsevier elsarticle format
src/            Model estimation and backtesting code
data/raw/       Raw commodity price series (cocoa, gold, oil) plus validation series
checkpoints/    Saved rolling-window forecast results per model per commodity
outputs/        Backtest result tables and figures
notebooks/      Data cleaning, exploratory analysis, and stylised facts (Jupyter)
docs/           Data dictionary, cleaning policy, and supporting documentation
```

## Reproducing this

Requirements are in `requirements.txt`. The one dependency worth knowing about ahead of time: fitting the Bayesian SV models via NUTS MCMC is slow — each commodity's full rolling backtest across four SV variants takes somewhere between 10 and 25 hours on a normal laptop, depending on which regime-change periods fall inside the sample. GARCH, EGARCH, and OU are fast by comparison (under 30 minutes for all three commodities together).

To run the benchmark models only:

```bash
cd src
python production_runner.py --benchmark-only
```

To run one commodity's SV models:

```bash
python production_runner.py --commodity gold --sv-only
```

The SV runner checkpoints progress every 50 steps, so an interrupted run resumes from the last checkpoint rather than starting over — this matters in practice, since a run this long will get interrupted by something eventually.

## Data sources and known limitations

Full provenance for every series, including the specific data-quality issues found and how they were handled, is in `docs/data_dictionary.md`. Briefly: cocoa and gold prices come from Yahoo Finance continuous futures, cross-validated against ICCO and Stooq respectively; oil uses FRED's Brent spot series as primary, since Brent — not WTI — is the internationally relevant benchmark for West African oil exposure, and the two turned out to behave as genuinely different risk processes during the April 2020 negative-price episode.

## Status of results in this repository

All 24 checkpoint files are now present: all three commodities (cocoa, gold, oil), all eight models (four benchmarks — GARCH, EGARCH, OU, Historical Simulation — and four SV variants), each a complete rolling-window forecast series. Every checkpoint was independently spot-checked before being added here: row counts match the expected out-of-sample step count exactly, OU non-convergence rates match the paper's stated 4.3%/5.1%/10.7% figures for cocoa/oil/gold respectively, and a backtest recomputed directly from the raw cocoa SV-t-Leverage checkpoint reproduces the paper's reported 36 violations at 99% VaR exactly.

The compiled backtest result tables in `outputs/tables/` are currently split by commodity (produced from separate `--commodity X --sv-only` runs); a single combined 48-row file has not yet been generated in one pass, though it can be reproduced directly from the 24 checkpoints above using `production_runner.py`'s compilation step.

A real correction is worth noting explicitly: an earlier draft of this paper had several cells wrong in the summary table, including one case where the Kupiec and Christoffersen test results were reversed for oil's best-performing model. This was caught by cross-checking the paper's claims directly against the aggregated results CSVs rather than against console screenshots, and is exactly the kind of error this repository's structure — raw checkpoints alongside aggregated tables alongside the paper's claims — is meant to make possible to catch.

## Six pre-committed robustness checks, not yet run

Rolling window length sensitivity (750/1,000/1,250 days), cocoa's three flagged 2024 rollover-illiquidity dates excluded, cocoa using ICCO's price series instead of Yahoo, oil using WTI instead of Brent, prior sensitivity on the persistence parameter, and a GARCH order check against (1,2) and (2,1). These are stated in the paper as outstanding, not silently assumed complete.

## Author

Jonathan, independent researcher, Ghana.
