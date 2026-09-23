"""
production_runner.py

Full rolling-window production run across all models and all commodities.
Implements Deliverable 2 exactly as specified, with:

  - Checkpointing: saves after every MCMC refit -- a crash loses at most
    one refit window (21 days), not the entire run
  - Progress logging: timestamped log file so you can monitor without
    watching the terminal
  - Incremental saves: forecast DataFrames written to CSV as they accumulate
  - Resume capability: detects existing checkpoint and skips already-done steps

Runtime estimates (based on actual sandbox timing of 40.6s/refit):
  Modern laptop (~4x faster): ~7-8 hours total for all models + commodities
  Budget laptop (~2x faster): ~15 hours total
  Run overnight. Do not interrupt mid-commodity if avoidable.

Usage:
  python production_runner.py                  # full run, all models
  python production_runner.py --commodity gold  # one commodity only
  python production_runner.py --sv-only         # SV variants only (skip GARCH/OU)
  python production_runner.py --benchmark-only  # GARCH/EGARCH/HS/OU only (fast)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_all_returns, load_all_prices
from garch_model import (rolling_var_es,
                                       historical_simulation_var_es)
from ou_model import rolling_ou_var_es
from sv_model import rolling_sv_var_es
from backtests import run_all_backtests

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR    = PROJECT_ROOT / "outputs"
BACKTESTS_DIR  = RESULTS_DIR / "model_results"
TABLES_DIR     = RESULTS_DIR / "tables"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for d in [RESULTS_DIR, BACKTESTS_DIR, TABLES_DIR, CHECKPOINT_DIR]:
    d.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# CONFIGURATION (Deliverable 2)
# -----------------------------------------------------------------------
WINDOW        = 1000   # primary rolling window
REFIT_EVERY   = 42     # MCMC refit frequency for SV (trading days)
ALPHAS        = [0.01, 0.05]
COMMODITIES   = ["cocoa", "gold", "oil"]

SV_VARIANTS   = ["SV-Gaussian", "SV-t", "SV-Leverage", "SV-t-Leverage"]
BENCHMARK_MODELS = ["GARCH", "EGARCH", "OU", "HistSim"]

# -----------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("production_runner")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# -----------------------------------------------------------------------
# CHECKPOINT HELPERS
# -----------------------------------------------------------------------
def legacy_checkpoint_path(commodity: str, model: str) -> Path:
    """Return the pre-configuration checkpoint path used by published runs."""
    return CHECKPOINT_DIR / f"{commodity}_{model.replace(' ', '_')}.csv"


def checkpoint_path(commodity: str, model: str, window: int,
                    refit_every: int | None = None) -> Path:
    """Return a checkpoint path that uniquely identifies the run configuration."""
    safe_model = model.replace(" ", "_")
    parts = [commodity, safe_model, f"w{window}"]
    if refit_every is not None:
        parts.append(f"refit{refit_every}")
    return CHECKPOINT_DIR / ("_".join(parts) + ".csv")


def existing_checkpoint_path(commodity: str, model: str, window: int,
                             refit_every: int | None = None) -> Path | None:
    """Find a compatible checkpoint without mixing different configurations."""
    configured = checkpoint_path(commodity, model, window, refit_every)
    if configured.exists():
        return configured

    # Backward compatibility is deliberately restricted to the historical
    # primary specification. Non-default robustness runs must never reuse
    # the old untagged W=1000 / refit=42 checkpoints.
    legacy_allowed = (
        window == WINDOW
        and (refit_every is None or refit_every == REFIT_EVERY)
    )
    legacy = legacy_checkpoint_path(commodity, model)
    if legacy_allowed and legacy.exists():
        return legacy

    return None


def load_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def save_checkpoint(df: pd.DataFrame, path: Path):
    df.to_csv(path)


# -----------------------------------------------------------------------
# BENCHMARK RUNNER (GARCH, EGARCH, OU, HistSim)
# -----------------------------------------------------------------------
def run_benchmark(commodity: str, model: str,
                  returns: pd.Series, prices: pd.Series,
                  logger: logging.Logger, window: int) -> pd.DataFrame:

    existing_path = existing_checkpoint_path(commodity, model, window)
    if existing_path is not None:
        logger.info(f"  [SKIP] {commodity}/{model}: compatible checkpoint "
                    f"found ({existing_path.name}), loading")
        return load_checkpoint(existing_path)

    logger.info(f"  [START] {commodity}/{model} (window={window})")
    t0 = time.time()

    if model == "GARCH":
        df = rolling_var_es(returns, "GARCH", window=window, alphas=ALPHAS)
    elif model == "EGARCH":
        df = rolling_var_es(returns, "EGARCH", window=window, alphas=ALPHAS)
    elif model == "HistSim":
        df = historical_simulation_var_es(returns, window=window, alphas=ALPHAS)
    elif model == "OU":
        df = rolling_ou_var_es(prices, returns, window=window,
                               alphas=ALPHAS)
    else:
        raise ValueError(f"Unknown benchmark model: {model}")

    elapsed = time.time() - t0
    logger.info(f"  [DONE] {commodity}/{model}: "
                f"{len(df)} forecasts in {elapsed/60:.1f} min")
    save_checkpoint(df, checkpoint_path(commodity, model, window))
    return df


# -----------------------------------------------------------------------
# SV RUNNER
# -----------------------------------------------------------------------
def run_sv(commodity: str, variant: str,
           returns: pd.Series,
           logger: logging.Logger, window: int,
           refit_every: int = REFIT_EVERY) -> pd.DataFrame:

    configured_path = checkpoint_path(commodity, variant, window, refit_every)
    existing_path = existing_checkpoint_path(
        commodity, variant, window, refit_every
    )
    ckpt_path = existing_path if existing_path is not None else configured_path
    expected_steps = len(returns) - window

    if existing_path is not None:
        try:
            existing = pd.read_csv(existing_path, parse_dates=["date"])
            if len(existing) >= expected_steps:
                logger.info(f"  [SKIP] {commodity}/{variant}: complete "
                           f"compatible checkpoint found ({len(existing)} steps, "
                           f"{existing_path.name}), loading")
                return load_checkpoint(existing_path)
            else:
                logger.info(f"  [RESUME] {commodity}/{variant}: partial "
                           f"compatible checkpoint found "
                           f"({len(existing)}/{expected_steps} steps, "
                           f"{existing_path.name}) -- resuming")
        except Exception:
            logger.warning(f"  [WARN] {commodity}/{variant}: existing "
                          f"checkpoint unreadable, starting fresh")
            ckpt_path = configured_path

    logger.info(f"  [START] {commodity}/{variant} "
                f"(window={window}, refit_every={refit_every})")
    t0 = time.time()

    df = rolling_sv_var_es(returns, variant=variant, window=window,
                           refit_every=refit_every, alphas=ALPHAS,
                           checkpoint_path=ckpt_path, checkpoint_every=50)

    elapsed = time.time() - t0
    logger.info(f"  [DONE] {commodity}/{variant}: "
                f"{len(df)} forecasts in {elapsed/60:.1f} min")
    return df


# -----------------------------------------------------------------------
# BACKTEST + RESULTS TABLE
# -----------------------------------------------------------------------
def compile_results(all_forecasts: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Run all backtest statistics and compile the full results table.
    all_forecasts: dict of {(commodity, model): forecast_df}
    """
    logger.info("Compiling backtest results...")
    all_rows = []

    for (commodity, model), fc_df in all_forecasts.items():
        n_nan = fc_df[[c for c in fc_df.columns if c.startswith("var_")]]\
                .isna().any(axis=1).sum()
        if n_nan > 0:
            logger.warning(f"  {commodity}/{model}: {n_nan} NaN forecast rows")

        bt = run_all_backtests(fc_df, alphas=ALPHAS, label=f"{commodity}/{model}")
        bt["commodity"] = commodity
        bt["model"]     = model
        all_rows.append(bt)

    results = pd.concat(all_rows, ignore_index=True)

    # Save full results table
    out_path = TABLES_DIR / "full_backtest_results.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Full results saved to {out_path}")

    # Summary table: pass/fail counts per model
    summary = results.groupby(["commodity", "model"])\
                     [["kupiec_passed", "cc_passed", "ind_passed", "as_passed"]]\
                     .sum().reset_index()
    summary["total_tests"] = results.groupby(["commodity", "model"])["alpha"]\
                                    .count().values * 4
    summary_path = TABLES_DIR / "summary_pass_counts.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary pass counts saved to {summary_path}")

    return results


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Production backtest runner")
    parser.add_argument("--commodity", choices=COMMODITIES + ["all"],
                        default="all")
    parser.add_argument("--sv-only",        action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--window",     type=int, default=WINDOW)
    args = parser.parse_args()

    run_sv_models   = not args.benchmark_only
    run_benchmarks  = not args.sv_only
    commodities     = COMMODITIES if args.commodity == "all" \
                      else [args.commodity]
    window          = args.window

    log_path = PROJECT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger   = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("PRODUCTION RUN STARTED")
    logger.info(f"  Commodities:   {commodities}")
    logger.info(f"  Window:        {window}")
    logger.info(f"  Benchmarks:    {run_benchmarks}")
    logger.info(f"  SV models:     {run_sv_models}")
    logger.info(f"  Refit every:   {REFIT_EVERY} days (SV only)")
    logger.info(f"  Log file:      {log_path}")
    logger.info("=" * 60)

    # Load data once
    logger.info("Loading and cleaning data...")
    all_returns = load_all_returns(verbose=False)
    # Load raw prices for OU (needs price levels, not returns).
    # Uses load_all_prices() from data_utils.py, which correctly handles
    # the three different raw file formats (yfinance multi-header for
    # cocoa/gold, headerless for Brent) -- do not re-implement file
    # reading here, it was a real source of bugs before this fix.
    raw_prices = load_all_prices()

    all_forecasts = {}
    run_start = time.time()

    for commodity in commodities:
        ret    = all_returns[commodity]
        prices = raw_prices[commodity]
        # Align prices index with returns index
        prices = prices.reindex(ret.index).ffill()

        logger.info(f"\n{'='*40}")
        logger.info(f"COMMODITY: {commodity.upper()}")
        logger.info(f"  Returns: {len(ret)} obs, "
                    f"{ret.index.min().date()} to {ret.index.max().date()}")
        logger.info(f"  Expected out-of-sample steps: {len(ret) - window}")

        # --- Benchmarks ---
        if run_benchmarks:
            for model in BENCHMARK_MODELS:
                try:
                    fc = run_benchmark(commodity, model, ret, prices, logger,
                                       window=window)
                    all_forecasts[(commodity, model)] = fc
                except Exception as e:
                    logger.error(f"  [FAILED] {commodity}/{model}: {e}")

        # --- SV variants ---
        if run_sv_models:
            for variant in SV_VARIANTS:
                try:
                    fc = run_sv(commodity, variant, ret, logger,
                                window=window, refit_every=REFIT_EVERY)
                    all_forecasts[(commodity, variant)] = fc
                except Exception as e:
                    logger.error(f"  [FAILED] {commodity}/{variant}: {e}")

    # --- Compile results ---
    if all_forecasts:
        results = compile_results(all_forecasts, logger)
        logger.info("\n=== BACKTEST SUMMARY ===")
        logger.info(results[["commodity", "model", "alpha",
                              "kupiec_passed", "cc_passed",
                              "as_passed"]].to_string(index=False))

    total_elapsed = (time.time() - run_start) / 3600
    logger.info(f"\nTotal runtime: {total_elapsed:.2f} hours")
    logger.info("PRODUCTION RUN COMPLETE")


if __name__ == "__main__":
    main()
