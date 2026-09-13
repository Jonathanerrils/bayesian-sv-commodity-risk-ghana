"""Production runner for rolling commodity-risk backtests.

Validity repair v2:
* CLI window/refit settings are threaded into every model call.
* checkpoints are configuration-scoped and cannot silently reuse legacy v1 CSVs.
* SV forecasts use daily latent-state filtering between MCMC parameter refits.
* GARCH-family benchmarks cross Gaussian/Student-t innovations with symmetric
  GARCH and asymmetric EGARCH dynamics.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtests import run_all_backtests
from data_utils import load_all_prices, load_all_returns
from garch_model import historical_simulation_var_es, rolling_var_es
from ou_model import rolling_ou_var_es
from sv_model import MODEL_VERSION, rolling_sv_var_es

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v2"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "v2"

ALPHAS = [0.01, 0.05]
COMMODITIES = ["cocoa", "gold", "oil"]
SV_VARIANTS = ["SV-Gaussian", "SV-t", "SV-Leverage", "SV-t-Leverage"]
GARCH_BENCHMARK_SPECS = {
    "GARCH": ("GARCH", "normal"),
    "GARCH-t": ("GARCH", "t"),
    "EGARCH": ("EGARCH", "normal"),
    "EGARCH-t": ("EGARCH", "t"),
}
BENCHMARK_MODELS = [*GARCH_BENCHMARK_SPECS, "OU", "HistSim"]


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"production_runner.{path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def run_key(window: int, refit_every: int, predictive_draws: int) -> str:
    return (
        f"{MODEL_VERSION}_w{window}_r{refit_every}_p{predictive_draws}"
        .replace("/", "-")
        .replace(" ", "-")
    )


def checkpoint_path(
    commodity: str,
    model: str,
    window: int,
    refit_every: int,
    predictive_draws: int,
) -> Path:
    root = CHECKPOINT_ROOT / run_key(window, refit_every, predictive_draws)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{commodity}_{model.replace(' ', '_')}.csv"


def load_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def run_benchmark(
    commodity: str,
    model: str,
    returns: pd.Series,
    prices: pd.Series,
    window: int,
    refit_every: int,
    predictive_draws: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    path = checkpoint_path(
        commodity, model, window, refit_every, predictive_draws
    )
    if path.exists():
        existing = load_checkpoint(path)
        expected = len(returns) - window
        if len(existing) == expected:
            logger.info("[SKIP] %s/%s: validated checkpoint", commodity, model)
            return existing
        logger.warning("[STALE] %s has wrong row count; recomputing", path)

    t0 = time.time()
    if model in GARCH_BENCHMARK_SPECS:
        model_type, distribution = GARCH_BENCHMARK_SPECS[model]
        df = rolling_var_es(
            returns,
            model_type=model_type,
            distribution=distribution,
            window=window,
            alphas=ALPHAS,
        )
    elif model == "HistSim":
        df = historical_simulation_var_es(returns, window=window, alphas=ALPHAS)
    elif model == "OU":
        df = rolling_ou_var_es(prices, returns, window=window, alphas=ALPHAS)
    else:
        raise ValueError(f"Unknown benchmark: {model}")

    df.to_csv(path)
    logger.info(
        "[DONE] %s/%s: %d forecasts in %.1f min",
        commodity,
        model,
        len(df),
        (time.time() - t0) / 60,
    )
    return df


def run_sv(
    commodity: str,
    variant: str,
    returns: pd.Series,
    window: int,
    refit_every: int,
    predictive_draws: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    path = checkpoint_path(
        commodity, variant, window, refit_every, predictive_draws
    )
    expected = len(returns) - window
    if path.exists():
        existing = load_checkpoint(path)
        if len(existing) == expected:
            logger.info("[SKIP] %s/%s: validated checkpoint", commodity, variant)
            return existing

    t0 = time.time()
    df = rolling_sv_var_es(
        returns,
        variant=variant,
        window=window,
        refit_every=refit_every,
        alphas=ALPHAS,
        checkpoint_path=path,
        checkpoint_every=50,
        n_predictive=predictive_draws,
    )
    logger.info(
        "[DONE] %s/%s: %d forecasts in %.1f min",
        commodity,
        variant,
        len(df),
        (time.time() - t0) / 60,
    )
    return df


def compile_results(
    forecasts: dict,
    window: int,
    refit_every: int,
    predictive_draws: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    for (commodity, model), fc in forecasts.items():
        bt = run_all_backtests(fc, ALPHAS, label=f"{commodity}/{model}")
        bt["commodity"] = commodity
        bt["model"] = model
        bt["window"] = window
        bt["refit_every"] = refit_every if model in SV_VARIANTS else 1
        bt["predictive_draws"] = predictive_draws if model in SV_VARIANTS else 0
        bt["model_version"] = MODEL_VERSION
        rows.append(bt)

    results = pd.concat(rows, ignore_index=True)
    out_dir = TABLES_DIR / run_key(window, refit_every, predictive_draws)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "full_backtest_results.csv", index=False)

    # Primary pass count = Kupiec + conditional coverage + ES Test 2.
    # Christoffersen independence remains a separately reported diagnostic.
    grouped = results.groupby(["commodity", "model"], sort=True)
    summary = grouped[["kupiec_passed", "cc_passed", "as_passed"]].sum()
    summary["primary_tests_passed"] = summary.sum(axis=1)
    summary["primary_tests_total"] = grouped.size() * 3
    summary["independence_passed"] = grouped["ind_passed"].sum()
    summary["independence_total"] = grouped.size()
    summary = summary.reset_index()
    summary.to_csv(out_dir / "summary_pass_counts.csv", index=False)
    logger.info("Backtest tables saved under %s", out_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Commodity-risk production runner")
    parser.add_argument("--commodity", choices=COMMODITIES + ["all"], default="all")
    parser.add_argument("--sv-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--predictive-draws", type=int, default=20_000)
    args = parser.parse_args()

    if args.sv_only and args.benchmark_only:
        parser.error("--sv-only and --benchmark-only are mutually exclusive")
    if args.window < 100:
        parser.error("--window must be at least 100 observations")
    if args.refit_every < 1:
        parser.error("--refit-every must be >= 1")
    if args.predictive_draws < 2_000:
        parser.error("--predictive-draws must be >= 2000")

    commodities = COMMODITIES if args.commodity == "all" else [args.commodity]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    logger.info(
        "Run config: commodities=%s window=%d refit=%d predictive=%d version=%s",
        commodities,
        args.window,
        args.refit_every,
        args.predictive_draws,
        MODEL_VERSION,
    )

    returns_all = load_all_returns(verbose=False)
    prices_all = load_all_prices()
    forecasts = {}

    for commodity in commodities:
        returns = returns_all[commodity]
        prices = prices_all[commodity].reindex(returns.index).ffill()

        if not args.sv_only:
            for model in BENCHMARK_MODELS:
                forecasts[(commodity, model)] = run_benchmark(
                    commodity,
                    model,
                    returns,
                    prices,
                    args.window,
                    args.refit_every,
                    args.predictive_draws,
                    logger,
                )

        if not args.benchmark_only:
            for variant in SV_VARIANTS:
                forecasts[(commodity, variant)] = run_sv(
                    commodity,
                    variant,
                    returns,
                    args.window,
                    args.refit_every,
                    args.predictive_draws,
                    logger,
                )

    if forecasts:
        compile_results(
            forecasts,
            args.window,
            args.refit_every,
            args.predictive_draws,
            logger,
        )


if __name__ == "__main__":
    main()
