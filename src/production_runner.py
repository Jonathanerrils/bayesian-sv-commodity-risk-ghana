"""Production runner for rolling commodity-risk backtests.

Validity repair v2:
* CLI window/refit settings are threaded into every model call.
* checkpoints are configuration-scoped and cannot silently reuse legacy v1 CSVs.
* SV forecasts use daily latent-state filtering between MCMC parameter refits.
* rolling SV refits use a strict adaptive MCMC convergence policy.
* GARCH-family benchmarks cross Gaussian/Student-t innovations with symmetric
  GARCH and asymmetric EGARCH dynamics.
* primary model comparisons use the same valid forecast dates for every model
  within a commodity; native-sample diagnostics are retained separately.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtests import run_all_backtests
from data_utils import load_all_prices, load_all_returns
from garch_model import historical_simulation_var_es, rolling_var_es
from ou_model import rolling_ou_var_es
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    MODEL_VERSION,
    rolling_sv_var_es,
)

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


def mcmc_policy_label(attempts=None) -> str:
    """Compact deterministic label for the rolling MCMC escalation policy."""
    if attempts is None:
        attempts = DEFAULT_ROLLING_MCMC_ATTEMPTS
    return "-then-".join(
        f"{int(a['chains'])}c{int(a['tune'])}t{int(a['draws'])}d"
        for a in attempts
    )


def run_key(window: int, refit_every: int, predictive_draws: int) -> str:
    return (
        f"{MODEL_VERSION}_{mcmc_policy_label()}_w{window}_r{refit_every}_p{predictive_draws}"
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
        mcmc_attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    logger.info(
        "[DONE] %s/%s: %d forecasts in %.1f min",
        commodity,
        variant,
        len(df),
        (time.time() - t0) / 60,
    )
    return df


def _valid_forecast_mask(fc: pd.DataFrame, alphas=None) -> pd.Series:
    """Rows eligible for fair primary comparison for one model."""
    if alphas is None:
        alphas = ALPHAS
    mask = pd.Series(True, index=fc.index, dtype=bool)
    required = ["actual_return"]
    for alpha in alphas:
        required.extend([f"var_{alpha}", f"es_{alpha}"])
    for col in required:
        if col not in fc:
            return pd.Series(False, index=fc.index, dtype=bool)
        mask &= np.isfinite(pd.to_numeric(fc[col], errors="coerce"))
    if "estimation_failed" in fc:
        failed = fc["estimation_failed"].fillna(True).astype(bool)
        mask &= ~failed
    return mask


def common_valid_dates(forecasts: dict, commodity: str) -> pd.DatetimeIndex:
    """Intersection of valid forecast dates across all models for a commodity."""
    model_frames = [
        fc for (comm, _model), fc in forecasts.items() if comm == commodity
    ]
    if not model_frames:
        return pd.DatetimeIndex([])

    common = None
    for fc in model_frames:
        valid_idx = pd.DatetimeIndex(fc.index[_valid_forecast_mask(fc)])
        common = valid_idx if common is None else common.intersection(valid_idx)
    return pd.DatetimeIndex(common).sort_values()


def _backtest_rows(
    forecasts: dict,
    window: int,
    refit_every: int,
    predictive_draws: int,
    common_dates: bool,
) -> pd.DataFrame:
    rows = []
    date_cache = {
        commodity: common_valid_dates(forecasts, commodity)
        for commodity in sorted({c for c, _m in forecasts})
    }
    for (commodity, model), fc in forecasts.items():
        used = fc.loc[date_cache[commodity]] if common_dates else fc
        bt = run_all_backtests(used, ALPHAS, label=f"{commodity}/{model}")
        bt["commodity"] = commodity
        bt["model"] = model
        bt["sample"] = "common_dates" if common_dates else "native"
        bt["window"] = window
        bt["refit_every"] = refit_every if model in SV_VARIANTS else 1
        bt["predictive_draws"] = predictive_draws if model in SV_VARIANTS else 0
        bt["model_version"] = MODEL_VERSION
        bt["mcmc_policy"] = mcmc_policy_label() if model in SV_VARIANTS else "n/a"
        rows.append(bt)
    return pd.concat(rows, ignore_index=True)


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["commodity", "model"], sort=True)
    summary = grouped[["kupiec_passed", "cc_passed", "as_passed"]].sum()
    summary["primary_tests_passed"] = summary.sum(axis=1)
    summary["primary_tests_total"] = grouped.size() * 3
    summary["independence_passed"] = grouped["ind_passed"].sum()
    summary["independence_total"] = grouped.size()
    summary["n_obs_min"] = grouped["n_obs"].min()
    summary["n_obs_max"] = grouped["n_obs"].max()
    return summary.reset_index()


def compile_results(
    forecasts: dict,
    window: int,
    refit_every: int,
    predictive_draws: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    out_dir = TABLES_DIR / run_key(window, refit_every, predictive_draws)
    out_dir.mkdir(parents=True, exist_ok=True)

    native = _backtest_rows(
        forecasts, window, refit_every, predictive_draws, common_dates=False
    )
    primary = _backtest_rows(
        forecasts, window, refit_every, predictive_draws, common_dates=True
    )

    # Primary paper-facing table: identical dates across models.
    primary.to_csv(out_dir / "full_backtest_results.csv", index=False)
    primary.to_csv(out_dir / "full_backtest_results_common_dates.csv", index=False)
    native.to_csv(out_dir / "full_backtest_results_native.csv", index=False)

    _summary_table(primary).to_csv(out_dir / "summary_pass_counts.csv", index=False)
    _summary_table(primary).to_csv(
        out_dir / "summary_pass_counts_common_dates.csv", index=False
    )
    _summary_table(native).to_csv(
        out_dir / "summary_pass_counts_native.csv", index=False
    )

    for commodity in sorted({c for c, _m in forecasts}):
        n_common = len(common_valid_dates(forecasts, commodity))
        logger.info(
            "[COMMON-DATE] %s: %d dates retained across %d models",
            commodity,
            n_common,
            sum(1 for c, _m in forecasts if c == commodity),
        )
        if n_common == 0:
            raise RuntimeError(
                f"No common valid forecast dates remain for {commodity}; "
                "primary model comparison is undefined."
            )

    logger.info("Backtest tables saved under %s", out_dir)
    return primary


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
    logger.info("Rolling MCMC policy: %s", mcmc_policy_label())

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
