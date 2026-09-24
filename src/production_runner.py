"""Production runner for scientifically audited rolling commodity-risk backtests."""

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

from backtests import apply_bonferroni_reporting, run_all_backtests
from checkpoint_safety import parse_bool_series
from data_utils import load_all_prices, load_all_returns
from garch_model import historical_simulation_var_es, rolling_var_es
from ou_model import OU_MODEL_VERSION, rolling_ou_var_es
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    MODEL_VERSION,
    PRIMARY_SV_VARIANTS,
    rolling_sv_var_es,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v2"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "v2"

PIPELINE_VERSION = "risk-pipeline-v5-primary-symmetric-sv-checkpoint-safe"
ALPHAS = [0.01, 0.05]
COMMODITIES = ["cocoa", "gold", "oil"]
DEFAULT_TARGET_ACCEPT = 0.95
SV_VARIANTS = list(PRIMARY_SV_VARIANTS)
GARCH_BENCHMARK_SPECS = {
    "GARCH": ("GARCH", "normal"),
    "GARCH-t": ("GARCH", "t"),
    "EGARCH": ("EGARCH", "normal"),
    "EGARCH-t": ("EGARCH", "t"),
}
BENCHMARK_MODELS = [*GARCH_BENCHMARK_SPECS, "OU", "HistSim"]
ALL_MODELS = [*BENCHMARK_MODELS, *SV_VARIANTS]

# Frozen v5 multiplicity families: 8 models x 3 commodities x 2 alphas.
PER_TEST_BONFERRONI_FAMILY_SIZE = len(ALL_MODELS) * len(COMMODITIES) * len(ALPHAS)
GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE = PER_TEST_BONFERRONI_FAMILY_SIZE * 3
BONFERRONI_FAMILY_SIZE = PER_TEST_BONFERRONI_FAMILY_SIZE


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"production_runner.{path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def mcmc_policy_label(attempts=None) -> str:
    attempts = DEFAULT_ROLLING_MCMC_ATTEMPTS if attempts is None else attempts
    return "-then-".join(
        f"{int(a['chains'])}c{int(a['tune'])}t{int(a['draws'])}d" for a in attempts
    )


def _target_accept_label(target_accept: float) -> str:
    return f"{float(target_accept):.6g}".replace(".", "p")


def run_key(
    window: int,
    refit_every: int,
    predictive_draws: int,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
) -> str:
    return (
        f"{PIPELINE_VERSION}_{MODEL_VERSION}_{OU_MODEL_VERSION}_"
        f"{mcmc_policy_label()}_w{window}_r{refit_every}_p{predictive_draws}_"
        f"ta{_target_accept_label(target_accept)}"
        .replace("/", "-")
        .replace(" ", "-")
    )


def checkpoint_path(
    commodity, model, window, refit_every, predictive_draws,
    target_accept=DEFAULT_TARGET_ACCEPT,
) -> Path:
    root = CHECKPOINT_ROOT / run_key(
        window, refit_every, predictive_draws, target_accept=target_accept
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{commodity}_{model.replace(' ', '_')}.csv"


def load_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def run_benchmark(
    commodity, model, returns, prices, window, refit_every, predictive_draws,
    logger, target_accept=DEFAULT_TARGET_ACCEPT,
):
    path = checkpoint_path(
        commodity, model, window, refit_every, predictive_draws, target_accept
    )
    if path.exists():
        existing = load_checkpoint(path)
        if len(existing) == len(returns) - window:
            logger.info("[SKIP] %s/%s: validated checkpoint", commodity, model)
            return existing
        logger.warning("[STALE] %s has wrong row count; recomputing", path)

    t0 = time.time()
    if model in GARCH_BENCHMARK_SPECS:
        model_type, distribution = GARCH_BENCHMARK_SPECS[model]
        df = rolling_var_es(
            returns, model_type=model_type, distribution=distribution,
            window=window, alphas=ALPHAS,
        )
    elif model == "HistSim":
        df = historical_simulation_var_es(returns, window=window, alphas=ALPHAS)
    elif model == "OU":
        df = rolling_ou_var_es(prices, returns, window=window, alphas=ALPHAS)
    else:
        raise ValueError(f"Unknown benchmark: {model}")
    df.to_csv(path)
    logger.info("[DONE] %s/%s: %d forecasts in %.1f min", commodity, model, len(df), (time.time() - t0) / 60)
    return df


def run_sv(
    commodity, variant, returns, window, refit_every, predictive_draws,
    logger, target_accept=DEFAULT_TARGET_ACCEPT,
):
    if variant not in SV_VARIANTS:
        raise ValueError(f"{variant!r} is not in the frozen v5 primary SV family")
    path = checkpoint_path(
        commodity, variant, window, refit_every, predictive_draws, target_accept
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
        target_accept=target_accept,
        mcmc_attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    logger.info("[DONE] %s/%s: %d forecasts in %.1f min", commodity, variant, len(df), (time.time() - t0) / 60)
    return df


def _valid_forecast_mask(fc: pd.DataFrame, alphas=None) -> pd.Series:
    alphas = ALPHAS if alphas is None else alphas
    mask = pd.Series(True, index=fc.index, dtype=bool)
    required = ["actual_return"]
    for alpha in alphas:
        required.extend([f"var_{alpha}", f"es_{alpha}"])
    for col in required:
        if col not in fc:
            return pd.Series(False, index=fc.index, dtype=bool)
        mask &= np.isfinite(pd.to_numeric(fc[col], errors="coerce"))
    if "estimation_failed" in fc:
        mask &= ~parse_bool_series(fc["estimation_failed"], missing=True)
    return mask


def common_valid_dates(forecasts: dict, commodity: str) -> pd.DatetimeIndex:
    frames = [fc for (comm, _model), fc in forecasts.items() if comm == commodity]
    if not frames:
        return pd.DatetimeIndex([])
    common = None
    for fc in frames:
        idx = pd.DatetimeIndex(fc.index[_valid_forecast_mask(fc)])
        common = idx if common is None else common.intersection(idx)
    return pd.DatetimeIndex(common).sort_values()


def _backtest_rows(
    forecasts, window, refit_every, predictive_draws, common_dates,
    target_accept=DEFAULT_TARGET_ACCEPT,
):
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
        bt["target_accept"] = target_accept if model in SV_VARIANTS else np.nan
        bt["pipeline_version"] = PIPELINE_VERSION
        bt["model_version"] = MODEL_VERSION if model in SV_VARIANTS else "n/a"
        bt["ou_model_version"] = OU_MODEL_VERSION if model == "OU" else "n/a"
        bt["mcmc_policy"] = mcmc_policy_label() if model in SV_VARIANTS else "n/a"
        rows.append(bt)
    return pd.concat(rows, ignore_index=True)


def _add_dual_bonferroni(results: pd.DataFrame) -> pd.DataFrame:
    """Report both pre-declared multiplicity families without choosing post hoc."""
    per_test = apply_bonferroni_reporting(results, PER_TEST_BONFERRONI_FAMILY_SIZE)
    global_primary = apply_bonferroni_reporting(results, GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE)
    out = per_test.copy()
    corrected_cols = [
        c for c in global_primary.columns
        if c.endswith("_bonferroni")
        or c in {"bonferroni_family_size", "bonferroni_alpha", "as_bonferroni_decision", "as_bonferroni_decision_stable"}
    ]
    for col in corrected_cols:
        out[f"{col}_global_primary"] = global_primary[col].values
    out["multiplicity_per_test_family_size"] = PER_TEST_BONFERRONI_FAMILY_SIZE
    out["multiplicity_global_primary_family_size"] = GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE
    return out


def _summary_table(results: pd.DataFrame, scheme: str = "raw") -> pd.DataFrame:
    grouped = results.groupby(["commodity", "model"], sort=True)
    if scheme == "raw":
        cols = ["kupiec_passed", "cc_passed", "as_passed"]
    elif scheme == "per_test":
        cols = ["kupiec_passed_bonferroni", "cc_passed_bonferroni", "as_passed_bonferroni"]
    elif scheme == "global_primary":
        cols = [
            "kupiec_passed_bonferroni_global_primary",
            "cc_passed_bonferroni_global_primary",
            "as_passed_bonferroni_global_primary",
        ]
    else:
        raise ValueError(f"Unknown summary scheme: {scheme}")
    summary = grouped[cols].sum()
    summary["primary_nonrejections"] = summary.sum(axis=1)
    summary["primary_tests_total"] = grouped.size() * 3
    summary["independence_nonrejections"] = grouped["ind_passed"].sum()
    summary["independence_total"] = grouped.size()
    summary["n_obs_min"] = grouped["n_obs"].min()
    summary["n_obs_max"] = grouped["n_obs"].max()
    summary["scheme"] = scheme
    return summary.reset_index()


def compile_results(
    forecasts, window, refit_every, predictive_draws, logger,
    target_accept=DEFAULT_TARGET_ACCEPT,
) -> pd.DataFrame:
    out_dir = TABLES_DIR / run_key(
        window, refit_every, predictive_draws, target_accept=target_accept
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    native = _add_dual_bonferroni(
        _backtest_rows(
            forecasts, window, refit_every, predictive_draws,
            common_dates=False, target_accept=target_accept,
        )
    )
    primary = _add_dual_bonferroni(
        _backtest_rows(
            forecasts, window, refit_every, predictive_draws,
            common_dates=True, target_accept=target_accept,
        )
    )

    primary.to_csv(out_dir / "full_backtest_results.csv", index=False)
    primary.to_csv(out_dir / "full_backtest_results_common_dates.csv", index=False)
    native.to_csv(out_dir / "full_backtest_results_native.csv", index=False)
    _summary_table(primary, "raw").to_csv(out_dir / "summary_nonrejections.csv", index=False)
    _summary_table(native, "raw").to_csv(out_dir / "summary_nonrejections_native.csv", index=False)
    _summary_table(primary, "per_test").to_csv(out_dir / "summary_nonrejections_bonferroni_per_test.csv", index=False)
    _summary_table(native, "per_test").to_csv(out_dir / "summary_nonrejections_native_bonferroni_per_test.csv", index=False)
    _summary_table(primary, "global_primary").to_csv(out_dir / "summary_nonrejections_bonferroni_global_primary.csv", index=False)
    _summary_table(native, "global_primary").to_csv(out_dir / "summary_nonrejections_native_bonferroni_global_primary.csv", index=False)

    for commodity in sorted({c for c, _m in forecasts}):
        n_common = len(common_valid_dates(forecasts, commodity))
        logger.info(
            "[COMMON-DATE] %s: %d dates retained across %d models",
            commodity, n_common, sum(1 for c, _m in forecasts if c == commodity),
        )
        if n_common == 0:
            raise RuntimeError(
                f"No common valid forecast dates remain for {commodity}; common-date comparison is undefined."
            )

    logger.info(
        "Multiplicity families: per-test m=%d (alpha=%.8f); global-primary m=%d (alpha=%.8f)",
        PER_TEST_BONFERRONI_FAMILY_SIZE,
        0.05 / PER_TEST_BONFERRONI_FAMILY_SIZE,
        GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE,
        0.05 / GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE,
    )
    logger.info("Non-rejection is not evidence of adequacy; availability is reported separately.")
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
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    args = parser.parse_args()

    if args.sv_only and args.benchmark_only:
        parser.error("--sv-only and --benchmark-only are mutually exclusive")
    if args.window < 100:
        parser.error("--window must be at least 100 observations")
    if args.refit_every < 1:
        parser.error("--refit-every must be >= 1")
    if args.predictive_draws < 2_000:
        parser.error("--predictive-draws must be >= 2000")
    if not (0.0 < args.target_accept < 1.0):
        parser.error("--target-accept must lie strictly between 0 and 1")

    commodities = COMMODITIES if args.commodity == "all" else [args.commodity]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    logger.info(
        "Run config: commodities=%s window=%d refit=%d predictive=%d target_accept=%.6g pipeline=%s sv=%s ou=%s",
        commodities, args.window, args.refit_every, args.predictive_draws,
        args.target_accept, PIPELINE_VERSION, MODEL_VERSION, OU_MODEL_VERSION,
    )
    logger.info("Rolling MCMC policy: %s", mcmc_policy_label())

    returns_all = load_all_returns(verbose=False)
    prices_all = load_all_prices()
    forecasts = {}
    for commodity in commodities:
        returns = returns_all[commodity]
        prices = prices_all[commodity]
        if not args.sv_only:
            for model in BENCHMARK_MODELS:
                forecasts[(commodity, model)] = run_benchmark(
                    commodity, model, returns, prices,
                    args.window, args.refit_every, args.predictive_draws, logger,
                    target_accept=args.target_accept,
                )
        if not args.benchmark_only:
            for variant in SV_VARIANTS:
                forecasts[(commodity, variant)] = run_sv(
                    commodity, variant, returns,
                    args.window, args.refit_every, args.predictive_draws, logger,
                    target_accept=args.target_accept,
                )
    if forecasts:
        compile_results(
            forecasts, args.window, args.refit_every, args.predictive_draws, logger,
            target_accept=args.target_accept,
        )


if __name__ == "__main__":
    main()
