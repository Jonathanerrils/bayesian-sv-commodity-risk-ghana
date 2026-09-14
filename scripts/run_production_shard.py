"""Run one restartable shard of the corrected W=1000 production experiment.

SV shards operate on whole scheduled refit blocks.  A block starts at a global
forecast index that is an exact multiple of ``refit_every``; it performs the
same adaptive MCMC fit and uses the same per-day random seeds as the canonical
``rolling_sv_var_es`` implementation.  Because the canonical runner re-fits at
those boundaries, successful blocks are conditionally independent and may be
computed in parallel without changing the forecast sequence.

A scheduled SV refit that fails the strict convergence gate is a hard shard
failure.  We do not silently drop that block from the primary experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_prices, load_all_returns
from garch_model import historical_simulation_var_es, rolling_var_es
from ou_model import rolling_ou_var_es
from production_runner import (
    ALPHAS,
    BENCHMARK_MODELS,
    COMMODITIES,
    GARCH_BENCHMARK_SPECS,
    SV_VARIANTS,
)
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    _predictive_returns,
    _transition_filter_state,
    fit_sv_adaptive,
    initialize_filter_state,
    predictive_var_es,
    update_filter_state,
)

DEFAULT_WINDOW = 1000
DEFAULT_REFIT_EVERY = 42
DEFAULT_PREDICTIVE_DRAWS = 20_000
DEFAULT_RANDOM_SEED = 42


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-").replace("_", "-")


def _write_failure(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"failure__{payload.get('commodity', 'unknown')}__"
        f"{_slug(payload.get('model', 'unknown'))}__"
        f"block-{payload.get('block_id', 'na')}.json"
    )
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_benchmark_shard(
    commodity: str,
    model: str,
    window: int,
    output_dir: Path,
) -> Path:
    returns_all = load_all_returns(verbose=False)
    prices_all = load_all_prices()
    returns = returns_all[commodity]
    prices = prices_all[commodity].reindex(returns.index).ffill()

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
        raise ValueError(f"Unknown benchmark model: {model}")

    out = df.reset_index()
    out["global_i"] = np.arange(len(out), dtype=int)
    out["commodity"] = commodity
    out["model"] = model
    out["shard_kind"] = "benchmark"
    out["block_id"] = -1

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark__{commodity}__{_slug(model)}.csv"
    out.to_csv(path, index=False)
    return path


def _accepted_attempt(fit: dict) -> dict:
    accepted = fit.get("accepted_attempt")
    for record in fit.get("mcmc_attempts", []):
        if record.get("attempt") == accepted:
            return record
    return {}


def run_sv_shard(
    commodity: str,
    variant: str,
    start_block: int,
    n_blocks: int,
    window: int,
    refit_every: int,
    predictive_draws: int,
    output_dir: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    returns = load_all_returns(verbose=False)[commodity]
    ret_array = returns.to_numpy(dtype=float)
    dates = returns.index
    n_forecasts = len(ret_array) - window
    if n_forecasts <= 0:
        raise ValueError(f"{commodity}: series length must exceed window={window}")

    total_blocks = math.ceil(n_forecasts / refit_every)
    if start_block < 0 or start_block >= total_blocks:
        raise ValueError(
            f"start_block={start_block} outside [0, {total_blocks - 1}] for {commodity}"
        )
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")

    rows: list[dict] = []
    stop_block = min(start_block + n_blocks, total_blocks)

    for block_id in range(start_block, stop_block):
        block_start = block_id * refit_every
        block_end = min(block_start + refit_every, n_forecasts)
        train = ret_array[block_start : block_start + window]

        fit = fit_sv_adaptive(
            train,
            variant=variant,
            target_accept=0.95,
            random_seed=random_seed + block_start,
            attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
        )
        if not fit.get("converged", False):
            failure = {
                "commodity": commodity,
                "model": variant,
                "block_id": block_id,
                "global_start_i": block_start,
                "forecast_date": str(dates[block_start + window]),
                "max_rhat": fit.get("max_rhat"),
                "min_ess": fit.get("min_ess"),
                "n_divergences": fit.get("n_divergences"),
                "attempts": fit.get("mcmc_attempts", []),
                "error": fit.get("error"),
            }
            _write_failure(output_dir, failure)
            raise RuntimeError(
                f"Strict MCMC convergence failed for {commodity}/{variant} "
                f"at scheduled block {block_id} (i={block_start})"
            )

        accepted = _accepted_attempt(fit)
        filter_state = initialize_filter_state(fit)
        attempts_json = json.dumps(fit.get("mcmc_attempts", []), default=str)

        for global_i in range(block_start, block_end):
            actual_return = float(ret_array[global_i + window])
            forecast_date = dates[global_i + window]
            step_rng = np.random.default_rng(random_seed * 1_000_003 + global_i)

            transition = _transition_filter_state(filter_state, step_rng)
            r_pred = _predictive_returns(transition, predictive_draws, step_rng)

            row = {
                "date": forecast_date,
                "actual_return": actual_return,
                "estimation_failed": False,
                "refit": global_i == block_start,
                "filter_ess": np.nan,
                "global_i": global_i,
                "block_id": block_id,
                "commodity": commodity,
                "model": variant,
                "shard_kind": "sv",
                "mcmc_attempt": fit.get("accepted_attempt", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_max_rhat": fit.get("max_rhat", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_min_ess": fit.get("min_ess", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_divergences": fit.get("n_divergences", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_chains": accepted.get("chains", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_tune": accepted.get("tune", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_draws": accepted.get("draws", np.nan)
                if global_i == block_start
                else np.nan,
                "mcmc_attempts_json": attempts_json
                if global_i == block_start
                else "",
            }
            for alpha in ALPHAS:
                var, es = predictive_var_es(r_pred, alpha)
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"] = es

            # The forecast is generated before conditioning on its realised return.
            filter_state, filter_ess = update_filter_state(
                transition, actual_return, step_rng
            )
            row["filter_ess"] = filter_ess
            rows.append(row)

        # Persist completed blocks inside the shard so a diagnostic artifact can
        # still show how far the job got if a later block fails.
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = output_dir / (
            f"sv__{commodity}__{_slug(variant)}__blocks-"
            f"{start_block:04d}-{stop_block - 1:04d}.csv"
        )
        pd.DataFrame(rows).to_csv(partial, index=False)

    return partial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one distributed production shard")
    parser.add_argument("--kind", choices=["benchmark", "sv"], required=True)
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--refit-every", type=int, default=DEFAULT_REFIT_EVERY)
    parser.add_argument(
        "--predictive-draws", type=int, default=DEFAULT_PREDICTIVE_DRAWS
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "production_shards"
    )
    args = parser.parse_args()

    if args.kind == "benchmark":
        if args.model not in BENCHMARK_MODELS:
            parser.error(f"{args.model!r} is not a benchmark model")
        path = run_benchmark_shard(
            args.commodity, args.model, args.window, args.output_dir
        )
    else:
        if args.model not in SV_VARIANTS:
            parser.error(f"{args.model!r} is not an SV variant")
        path = run_sv_shard(
            args.commodity,
            args.model,
            args.start_block,
            args.n_blocks,
            args.window,
            args.refit_every,
            args.predictive_draws,
            args.output_dir,
        )

    print(f"Wrote production shard: {path}")


if __name__ == "__main__":
    main()
