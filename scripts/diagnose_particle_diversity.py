"""Measure SV filter ESS and posterior-particle ancestry over one refit block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from production_runner import COMMODITIES, SV_VARIANTS
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    MODEL_VERSION,
    _transition_filter_state,
    fit_sv_adaptive,
    initialize_filter_state,
    update_filter_state,
)


def _unique_fraction(state: dict) -> float:
    particle_id = np.asarray(state["particle_id"])
    return float(np.unique(particle_id).size / particle_id.size)


def trace_filter_block(
    state: dict,
    actual_returns: np.ndarray,
    *,
    start_i: int,
    random_seed: int,
) -> pd.DataFrame:
    """Apply the production filter update sequence and record diversity only."""
    rows = [{
        "step": 0,
        "forecast_index": int(start_i),
        "filter_ess": np.nan,
        "particle_unique_fraction": _unique_fraction(state),
    }]
    current = state
    for offset, actual_return in enumerate(np.asarray(actual_returns, dtype=float)):
        global_i = int(start_i + offset)
        step_rng = np.random.default_rng(random_seed * 1_000_003 + global_i)
        transition = _transition_filter_state(current, step_rng)
        current, filter_ess = update_filter_state(
            transition, float(actual_return), step_rng
        )
        rows.append({
            "step": int(offset + 1),
            "forecast_index": global_i,
            "filter_ess": float(filter_ess),
            "particle_unique_fraction": _unique_fraction(current),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--variant", choices=SV_VARIANTS, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    start_i = args.block * args.refit_every
    train_end = start_i + args.window
    block_end = train_end + args.refit_every
    if start_i < 0 or block_end > len(returns):
        raise ValueError("requested block does not contain a complete refit/filter interval")

    train = returns[start_i:train_end]
    fit = fit_sv_adaptive(
        train,
        variant=args.variant,
        target_accept=args.target_accept,
        random_seed=args.random_seed + start_i,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    if not fit.get("converged", False):
        raise RuntimeError(
            "particle-diversity diagnostic requires a converged structural refit; "
            "the requested case failed the unchanged MCMC gate"
        )

    state = initialize_filter_state(fit)
    realised = returns[train_end:block_end]
    trajectory = trace_filter_block(
        state,
        realised,
        start_i=start_i,
        random_seed=args.random_seed,
    )

    slug = args.variant.lower().replace(" ", "-")
    stem = f"particle_diversity__{args.commodity}__{slug}__block-{args.block}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    trajectory.to_csv(csv_path, index=False)

    filtered = trajectory.iloc[1:]
    summary = {
        "commodity": args.commodity,
        "variant": args.variant,
        "block": args.block,
        "window": args.window,
        "refit_every": args.refit_every,
        "model_version": MODEL_VERSION,
        "accepted_attempt": fit.get("accepted_attempt"),
        "mcmc_max_rhat": fit.get("max_rhat"),
        "mcmc_min_ess": fit.get("min_ess"),
        "mcmc_divergences": fit.get("n_divergences"),
        "initial_particle_unique_fraction": float(
            trajectory.iloc[0]["particle_unique_fraction"]
        ),
        "final_particle_unique_fraction": float(
            trajectory.iloc[-1]["particle_unique_fraction"]
        ),
        "minimum_particle_unique_fraction": float(
            trajectory["particle_unique_fraction"].min()
        ),
        "minimum_filter_ess": float(filtered["filter_ess"].min()),
        "median_filter_ess": float(filtered["filter_ess"].median()),
        "n_filter_updates": int(len(filtered)),
        "csv_artifact": csv_path.name,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
