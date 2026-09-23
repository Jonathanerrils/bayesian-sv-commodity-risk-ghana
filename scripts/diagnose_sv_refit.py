"""Diagnose one production-scale SV refit and persist structural diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from production_runner import COMMODITIES, SV_VARIANTS
from sv_model import DEFAULT_ROLLING_MCMC_ATTEMPTS, MODEL_VERSION, fit_sv_adaptive


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--variant", choices=SV_VARIANTS, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--attempt",
        type=int,
        default=None,
        help=(
            "Run one numbered entry from DEFAULT_ROLLING_MCMC_ATTEMPTS instead "
            "of the full adaptive escalation. The attempt keeps the same seed "
            "it would have received inside fit_sv_adaptive."
        ),
    )
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help=(
            "Write diagnostics and exit successfully even when this individual "
            "attempt does not converge. Intended for split-attempt CI where a "
            "separate summary job enforces the unchanged convergence gate."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    start_i = args.block * args.refit_every
    if start_i < 0 or start_i + args.window > len(returns):
        raise ValueError("requested block does not have a complete fitting window")

    train = returns[start_i : start_i + args.window]
    base_seed = 42 + start_i

    if args.attempt is None:
        fit = fit_sv_adaptive(
            train,
            variant=args.variant,
            target_accept=args.target_accept,
            random_seed=base_seed,
            attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
        )
        requested_attempt = None
    else:
        if args.attempt < 1 or args.attempt > len(DEFAULT_ROLLING_MCMC_ATTEMPTS):
            raise ValueError(
                f"--attempt must be between 1 and {len(DEFAULT_ROLLING_MCMC_ATTEMPTS)}"
            )
        requested_attempt = int(args.attempt)
        config = DEFAULT_ROLLING_MCMC_ATTEMPTS[requested_attempt - 1]
        fit = fit_sv_adaptive(
            train,
            variant=args.variant,
            target_accept=args.target_accept,
            random_seed=base_seed + requested_attempt - 1,
            attempts=[config],
        )
        # A one-entry adaptive call labels its local attempt as 1. Remap the
        # diagnostic metadata to the production escalation number so split jobs
        # can be recombined without ambiguity.
        attempts = []
        for record in fit.get("mcmc_attempts", []):
            record = dict(record)
            record["attempt"] = requested_attempt
            attempts.append(record)
        fit["mcmc_attempts"] = attempts
        fit["accepted_attempt"] = requested_attempt if fit.get("converged", False) else None

    payload = {
        "commodity": args.commodity,
        "variant": args.variant,
        "block": args.block,
        "global_start_i": start_i,
        "window": args.window,
        "refit_every": args.refit_every,
        "model_version": MODEL_VERSION,
        "target_accept": args.target_accept,
        "requested_attempt": requested_attempt,
        "converged": bool(fit.get("converged", False)),
        "accepted_attempt": fit.get("accepted_attempt"),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "rhat_by_var": fit.get("rhat_by_var", {}),
        "ess_by_var": fit.get("ess_by_var", {}),
        "attempts": fit.get("mcmc_attempts", []),
        "error": fit.get("error"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.variant.lower().replace(" ", "-")
    attempt_suffix = "" if requested_attempt is None else f"__attempt-{requested_attempt}"
    path = (
        args.output_dir
        / f"sv_refit__{args.commodity}__{slug}__block-{args.block}{attempt_suffix}.json"
    )
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))

    if not payload["converged"] and not args.allow_nonconverged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
