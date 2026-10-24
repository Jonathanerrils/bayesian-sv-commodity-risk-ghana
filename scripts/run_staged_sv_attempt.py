"""Single-stage SV MCMC command-line runner.

Runs one attempt for one block and writes either a rejected-attempt checkpoint
or a converged posterior artifact. It does NOT generate forecast CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_all_returns
from production_runner import COMMODITIES, SV_VARIANTS, DEFAULT_TARGET_ACCEPT
from staged_sv_attempt import run_attempt
from staged_posterior_artifact import save_posterior_artifact
from sv_model import DEFAULT_ROLLING_MCMC_ATTEMPTS


def main():
    parser = argparse.ArgumentParser(description="Execute exactly one SV MCMC attempt")
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--model", choices=SV_VARIANTS, required=True)
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--attempt", type=int, choices=[1, 2], required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.window < 2 or args.refit_every < 1 or args.start_block < 0:
        parser.error("invalid block/window configuration")
    if not 0 < args.target_accept < 1:
        parser.error("target accept must be in (0, 1)")
    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    block_start = args.start_block * args.refit_every
    training = returns[block_start:block_start + args.window]
    if len(training) != args.window or block_start + args.window >= len(returns):
        parser.error("block outside available forecast range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "failed-attempts.json"
    fit, outcome = run_attempt(
        training, variant=args.model, target_accept=args.target_accept,
        random_seed=args.random_seed + block_start,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS, attempt_no=args.attempt,
        checkpoint_path=checkpoint, provenance=args.provenance)
    if outcome == "complete":
        save_posterior_artifact(
            args.output_dir / "posterior", fit, training,
            variant=args.model, target_accept=args.target_accept,
            random_seed=args.random_seed + block_start,
            attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS, provenance=args.provenance)
    (args.output_dir / "stage-outcome.json").write_text(
        json.dumps({"outcome": outcome, "attempt": args.attempt,
                    "commodity": args.commodity, "model": args.model,
                    "block_id": args.start_block, "provenance": args.provenance}),
        encoding="utf-8")
    print(f"SV stage {args.attempt}: {outcome}")


if __name__ == "__main__":
    main()
