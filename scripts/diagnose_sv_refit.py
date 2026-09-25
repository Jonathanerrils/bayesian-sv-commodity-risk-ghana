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




def _resolve_attempt(attempt: int | None, base_seed: int):
    """Return the production-equivalent attempt list and seed for one split run."""
    if attempt is None:
        return DEFAULT_ROLLING_MCMC_ATTEMPTS, int(base_seed), None
    if attempt < 1 or attempt > len(DEFAULT_ROLLING_MCMC_ATTEMPTS):
        raise ValueError(
            f"attempt must be between 1 and {len(DEFAULT_ROLLING_MCMC_ATTEMPTS)}"
        )
    config = dict(DEFAULT_ROLLING_MCMC_ATTEMPTS[attempt - 1])
    # fit_sv_adaptive numbers a supplied one-entry attempt list from 1, so shift
    # the base seed here to match the seed used by the normal adaptive routine.
    return [config], int(base_seed + attempt - 1), int(attempt)


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(q50),
        "q05": float(q05),
        "q95": float(q95),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _divergence_geometry(fit: dict) -> dict:
    """Summarize where NUTS divergences occur without altering the fit."""
    trace = fit.get("trace")
    if trace is None or "diverging" not in trace.sample_stats:
        return {
            "available": False,
            "n_divergent": 0,
            "n_total": 0,
            "divergence_rate": None,
            "metrics": {},
        }

    divergent = np.asarray(trace.sample_stats["diverging"].values, dtype=bool)
    if divergent.ndim != 2:
        raise ValueError("expected diverging sample-stat array with chain x draw dimensions")
    nondivergent = ~divergent

    metrics = {}

    def add_metric(name: str, values) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.shape[:2] != divergent.shape:
            return
        if arr.ndim > 2:
            return
        metrics[name] = {
            "divergent": _distribution_summary(arr[divergent]),
            "nondivergent": _distribution_summary(arr[nondivergent]),
        }

    for name in (
        "mu",
        "phi",
        "phi_raw",
        "sigma_eta",
        "nu",
        "nu_minus_two",
        "h0_std",
        "h0",
    ):
        if name in trace.posterior:
            add_metric(name, trace.posterior[name].values)

    if "eta" in trace.posterior:
        eta = np.asarray(trace.posterior["eta"].values, dtype=float)
        if eta.shape[:2] == divergent.shape and eta.ndim >= 3:
            reduce_axes = tuple(range(2, eta.ndim))
            add_metric("eta_max_abs", np.max(np.abs(eta), axis=reduce_axes))
            add_metric("eta_rms", np.sqrt(np.mean(eta**2, axis=reduce_axes)))

    if "h" in trace.posterior:
        h = np.asarray(trace.posterior["h"].values, dtype=float)
        if h.shape[:2] == divergent.shape and h.ndim >= 3:
            reduce_axes = tuple(range(2, h.ndim))
            add_metric("h_min", np.min(h, axis=reduce_axes))
            add_metric("h_max", np.max(h, axis=reduce_axes))
            add_metric(
                "h_range",
                np.max(h, axis=reduce_axes) - np.min(h, axis=reduce_axes),
            )
            add_metric("h_last", h[..., -1])

    for name in ("energy_error", "max_energy_error", "tree_depth", "step_size"):
        if name in trace.sample_stats:
            add_metric(name, trace.sample_stats[name].values)

    n_total = int(divergent.size)
    n_divergent = int(divergent.sum())
    return {
        "available": True,
        "n_divergent": n_divergent,
        "n_total": n_total,
        "divergence_rate": float(n_divergent / n_total) if n_total else None,
        "metrics": metrics,
    }


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
        help="Run one numbered production escalation attempt instead of both.",
    )
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Persist diagnostics and exit successfully after a statistical failure.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    start_i = args.block * args.refit_every
    if start_i < 0 or start_i + args.window > len(returns):
        raise ValueError("requested block does not have a complete fitting window")

    train = returns[start_i : start_i + args.window]
    base_seed = 42 + start_i
    attempts, fit_seed, requested_attempt = _resolve_attempt(args.attempt, base_seed)
    fit = fit_sv_adaptive(
        train,
        variant=args.variant,
        target_accept=args.target_accept,
        random_seed=fit_seed,
        attempts=attempts,
    )
    if requested_attempt is not None:
        records = []
        for record in fit.get("mcmc_attempts", []):
            record = dict(record)
            record["attempt"] = requested_attempt
            records.append(record)
        fit["mcmc_attempts"] = records
        fit["accepted_attempt"] = (
            requested_attempt if fit.get("converged", False) else None
        )

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
        "production_base_seed": int(base_seed),
        "fit_seed": int(fit_seed),
        "converged": bool(fit.get("converged", False)),
        "accepted_attempt": fit.get("accepted_attempt"),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "rhat_by_var": fit.get("rhat_by_var", {}),
        "ess_by_var": fit.get("ess_by_var", {}),
        "attempts": fit.get("mcmc_attempts", []),
        "divergence_geometry": _divergence_geometry(fit),
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
