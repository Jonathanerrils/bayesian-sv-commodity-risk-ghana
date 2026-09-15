"""Focused identifiability diagnostic for the combined heavy-tail/leverage SV model.

This script answers two separate questions before the full W=1000 production run:

1. On the exact oil window where SV-t-Leverage failed, do the nested SV-t and
   SV-Leverage specifications converge under the same production MCMC policy?
2. Can the full SV-t-Leverage specification recover known rho/nu values from a
   synthetic T=1000 sample generated from its own data-generating process?

The diagnostic does not relax the production R-hat/ESS/divergence gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    MODEL_VERSION,
    fit_sv_adaptive,
)


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


def _fit_payload(fit: dict) -> dict:
    return {
        "model_version": MODEL_VERSION,
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


def _posterior_summary(fit: dict, truth: dict[str, float]) -> dict:
    trace = fit.get("trace")
    if trace is None:
        return {}
    out = {}
    for name, true_value in truth.items():
        if name not in trace.posterior:
            continue
        values = trace.posterior[name].values.reshape(-1).astype(float)
        q025, q05, q50, q95, q975 = np.quantile(
            values, [0.025, 0.05, 0.5, 0.95, 0.975]
        )
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        out[name] = {
            "true": float(true_value),
            "mean": mean,
            "sd": sd,
            "median": float(q50),
            "q025": float(q025),
            "q05": float(q05),
            "q95": float(q95),
            "q975": float(q975),
            "contains_true_90pct": bool(q05 <= true_value <= q95),
            "contains_true_95pct": bool(q025 <= true_value <= q975),
            "standardized_mean_error": (
                float((mean - true_value) / sd) if sd > 0 else None
            ),
        }
    return out


def simulate_sv_t_leverage(
    n: int,
    *,
    mu: float,
    phi: float,
    sigma_eta: float,
    rho: float,
    nu: float,
    mean_return: float = 0.0,
    random_seed: int = 20260915,
) -> np.ndarray:
    """Simulate exactly the SV-t-Leverage specification implemented in sv_model."""
    if not (-1.0 < phi < 1.0):
        raise ValueError("phi must lie in (-1, 1)")
    if sigma_eta <= 0 or nu <= 2 or not (-1.0 < rho < 1.0):
        raise ValueError("invalid SV-t-Leverage parameters")

    rng = np.random.default_rng(random_seed)
    eta = rng.normal(size=n)
    xi = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2.0) / nu)
    h = np.empty(n + 1, dtype=float)
    h[0] = mu + sigma_eta / np.sqrt(1.0 - phi**2) * rng.normal()
    returns = np.empty(n, dtype=float)
    orthogonal_scale = np.sqrt(1.0 - rho**2)

    for t in range(n):
        eps = rho * eta[t] + orthogonal_scale * xi[t]
        returns[t] = mean_return + np.exp(h[t] / 2.0) * eps
        h[t + 1] = mu + phi * (h[t] - mu) + sigma_eta * eta[t]
    return returns


def run_real_oil(variant: str, window: int, block: int, refit_every: int) -> dict:
    returns = load_all_returns(verbose=False)["oil"].to_numpy(dtype=float)
    start_i = block * refit_every
    train = returns[start_i : start_i + window]
    if len(train) != window:
        raise ValueError("requested oil block does not contain a full fitting window")
    fit = fit_sv_adaptive(
        train,
        variant=variant,
        random_seed=42 + start_i,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    return {
        "kind": "real_oil_nested",
        "commodity": "oil",
        "variant": variant,
        "block": block,
        "global_start_i": start_i,
        "window": window,
        **_fit_payload(fit),
    }


def run_synthetic(window: int) -> dict:
    truth = {
        "mu": -9.0,
        "phi": 0.97,
        "sigma_eta": 0.22,
        "rho": -0.50,
        "nu": 8.0,
    }
    returns = simulate_sv_t_leverage(window, **truth)
    fit = fit_sv_adaptive(
        returns,
        variant="SV-t-Leverage",
        random_seed=20260915,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    summary = _posterior_summary(fit, truth)
    recovery = {
        "rho_true_in_95pct": summary.get("rho", {}).get("contains_true_95pct"),
        "nu_true_in_95pct": summary.get("nu", {}).get("contains_true_95pct"),
    }
    return {
        "kind": "synthetic_recovery",
        "variant": "SV-t-Leverage",
        "window": window,
        "truth": truth,
        **_fit_payload(fit),
        "posterior_recovery": summary,
        "recovery_checks": recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["oil-sv-t", "oil-sv-leverage", "synthetic-sv-t-leverage"],
        required=True,
    )
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    if args.case == "oil-sv-t":
        payload = run_real_oil("SV-t", args.window, args.block, args.refit_every)
    elif args.case == "oil-sv-leverage":
        payload = run_real_oil("SV-Leverage", args.window, args.block, args.refit_every)
    else:
        payload = run_synthetic(args.window)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"identifiability__{args.case}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))

    # The MCMC gate is the hard diagnostic gate.  Synthetic coverage is retained
    # as a separate recovery diagnostic rather than used as an automatic CI
    # failure because one finite simulated sample need not cover every true value.
    if not payload["converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
