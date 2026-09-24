"""Synthetic parameter-recovery diagnostic for plain Gaussian SV leverage.

This isolates the leverage mechanism from heavy tails.  Data are generated from
exactly the Gaussian leverage timing used by the production model:

    r_t = exp(h_t / 2) * epsilon_t
    h_{t+1} = mu + phi * (h_t - mu) + sigma_eta * eta_t
    epsilon_t = rho * eta_t + sqrt(1-rho^2) * xi_t

with eta_t and xi_t independent standard normals.  The fitted model must satisfy
the unchanged production convergence gate; posterior coverage is reported as a
separate finite-sample recovery diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def simulate_sv_leverage(
    n: int,
    *,
    mu: float,
    phi: float,
    sigma_eta: float,
    rho: float,
    mean_return: float = 0.0,
    random_seed: int = 20260917,
) -> np.ndarray:
    if not (-1.0 < phi < 1.0):
        raise ValueError("phi must lie in (-1, 1)")
    if sigma_eta <= 0.0 or not (-1.0 < rho < 1.0):
        raise ValueError("invalid SV-Leverage parameters")

    rng = np.random.default_rng(random_seed)
    eta = rng.normal(size=n)
    xi = rng.normal(size=n)
    h = np.empty(n + 1, dtype=float)
    h[0] = mu + sigma_eta / np.sqrt(1.0 - phi**2) * rng.normal()
    returns = np.empty(n, dtype=float)
    orthogonal_scale = np.sqrt(1.0 - rho**2)

    for t in range(n):
        epsilon = rho * eta[t] + orthogonal_scale * xi[t]
        returns[t] = mean_return + np.exp(h[t] / 2.0) * epsilon
        h[t + 1] = mu + phi * (h[t] - mu) + sigma_eta * eta[t]
    return returns


def posterior_summary(fit: dict, truth: dict[str, float]) -> dict:
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
                float((mean - true_value) / sd) if sd > 0.0 else None
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=-0.50)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics"
    )
    args = parser.parse_args()

    truth = {
        "mu": -9.0,
        "phi": 0.97,
        "sigma_eta": 0.22,
        "rho": float(args.rho),
    }
    returns = simulate_sv_leverage(
        args.window,
        **truth,
        random_seed=args.seed,
    )
    fit = fit_sv_adaptive(
        returns,
        variant="SV-Leverage",
        random_seed=args.seed,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    summary = posterior_summary(fit, truth)

    payload = {
        "kind": "synthetic_sv_leverage_recovery",
        "variant": "SV-Leverage",
        "model_version": MODEL_VERSION,
        "window": args.window,
        "seed": args.seed,
        "truth": truth,
        "converged": bool(fit.get("converged", False)),
        "accepted_attempt": fit.get("accepted_attempt"),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "rhat_by_var": fit.get("rhat_by_var", {}),
        "ess_by_var": fit.get("ess_by_var", {}),
        "attempts": fit.get("mcmc_attempts", []),
        "error": fit.get("error"),
        "posterior_recovery": summary,
        "recovery_checks": {
            "rho_true_in_90pct": summary.get("rho", {}).get("contains_true_90pct"),
            "rho_true_in_95pct": summary.get("rho", {}).get("contains_true_95pct"),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "synthetic_sv_leverage_recovery.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))

    # Convergence is the hard gate. Coverage is reported separately because one
    # finite simulated sample need not cover every true value.
    if not payload["converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
