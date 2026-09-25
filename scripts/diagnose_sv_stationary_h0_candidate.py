"""High-persistence SV-t recovery diagnostic for the stationary-h0 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sv_model import DEFAULT_ROLLING_MCMC_ATTEMPTS, MODEL_VERSION, fit_sv_adaptive


def simulate_sv_t(
    n: int,
    *,
    mu: float,
    phi: float,
    sigma_eta: float,
    nu: float,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    h = np.empty(n + 1, dtype=float)
    h[0] = mu + sigma_eta / np.sqrt(1.0 - phi**2) * rng.normal()
    eta = rng.normal(size=n)
    eps = rng.standard_t(df=nu, size=n) * np.sqrt((nu - 2.0) / nu)
    out = np.empty(n, dtype=float)
    for t in range(n):
        out[t] = np.exp(h[t] / 2.0) * eps[t]
        h[t + 1] = mu + phi * (h[t] - mu) + sigma_eta * eta[t]
    return out


def posterior_summary(fit: dict, truth: dict[str, float]) -> dict:
    trace = fit.get("trace")
    if trace is None:
        return {}
    out = {}
    for name, true_value in truth.items():
        if name not in trace.posterior:
            continue
        values = trace.posterior[name].values.reshape(-1).astype(float)
        q025, q50, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        out[name] = {
            "true": float(true_value),
            "mean": float(values.mean()),
            "median": float(q50),
            "q025": float(q025),
            "q975": float(q975),
            "contains_true_95pct": bool(q025 <= true_value <= q975),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=20260925)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    truth = {
        "mu": -8.0,
        "phi": 0.995,
        "sigma_eta": 0.05,
        "nu": 5.0,
    }
    returns = simulate_sv_t(
        args.window,
        **truth,
        random_seed=args.random_seed,
    )
    fit = fit_sv_adaptive(
        returns,
        variant="SV-t",
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    payload = {
        "kind": "synthetic_high_persistence_sv_t",
        "model_version": MODEL_VERSION,
        "window": args.window,
        "target_accept": args.target_accept,
        "truth": truth,
        "converged": bool(fit.get("converged", False)),
        "accepted_attempt": fit.get("accepted_attempt"),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "posterior_recovery": posterior_summary(fit, truth),
        "attempts": fit.get("mcmc_attempts", []),
        "error": fit.get("error"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "stationary_h0_synthetic_recovery.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
