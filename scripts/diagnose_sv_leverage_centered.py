"""Centered-parameterization diagnostic for Gaussian SV with leverage.

This keeps the same probability model, priors, forward leverage timing, and
production convergence gate as the current SV-Leverage specification.  Only the
latent-state parameterization changes: h_0,...,h_T are sampled directly and the
standardized state innovations eta_t are reconstructed from the AR(1) path.

The purpose is to determine whether the rho-mixing failure seen under the
innovation-noncentred parameterization is representation-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from sv_model import DEFAULT_ROLLING_MCMC_ATTEMPTS, MIN_STRUCTURAL_ESS, RHAT_THRESHOLD

CANDIDATE_VERSION = "sv-leverage-centered-diagnostic-v1"


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
    random_seed: int = 20260917,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    eta = rng.normal(size=n)
    xi = rng.normal(size=n)
    h = np.empty(n + 1)
    h[0] = mu + sigma_eta / np.sqrt(1.0 - phi**2) * rng.normal()
    x = np.empty(n)
    orth = np.sqrt(1.0 - rho**2)
    for t in range(n):
        eps = rho * eta[t] + orth * xi[t]
        x[t] = np.exp(h[t] / 2.0) * eps
        h[t + 1] = mu + phi * (h[t] - mu) + sigma_eta * eta[t]
    return x


def build_centered_model(returns: np.ndarray) -> pm.Model:
    T = len(returns)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10.0, sigma=3.0)
        phi_raw = pm.Beta("phi_raw", alpha=20.0, beta=1.5)
        phi = pm.Deterministic("phi", 2.0 * phi_raw - 1.0)
        sigma_eta = pm.HalfCauchy("sigma_eta", beta=0.5)
        rho = pm.Uniform("rho", lower=-1.0, upper=1.0)

        # Directly sample the latent log-volatility path.  A Flat carrier plus
        # the AR(1) potential gives the exact centered state density without an
        # additional approximation or change of prior.
        h = pm.Flat("h", shape=T + 1)
        stationary_sd = sigma_eta / pt.sqrt(pt.clip(1.0 - phi**2, 1e-8, np.inf))
        h0_logp = pm.logp(pm.Normal.dist(mu=mu, sigma=stationary_sd), h[0])
        state_mu = mu + phi * (h[:-1] - mu)
        transition_logp = pm.logp(
            pm.Normal.dist(mu=state_mu, sigma=sigma_eta), h[1:]
        ).sum()
        pm.Potential("state_logp", h0_logp + transition_logp)

        eta = pm.Deterministic("eta", (h[1:] - state_mu) / sigma_eta)
        vol = pt.exp(h[:-1] / 2.0)
        obs_mu = rho * vol * eta
        obs_sd = pt.sqrt(pt.clip(1.0 - rho**2, 1e-9, 1.0)) * vol
        pm.Normal("obs", mu=obs_mu, sigma=obs_sd, observed=returns)
    return model


def diagnostics(trace) -> dict:
    vars_ = ["mu", "phi", "sigma_eta", "rho"]
    rhat = az.rhat(trace, var_names=vars_)
    ess = az.ess(trace, var_names=vars_, method="bulk")
    rhat_by_var = {
        name: float(np.nanmax(np.asarray(rhat[name].values, dtype=float)))
        for name in vars_
    }
    ess_by_var = {
        name: float(np.nanmin(np.asarray(ess[name].values, dtype=float)))
        for name in vars_
    }
    max_rhat = max(rhat_by_var.values())
    min_ess = min(ess_by_var.values())
    n_divergences = int(trace.sample_stats["diverging"].sum().values)
    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_divergences": n_divergences,
        "rhat_by_var": rhat_by_var,
        "ess_by_var": ess_by_var,
        "converged": bool(
            max_rhat < RHAT_THRESHOLD
            and min_ess > MIN_STRUCTURAL_ESS
            and n_divergences == 0
        ),
    }


def posterior_recovery(trace, truth: dict[str, float]) -> dict:
    out = {}
    for name, true_value in truth.items():
        values = np.asarray(trace.posterior[name].values, dtype=float).reshape(-1)
        q025, q50, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        out[name] = {
            "true": true_value,
            "mean": float(values.mean()),
            "median": float(q50),
            "q025": float(q025),
            "q975": float(q975),
            "contains_true_95pct": bool(q025 <= true_value <= q975),
        }
    return out


def run_case(case: str, window: int, output_dir: Path) -> dict:
    truth = None
    if case == "synthetic":
        truth = {"mu": -9.0, "phi": 0.97, "sigma_eta": 0.22, "rho": -0.50}
        returns = simulate_sv_leverage(window, **truth)
        seed = 20260917
    else:
        returns = load_all_returns(verbose=False)["oil"].to_numpy(dtype=float)[:window]
        seed = 42

    mean_return = float(np.mean(returns))
    x = np.asarray(returns, dtype=float) - mean_return
    attempts = []
    final_trace = None
    final_diag = None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"centered_sv_leverage__{case}.json"

    for attempt_no, cfg in enumerate(DEFAULT_ROLLING_MCMC_ATTEMPTS, start=1):
        try:
            model = build_centered_model(x)
            with model:
                trace = pm.sample(
                    draws=int(cfg["draws"]),
                    tune=int(cfg["tune"]),
                    chains=int(cfg["chains"]),
                    cores=1,
                    progressbar=False,
                    random_seed=seed + attempt_no - 1,
                    target_accept=0.95,
                    nuts_sampler_kwargs={"max_treedepth": 12},
                )
            diag = diagnostics(trace)
            attempts.append({"attempt": attempt_no, **cfg, **diag})
            final_trace, final_diag = trace, diag
        except Exception as exc:
            attempts.append({
                "attempt": attempt_no,
                **cfg,
                "converged": False,
                "error": str(exc),
            })
            final_trace = None
            final_diag = None

        # Persist after every attempt so a later wall-time cancellation cannot
        # erase the diagnostic already completed.
        interim = {
            "candidate": CANDIDATE_VERSION,
            "case": case,
            "window": window,
            "truth": truth,
            "attempts": attempts,
            "converged": bool(final_diag and final_diag["converged"]),
            "accepted_attempt": next(
                (a["attempt"] for a in attempts if a.get("converged")), None
            ),
            **(final_diag or {}),
            "posterior_recovery": (
                posterior_recovery(final_trace, truth)
                if truth is not None and final_trace is not None
                else {}
            ),
        }
        path.write_text(json.dumps(_jsonable(interim), indent=2), encoding="utf-8")
        print(json.dumps(_jsonable(interim), indent=2), flush=True)
        if interim["converged"]:
            return interim

    return interim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["oil", "synthetic"], required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    payload = run_case(args.case, args.window, args.output_dir)
    if not payload["converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
