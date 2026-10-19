"""Evaluate a mathematically coherent bivariate-Student-t leverage SV candidate.

The candidate replaces the existing normal-plus-t convolution used by the
SV-t-Leverage label with a standardized bivariate Student-t pair
(epsilon_t, eta_t).  Both shocks have unit marginal variance, correlation rho,
and common degrees of freedom nu > 2:

    x_t       = exp(h_t / 2) * epsilon_t
    h_{t+1}   = mu + phi (h_t - mu) + sigma_eta * eta_t

Conditioning a bivariate t on eta_t gives

    epsilon_t | eta_t ~ t_{nu+1}(
        loc=rho * eta_t,
        scale=sqrt((1-rho^2) * ((nu-2)+eta_t^2)/(nu+1))
    )

when both marginals are variance-standardized.  This permits direct Bayesian
estimation without introducing one additional scale latent per observation.

This script is diagnostic only; it does not alter the production model family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from sv_model import DEFAULT_ROLLING_MCMC_ATTEMPTS, MIN_STRUCTURAL_ESS, RHAT_THRESHOLD


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


def _nu_prior():
    nu_minus_two = pm.Exponential("nu_minus_two", lam=0.1)
    return pm.Deterministic("nu", 2.0 + nu_minus_two)


def _build_model(returns: np.ndarray) -> pm.Model:
    T = len(returns)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10.0, sigma=3.0)
        phi_raw = pm.Beta("phi_raw", alpha=20.0, beta=1.5)
        phi = pm.Deterministic("phi", 2.0 * phi_raw - 1.0)
        sigma_eta = pm.HalfCauchy("sigma_eta", beta=0.5)
        nu = _nu_prior()
        rho = pm.Uniform("rho", lower=-1.0, upper=1.0)

        h0_std = pm.Normal("h0_std", mu=0.0, sigma=1.0)
        # Student-t scale is chosen so eta_t has unit marginal variance.
        eta_scale = pt.sqrt((nu - 2.0) / nu)
        eta = pm.StudentT("eta", nu=nu, mu=0.0, sigma=eta_scale, shape=T)

        # A variance-matched Gaussian initial state is retained to match the
        # production family's h_0 convention.  The AR innovations themselves
        # are the exact standardized Student-t shocks defined above.
        stationary_sd = sigma_eta / pt.sqrt(pt.clip(1.0 - phi**2, 1e-8, np.inf))
        h0 = mu + stationary_sd * h0_std

        def step(eta_t, h_prev, mu_, phi_, sigma_):
            return mu_ + phi_ * (h_prev - mu_) + sigma_ * eta_t

        h_rest, _ = pytensor.scan(
            fn=step,
            sequences=[eta],
            outputs_info=[h0],
            non_sequences=[mu, phi, sigma_eta],
            strict=True,
        )
        h = pm.Deterministic("h", pt.concatenate([h0[None], h_rest]))
        vol = pt.exp(h[:-1] / 2.0)

        cond_scale = vol * pt.sqrt(
            pt.clip(1.0 - rho**2, 1e-9, 1.0)
            * ((nu - 2.0) + eta**2)
            / (nu + 1.0)
        )
        pm.StudentT(
            "obs",
            nu=nu + 1.0,
            mu=rho * vol * eta,
            sigma=cond_scale,
            observed=returns,
        )
    return model


def _diagnostics(trace) -> dict:
    vars_ = ["mu", "phi", "sigma_eta", "nu", "rho"]
    rhat = az.rhat(trace, var_names=vars_)
    ess = az.ess(trace, var_names=vars_, method="bulk")
    rhat_by_var = {
        v: float(np.nanmax(np.asarray(rhat[v].values, dtype=float))) for v in vars_
    }
    ess_by_var = {
        v: float(np.nanmin(np.asarray(ess[v].values, dtype=float))) for v in vars_
    }
    max_rhat = max(rhat_by_var.values())
    min_ess = min(ess_by_var.values())
    divergences = int(trace.sample_stats["diverging"].sum().values)
    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_divergences": divergences,
        "rhat_by_var": rhat_by_var,
        "ess_by_var": ess_by_var,
        "converged": bool(
            max_rhat < RHAT_THRESHOLD
            and min_ess > MIN_STRUCTURAL_ESS
            and divergences == 0
        ),
    }


def _fit(returns: np.ndarray, seed: int) -> dict:
    mean_return = float(np.mean(returns))
    demeaned = np.asarray(returns, dtype=float) - mean_return
    records = []
    final_trace = None
    final_diag = None
    for attempt, cfg in enumerate(DEFAULT_ROLLING_MCMC_ATTEMPTS, start=1):
        model = _build_model(demeaned)
        try:
            with model:
                trace = pm.sample(
                    draws=int(cfg["draws"]),
                    tune=int(cfg["tune"]),
                    chains=int(cfg["chains"]),
                    cores=1,
                    progressbar=False,
                    random_seed=seed + attempt - 1,
                    target_accept=0.95,
                    nuts_sampler_kwargs={"max_treedepth": 12},
                )
            diag = _diagnostics(trace)
            records.append({"attempt": attempt, **cfg, **diag})
            final_trace, final_diag = trace, diag
            if diag["converged"]:
                break
        except Exception as exc:
            records.append({
                "attempt": attempt,
                **cfg,
                "converged": False,
                "error": str(exc),
            })
    return {
        "trace": final_trace,
        "mean_return": mean_return,
        "accepted_attempt": next((r["attempt"] for r in records if r.get("converged")), None),
        "attempts": records,
        **(final_diag or {
            "max_rhat": np.nan,
            "min_ess": np.nan,
            "n_divergences": np.nan,
            "rhat_by_var": {},
            "ess_by_var": {},
            "converged": False,
        }),
    }


def _simulate(n: int, seed: int = 20260915) -> tuple[np.ndarray, dict[str, float]]:
    truth = {"mu": -9.0, "phi": 0.97, "sigma_eta": 0.22, "rho": -0.50, "nu": 8.0}
    rng = np.random.default_rng(seed)
    nu = truth["nu"]
    rho = truth["rho"]
    scale = np.sqrt((nu - 2.0) / nu)
    mixing = rng.chisquare(df=nu, size=n) / nu
    z_eta = rng.normal(size=n)
    z_orth = rng.normal(size=n)
    eta = scale * z_eta / np.sqrt(mixing)
    eps = scale * (rho * z_eta + np.sqrt(1.0 - rho**2) * z_orth) / np.sqrt(mixing)

    h = np.empty(n + 1)
    h[0] = truth["mu"] + truth["sigma_eta"] / np.sqrt(1.0 - truth["phi"]**2) * rng.normal()
    x = np.empty(n)
    for t in range(n):
        x[t] = np.exp(h[t] / 2.0) * eps[t]
        h[t + 1] = truth["mu"] + truth["phi"] * (h[t] - truth["mu"]) + truth["sigma_eta"] * eta[t]
    return x, truth


def _recovery(trace, truth: dict[str, float]) -> dict:
    if trace is None:
        return {}
    out = {}
    for name, true in truth.items():
        values = np.asarray(trace.posterior[name].values, dtype=float).reshape(-1)
        q025, q50, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        out[name] = {
            "true": true,
            "posterior_mean": float(values.mean()),
            "posterior_median": float(q50),
            "q025": float(q025),
            "q975": float(q975),
            "contains_true_95pct": bool(q025 <= true <= q975),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["oil", "synthetic"], required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    if args.case == "oil":
        returns = load_all_returns(verbose=False)["oil"].to_numpy(dtype=float)[: args.window]
        truth = None
        seed = 42
    else:
        returns, truth = _simulate(args.window)
        seed = 20260915

    fit = _fit(returns, seed)
    payload = {
        "candidate": "SV-bivariate-t-Leverage",
        "case": args.case,
        "window": args.window,
        "converged": fit["converged"],
        "accepted_attempt": fit["accepted_attempt"],
        "max_rhat": fit["max_rhat"],
        "min_ess": fit["min_ess"],
        "n_divergences": fit["n_divergences"],
        "rhat_by_var": fit["rhat_by_var"],
        "ess_by_var": fit["ess_by_var"],
        "attempts": fit["attempts"],
        "truth": truth,
        "posterior_recovery": _recovery(fit["trace"], truth) if truth else {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"bivariate_t_candidate__{args.case}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))
    if not payload["converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
