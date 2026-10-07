"""Controlled Gold SV-Gaussian geometry candidates with the statistical model held fixed."""

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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_utils import load_all_returns
from diagnose_sv_refit import _divergence_geometry, _jsonable
from sv_model import _structural_diagnostics


def _state_path(mu, phi, sigma_eta, T: int, *, centered_h0: bool):
    eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=T)
    stationary_sd = sigma_eta / pt.sqrt(pt.clip(1.0 - phi**2, 1e-12, np.inf))
    if centered_h0:
        h0 = pm.Normal("h0", mu=mu, sigma=stationary_sd)
    else:
        h0_std = pm.Normal("h0_std", mu=0.0, sigma=1.0)
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
    return h


def _persistence(*, logit_phi: bool):
    if not logit_phi:
        phi_raw = pm.Beta("phi_raw", alpha=20.0, beta=1.5)
    else:
        phi_logit = pm.Flat("phi_logit")
        phi_raw = pm.Deterministic("phi_raw", pm.math.sigmoid(phi_logit))
        pm.Potential(
            "phi_raw_transformed_prior",
            pm.logp(pm.Beta.dist(alpha=20.0, beta=1.5), phi_raw)
            + pt.log(phi_raw)
            + pt.log1p(-phi_raw),
        )
    return pm.Deterministic("phi", 2.0 * phi_raw - 1.0)


def build_model(returns: np.ndarray, geometry: str) -> pm.Model:
    centered_h0 = geometry in {"centered-h0", "centered-h0-logit-phi"}
    logit_phi = geometry in {"logit-phi", "centered-h0-logit-phi"}
    T = len(returns)
    mean_return = float(np.mean(returns))
    demeaned = returns - mean_return
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10.0, sigma=3.0)
        phi = _persistence(logit_phi=logit_phi)
        sigma_eta = pm.HalfCauchy("sigma_eta", beta=0.5)
        h = _state_path(mu, phi, sigma_eta, T, centered_h0=centered_h0)
        vol = pt.exp(h[:-1] / 2.0)
        pm.Normal("obs", mu=0.0, sigma=vol, observed=demeaned)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=["centered-h0", "logit-phi", "centered-h0-logit-phi"],
        required=True,
    )
    parser.add_argument("--commodity", default="gold", choices=["cocoa", "gold", "oil"])
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--attempt", type=int, default=2, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    attempts = {
        1: {"chains": 4, "tune": 1000, "draws": 1000},
        2: {"chains": 4, "tune": 2000, "draws": 2000},
    }
    cfg = attempts[args.attempt]
    start_i = args.block * args.refit_every
    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    train = returns[start_i:start_i + args.window]
    if len(train) != args.window:
        raise ValueError("requested block does not have a complete fitting window")

    model = build_model(train, args.geometry)
    fit_seed = 42 + start_i + args.attempt - 1
    with model:
        trace = pm.sample(
            draws=cfg["draws"],
            tune=cfg["tune"],
            chains=cfg["chains"],
            cores=1,
            progressbar=False,
            random_seed=fit_seed,
            target_accept=args.target_accept,
            nuts_sampler_kwargs={"max_treedepth": 12},
        )

    diagnostics = _structural_diagnostics(trace, cfg["chains"])
    fit = {"trace": trace, **diagnostics}
    payload = {
        "kind": "sv_gaussian_geometry_candidate",
        "commodity": args.commodity,
        "geometry": args.geometry,
        "block": args.block,
        "window": args.window,
        "attempt": args.attempt,
        "chains": cfg["chains"],
        "tune": cfg["tune"],
        "draws": cfg["draws"],
        "target_accept": args.target_accept,
        "fit_seed": fit_seed,
        "converged": bool(diagnostics["converged"]),
        "max_rhat": diagnostics["max_rhat"],
        "min_ess": diagnostics["min_ess"],
        "n_divergences": diagnostics["n_divergences"],
        "rhat_by_var": diagnostics["rhat_by_var"],
        "ess_by_var": diagnostics["ess_by_var"],
        "divergence_geometry": _divergence_geometry(fit),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"gaussian_geometry__{args.commodity}__{args.geometry}__attempt-{args.attempt}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
