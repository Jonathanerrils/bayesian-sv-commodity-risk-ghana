"""Bayesian stochastic-volatility models and adaptive rolling VaR/ES forecasts.

The rolling implementation deliberately separates two operations:

* structural parameters are re-estimated by MCMC every ``refit_every`` days;
* the latent volatility state is filtered after every realised return.

This avoids stale 42-day forecast blocks while retaining the computational
benefit of infrequent MCMC refits. Rolling MCMC itself is adaptive: each refit
starts with the empirically validated 4-chain / 1,000-draw configuration and is
retried at 4 chains / 2,000 draws only when the strict convergence gate fails.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

RHAT_THRESHOLD = 1.01
FAST_MIN_ESS = 200
MODEL_VERSION = "sv-filter-v2-adaptive-mcmc"
DEFAULT_ROLLING_MCMC_ATTEMPTS = (
    {"chains": 4, "tune": 1_000, "draws": 1_000},
    {"chains": 4, "tune": 2_000, "draws": 2_000},
)


def _nu_prior(name: str = "nu"):
    """Degrees of freedom constrained to nu > 2 so variance exists."""
    nu_minus_two = pm.Exponential(f"{name}_minus_two", lam=0.1)
    return pm.Deterministic(name, 2.0 + nu_minus_two)


def _pt_unit_variance_t_scale(nu):
    """PyTensor scale multiplier making a Student-t innovation unit variance."""
    return pt.sqrt((nu - 2.0) / nu)


def _np_unit_variance_t_scale(nu):
    nu = np.asarray(nu, dtype=float)
    return np.sqrt((nu - 2.0) / nu)


def _stationary_first_innovation(z, phi):
    """Innovation associated with the stationary initial AR(1) state."""
    first = pt.sqrt(pt.clip(1.0 - phi**2, 1e-9, 1.0)) * z[0]
    rest = z[1:] - phi * z[:-1]
    return pt.concatenate([[first], rest])


def build_sv_gaussian(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10, sigma=3)
        phi_raw = pm.Beta("phi_raw", alpha=20, beta=1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma = pm.HalfCauchy("sigma_eta", beta=0.5)
        z = pm.AR(
            "z", rho=phi, sigma=1.0,
            init_dist=pm.Normal.dist(0, 1 / pt.sqrt(1 - phi**2 + 1e-6)),
            shape=T,
        )
        h = pm.Deterministic("h", mu + sigma * z)
        pm.Normal("obs", mu=0, sigma=pt.exp(h / 2), observed=returns)
    return model


def build_sv_t(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10, sigma=3)
        phi_raw = pm.Beta("phi_raw", alpha=20, beta=1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma = pm.HalfCauchy("sigma_eta", beta=0.5)
        nu = _nu_prior()
        z = pm.AR(
            "z", rho=phi, sigma=1.0,
            init_dist=pm.Normal.dist(0, 1 / pt.sqrt(1 - phi**2 + 1e-6)),
            shape=T,
        )
        h = pm.Deterministic("h", mu + sigma * z)
        obs_scale = pt.exp(h / 2) * _pt_unit_variance_t_scale(nu)
        pm.StudentT("obs", nu=nu, mu=0, sigma=obs_scale, observed=returns)
    return model


def build_sv_leverage(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10, sigma=3)
        phi_raw = pm.Beta("phi_raw", alpha=20, beta=1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma = pm.HalfCauchy("sigma_eta", beta=0.5)
        rho = pm.Uniform("rho", lower=-1, upper=1)
        z = pm.AR(
            "z", rho=phi, sigma=1.0,
            init_dist=pm.Normal.dist(0, 1 / pt.sqrt(1 - phi**2 + 1e-6)),
            shape=T,
        )
        h = pm.Deterministic("h", mu + sigma * z)
        eta = _stationary_first_innovation(z, phi)
        vol = pt.exp(h / 2)
        mu_r = rho * vol * eta
        sigma_r = pt.sqrt(pt.clip(1 - rho**2, 1e-9, 1.0)) * vol
        pm.Normal("obs", mu=mu_r, sigma=sigma_r, observed=returns)
    return model


def build_sv_t_leverage(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=-10, sigma=3)
        phi_raw = pm.Beta("phi_raw", alpha=20, beta=1.5)
        phi = pm.Deterministic("phi", 2 * phi_raw - 1)
        sigma = pm.HalfCauchy("sigma_eta", beta=0.5)
        nu = _nu_prior()
        rho = pm.Uniform("rho", lower=-1, upper=1)
        z = pm.AR(
            "z", rho=phi, sigma=1.0,
            init_dist=pm.Normal.dist(0, 1 / pt.sqrt(1 - phi**2 + 1e-6)),
            shape=T,
        )
        h = pm.Deterministic("h", mu + sigma * z)
        eta = _stationary_first_innovation(z, phi)
        vol = pt.exp(h / 2)
        mu_r = rho * vol * eta
        sigma_r = (
            pt.sqrt(pt.clip(1 - rho**2, 1e-9, 1.0))
            * vol * _pt_unit_variance_t_scale(nu)
        )
        pm.StudentT("obs", nu=nu, mu=mu_r, sigma=sigma_r, observed=returns)
    return model


MODEL_BUILDERS = {
    "SV-Gaussian": build_sv_gaussian,
    "SV-t": build_sv_t,
    "SV-Leverage": build_sv_leverage,
    "SV-t-Leverage": build_sv_t_leverage,
}


def fit_sv(
    returns: np.ndarray,
    variant: str = "SV-t",
    chains: int = 2,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int = 42,
    fast_mode: bool = False,
) -> dict:
    """Fit one SV specification and return diagnostics with the trace."""
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(MODEL_BUILDERS)}")

    if fast_mode:
        chains, draws, tune = 1, 500, 500

    returns = np.asarray(returns, dtype=float)
    mean_return = float(np.mean(returns))
    demeaned = returns - mean_return
    model = MODEL_BUILDERS[variant](demeaned, len(demeaned))

    try:
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                progressbar=False,
                random_seed=random_seed,
                target_accept=target_accept,
                nuts_sampler_kwargs={"max_treedepth": 12},
            )

        n_chains = int(trace.posterior.sizes.get("chain", chains))
        scalar_vars = [v for v in trace.posterior.data_vars if v not in ("h", "z")]

        if n_chains > 1:
            rhat = az.rhat(trace, var_names=scalar_vars)
            rhat_values = [
                float(np.nanmax(rhat[v].values))
                for v in rhat.data_vars
                if np.isfinite(rhat[v].values).any()
            ]
            max_rhat = max(rhat_values) if rhat_values else np.nan
        else:
            max_rhat = np.nan

        ess = az.ess(trace, var_names=scalar_vars)
        ess_values = [
            float(np.nanmin(ess[v].values))
            for v in ess.data_vars
            if np.isfinite(ess[v].values).any()
        ]
        min_ess = min(ess_values) if ess_values else np.nan

        if "diverging" in trace.sample_stats:
            n_divergences = int(trace.sample_stats["diverging"].sum().values)
        else:
            n_divergences = 0

        if n_chains > 1:
            converged = (
                np.isfinite(max_rhat)
                and max_rhat < RHAT_THRESHOLD
                and min_ess > 400
                and n_divergences == 0
            )
        else:
            converged = min_ess > FAST_MIN_ESS and n_divergences == 0

        return {
            "trace": trace,
            "converged": bool(converged),
            "max_rhat": max_rhat,
            "min_ess": min_ess,
            "n_divergences": n_divergences,
            "variant": variant,
            "T": len(returns),
            "mean_return": mean_return,
            "model_version": MODEL_VERSION,
            "chains": chains,
            "draws": draws,
            "tune": tune,
        }
    except Exception as exc:
        return {
            "trace": None,
            "converged": False,
            "max_rhat": np.nan,
            "min_ess": np.nan,
            "n_divergences": np.nan,
            "variant": variant,
            "T": len(returns),
            "mean_return": mean_return,
            "model_version": MODEL_VERSION,
            "chains": chains,
            "draws": draws,
            "tune": tune,
            "error": str(exc),
        }


def fit_sv_adaptive(
    returns: np.ndarray,
    variant: str = "SV-t",
    target_accept: float = 0.95,
    random_seed: int = 42,
    attempts=None,
) -> dict:
    """Fit a rolling SV refit using a strict, empirically validated escalation.

    The default policy was validated on the real gold pilot: 4x1000 cleared the
    strict R-hat/ESS gate for SV-t, while SV-t-Leverage required escalation to
    4x2000. A failed first attempt is never used for forecasting.
    """
    if attempts is None:
        attempts = DEFAULT_ROLLING_MCMC_ATTEMPTS

    records = []
    final_fit = None
    for attempt_no, cfg in enumerate(attempts, start=1):
        fit = fit_sv(
            returns,
            variant=variant,
            chains=int(cfg["chains"]),
            draws=int(cfg["draws"]),
            tune=int(cfg["tune"]),
            target_accept=target_accept,
            random_seed=random_seed + attempt_no - 1,
            fast_mode=False,
        )
        records.append({
            "attempt": attempt_no,
            "chains": int(cfg["chains"]),
            "draws": int(cfg["draws"]),
            "tune": int(cfg["tune"]),
            "converged": bool(fit.get("converged", False)),
            "max_rhat": fit.get("max_rhat", np.nan),
            "min_ess": fit.get("min_ess", np.nan),
            "n_divergences": fit.get("n_divergences", np.nan),
            "error": fit.get("error"),
        })
        final_fit = fit
        if fit.get("converged", False):
            break

    final_fit = dict(final_fit)
    final_fit["mcmc_attempts"] = records
    final_fit["accepted_attempt"] = next(
        (r["attempt"] for r in records if r["converged"]), None
    )
    return final_fit


def initialize_filter_state(fit_result: dict) -> dict:
    """Create particle state from posterior draws at the latest fitted date."""
    trace = fit_result.get("trace")
    if trace is None:
        raise ValueError("Cannot initialize filter state without a posterior trace")

    state = {
        "variant": fit_result["variant"],
        "mean_return": float(fit_result["mean_return"]),
        "mu": trace.posterior["mu"].values.reshape(-1).astype(float),
        "phi": trace.posterior["phi"].values.reshape(-1).astype(float),
        "sigma_eta": trace.posterior["sigma_eta"].values.reshape(-1).astype(float),
        "h": trace.posterior["h"].values[:, :, -1].reshape(-1).astype(float),
    }
    if "nu" in trace.posterior:
        state["nu"] = trace.posterior["nu"].values.reshape(-1).astype(float)
    if "rho" in trace.posterior:
        state["rho"] = trace.posterior["rho"].values.reshape(-1).astype(float)
    return state


def _transition_filter_state(state: dict, rng: np.random.Generator) -> dict:
    """Propagate latent log variance one day forward for every particle."""
    eta = rng.normal(size=len(state["h"]))
    h_next = (
        state["mu"]
        + state["phi"] * (state["h"] - state["mu"])
        + state["sigma_eta"] * eta
    )
    transition = {k: v for k, v in state.items() if k != "h"}
    transition["h"] = h_next
    transition["eta"] = eta
    return transition


def _predictive_returns(
    transition: dict,
    n_predictive: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from the one-day posterior predictive distribution."""
    n_particles = len(transition["h"])
    idx = rng.integers(0, n_particles, size=int(n_predictive))
    h = transition["h"][idx]
    eta = transition["eta"][idx]
    vol = np.exp(h / 2)
    variant = transition["variant"]

    if variant in ("SV-t", "SV-t-Leverage"):
        nu = transition["nu"][idx]
        xi = rng.standard_t(nu) * _np_unit_variance_t_scale(nu)
    else:
        xi = rng.normal(size=len(idx))

    if variant in ("SV-Leverage", "SV-t-Leverage"):
        rho = transition["rho"][idx]
        eps = rho * eta + np.sqrt(np.clip(1 - rho**2, 1e-12, 1.0)) * xi
    else:
        eps = xi

    return transition["mean_return"] + vol * eps


def _observation_loglik(transition: dict, actual_return: float) -> np.ndarray:
    """Observation likelihood used to filter the latent state after each day."""
    x = float(actual_return) - transition["mean_return"]
    h = transition["h"]
    eta = transition["eta"]
    vol = np.exp(h / 2)
    variant = transition["variant"]

    if variant in ("SV-Leverage", "SV-t-Leverage"):
        rho = transition["rho"]
        loc = rho * vol * eta
        base_scale = np.sqrt(np.clip(1 - rho**2, 1e-12, 1.0)) * vol
    else:
        loc = np.zeros_like(vol)
        base_scale = vol

    if variant in ("SV-t", "SV-t-Leverage"):
        nu = transition["nu"]
        scale = base_scale * _np_unit_variance_t_scale(nu)
        return stats.t.logpdf(x, df=nu, loc=loc, scale=np.maximum(scale, 1e-12))

    return stats.norm.logpdf(x, loc=loc, scale=np.maximum(base_scale, 1e-12))


def update_filter_state(
    transition: dict,
    actual_return: float,
    rng: np.random.Generator,
) -> tuple[dict, float]:
    """Condition the propagated particles on the newly observed return."""
    logw = _observation_loglik(transition, actual_return)
    finite = np.isfinite(logw)
    if not finite.any():
        weights = np.full(len(logw), 1 / len(logw))
    else:
        floor = np.nanmax(logw[finite]) - 1_000.0
        logw = np.where(finite, logw, floor)
        logw -= np.max(logw)
        weights = np.exp(logw)
        weights /= weights.sum()

    filter_ess = float(1.0 / np.sum(weights**2))
    idx = rng.choice(len(weights), size=len(weights), replace=True, p=weights)

    new_state = {
        "variant": transition["variant"],
        "mean_return": transition["mean_return"],
    }
    for key in ("mu", "phi", "sigma_eta", "nu", "rho"):
        if key in transition:
            new_state[key] = transition[key][idx]
    new_state["h"] = transition["h"][idx]
    return new_state, filter_ess


def predictive_var_es(r_pred: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute positive-loss VaR and ES from predictive returns."""
    q = float(np.quantile(r_pred, alpha))
    var = -q
    tail = r_pred[r_pred <= q]
    es = -float(tail.mean()) if len(tail) else var
    return float(var), float(es)


def forecast_sv_var_es(
    fit_result: dict,
    alpha: float,
    n_predictive: int = 20_000,
    random_seed: int = 42,
) -> tuple:
    """Compatibility helper for a single one-step forecast after a fresh fit."""
    if fit_result.get("trace") is None:
        return np.nan, np.nan
    state = initialize_filter_state(fit_result)
    rng = np.random.default_rng(random_seed)
    transition = _transition_filter_state(state, rng)
    r_pred = _predictive_returns(transition, n_predictive, rng)
    return predictive_var_es(r_pred, alpha)


def _state_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".state.npz")


def _save_filter_state(path: Path, state: dict, next_i: int, last_fit_idx: int):
    arrays = {
        "next_i": np.array([next_i], dtype=int),
        "last_fit_idx": np.array([last_fit_idx], dtype=int),
        "variant": np.array([state["variant"]]),
        "mean_return": np.array([state["mean_return"]], dtype=float),
    }
    for key in ("mu", "phi", "sigma_eta", "nu", "rho", "h"):
        if key in state:
            arrays[key] = np.asarray(state[key])
    np.savez_compressed(path, **arrays)


def _load_filter_state(path: Path) -> tuple[dict, int, int]:
    data = np.load(path, allow_pickle=False)
    state = {
        "variant": str(data["variant"][0]),
        "mean_return": float(data["mean_return"][0]),
    }
    for key in ("mu", "phi", "sigma_eta", "nu", "rho", "h"):
        if key in data.files:
            state[key] = data[key]
    return state, int(data["next_i"][0]), int(data["last_fit_idx"][0])


def rolling_sv_var_es(
    returns: pd.Series,
    variant: str = "SV-t",
    window: int = 1000,
    refit_every: int = 42,
    target_accept: float = 0.95,
    alphas: list | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
    n_predictive: int = 20_000,
    random_seed: int = 42,
    mcmc_attempts=None,
) -> pd.DataFrame:
    """Adaptive walk-forward VaR/ES with strict MCMC refits and daily filtering."""
    if alphas is None:
        alphas = [0.01, 0.05]
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"Unknown variant: {variant}")
    if len(returns) <= window:
        raise ValueError("Series length must exceed rolling window")
    if mcmc_attempts is None:
        mcmc_attempts = DEFAULT_ROLLING_MCMC_ATTEMPTS

    ret_array = returns.to_numpy(dtype=float)
    dates = returns.index
    n_forecasts = len(ret_array) - window
    results: list[dict] = []
    n_failures = 0
    filter_state = None
    last_fit_idx = -refit_every
    start_i = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    sidecar = _state_sidecar_path(checkpoint_path) if checkpoint_path else None

    if checkpoint_path and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path, parse_dates=["date"])
        if len(existing) >= n_forecasts:
            return existing.set_index("date")
        if len(existing) > 0:
            if sidecar is None or not sidecar.exists():
                raise RuntimeError(
                    "Partial CSV exists without matching filter-state sidecar; "
                    "refusing an inexact resume. Remove the partial checkpoint "
                    "to restart this model cleanly."
                )
            filter_state, state_next_i, last_fit_idx = _load_filter_state(sidecar)
            if state_next_i != len(existing):
                raise RuntimeError("Checkpoint CSV/state sidecar are out of sync")
            if filter_state["variant"] != variant:
                raise RuntimeError("Checkpoint state belongs to a different SV variant")
            results = existing.to_dict("records")
            start_i = len(existing)
            n_failures = int(existing["estimation_failed"].sum())

    for i in range(start_i, n_forecasts):
        train = ret_array[i : i + window]
        actual_return = ret_array[i + window]
        forecast_date = dates[i + window]
        row = {
            "date": forecast_date,
            "actual_return": actual_return,
            "estimation_failed": False,
            "refit": False,
        }

        if filter_state is None or (i - last_fit_idx) >= refit_every:
            fit = fit_sv_adaptive(
                train,
                variant=variant,
                target_accept=target_accept,
                random_seed=random_seed + i,
                attempts=mcmc_attempts,
            )
            row["refit"] = True
            row["mcmc_attempt"] = fit.get("accepted_attempt", np.nan)
            row["mcmc_max_rhat"] = fit.get("max_rhat", np.nan)
            row["mcmc_min_ess"] = fit.get("min_ess", np.nan)
            row["mcmc_divergences"] = fit.get("n_divergences", np.nan)
            if not fit.get("converged", False):
                n_failures += 1
                filter_state = None
            else:
                filter_state = initialize_filter_state(fit)
                last_fit_idx = i

        if filter_state is None:
            row["estimation_failed"] = True
            row["filter_ess"] = np.nan
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"] = np.nan
        else:
            step_rng = np.random.default_rng(random_seed * 1_000_003 + i)
            transition = _transition_filter_state(filter_state, step_rng)
            r_pred = _predictive_returns(transition, n_predictive, step_rng)
            for alpha in alphas:
                var, es = predictive_var_es(r_pred, alpha)
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"] = es

            # Forecast first, then condition on today's realised return.
            filter_state, filter_ess = update_filter_state(
                transition, actual_return, step_rng
            )
            row["filter_ess"] = filter_ess

        results.append(row)

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            if filter_state is not None:
                _save_filter_state(sidecar, filter_state, i + 1, last_fit_idx)

    df = pd.DataFrame(results).set_index("date")
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(checkpoint_path)
        if filter_state is not None:
            _save_filter_state(sidecar, filter_state, n_forecasts, last_fit_idx)

    if n_failures:
        print(f"WARNING [{variant}]: {n_failures} failed MCMC refits after escalation")
    return df
