"""Bayesian stochastic-volatility models and rolling VaR/ES forecasts.

The model uses an innovation-noncentred AR(1) state parameterisation and an
explicit forecast-date state convention.

For a mean-adjusted return x_t, the symmetric state equation is

    x_t = exp(h_t / 2) * epsilon_t
    h_{t+1} = mu + phi * (h_t - mu) + sigma_eta * eta_t

The frozen v5 primary family contains only the symmetric SV-Gaussian and SV-t
specifications. Leverage builders remain available for diagnostic/sensitivity
work so the audit trail is reproducible, but they are not production-primary
models after failing the pre-production convergence programme.

A rolling fit over T observed returns contains states h_0,...,h_T. The final
posterior state h_T is therefore the state for the next forecast date. Daily
filtering forecasts from h_t first, then conditions on the realised return to
infer eta_t and propagates to h_{t+1}.

Structural parameters are re-estimated only at the predeclared refit boundaries
0, refit_every, 2*refit_every, ... . If a scheduled refit fails the strict MCMC
gate, that entire forecast block is explicitly unavailable; the algorithm does
not retry on an easier next-day window. This makes model availability an
observable outcome and keeps canonical and distributed production semantics
identical.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats

from checkpoint_safety import (
    checkpoint_schema_array,
    parse_bool_series,
    parse_bool_value,
    validate_checkpoint_schema,
)

warnings.filterwarnings("ignore", category=FutureWarning)

RHAT_THRESHOLD = 1.01
MIN_STRUCTURAL_ESS = 400
FAST_MIN_ESS = 200
MODEL_VERSION = "sv-filter-v4-primary-symmetric-noncentered"
DEFAULT_ROLLING_MCMC_ATTEMPTS = (
    {"chains": 4, "tune": 1_000, "draws": 1_000},
    {"chains": 4, "tune": 2_000, "draws": 2_000},
)
STRUCTURAL_DIAGNOSTIC_VARS = ("mu", "phi", "sigma_eta", "nu", "rho")
PRIMARY_SV_VARIANTS = ("SV-Gaussian", "SV-t")
DIAGNOSTIC_SV_VARIANTS = ("SV-Leverage", "SV-t-Leverage")


def _nu_prior(name: str = "nu"):
    """Degrees of freedom constrained to nu > 2 so variance exists."""
    nu_minus_two = pm.Exponential(f"{name}_minus_two", lam=0.1)
    return pm.Deterministic(name, 2.0 + nu_minus_two)


def _pt_unit_variance_t_scale(nu):
    """PyTensor multiplier making a Student-t innovation unit variance."""
    return pt.sqrt((nu - 2.0) / nu)


def _np_unit_variance_t_scale(nu):
    """NumPy multiplier making a Student-t innovation unit variance."""
    nu = np.asarray(nu, dtype=float)
    return np.sqrt((nu - 2.0) / nu)


def _noncentered_state_path(mu, phi, sigma_eta, T: int):
    """Construct h_0,...,h_T from independent standard-normal innovations."""
    h0_std = pm.Normal("h0_std", mu=0.0, sigma=1.0)
    eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=T)
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
    return h, eta


def _common_parameters(T: int):
    mu = pm.Normal("mu", mu=-10.0, sigma=3.0)
    phi_raw = pm.Beta("phi_raw", alpha=20.0, beta=1.5)
    phi = pm.Deterministic("phi", 2.0 * phi_raw - 1.0)
    sigma_eta = pm.HalfCauchy("sigma_eta", beta=0.5)
    h, eta = _noncentered_state_path(mu, phi, sigma_eta, T)
    return mu, phi, sigma_eta, h, eta


def build_sv_gaussian(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        _mu, _phi, _sigma, h, _eta = _common_parameters(T)
        vol = pt.exp(h[:-1] / 2.0)
        pm.Normal("obs", mu=0.0, sigma=vol, observed=returns)
    return model


def build_sv_t(returns: np.ndarray, T: int) -> pm.Model:
    with pm.Model() as model:
        _mu, _phi, _sigma, h, _eta = _common_parameters(T)
        nu = _nu_prior()
        vol = pt.exp(h[:-1] / 2.0)
        obs_scale = vol * _pt_unit_variance_t_scale(nu)
        pm.StudentT("obs", nu=nu, mu=0.0, sigma=obs_scale, observed=returns)
    return model


def build_sv_leverage(returns: np.ndarray, T: int) -> pm.Model:
    """Diagnostic-only Gaussian SV with corr(epsilon_t, eta_t)=rho."""
    with pm.Model() as model:
        _mu, _phi, _sigma, h, eta = _common_parameters(T)
        rho = pm.Uniform("rho", lower=-1.0, upper=1.0)
        vol = pt.exp(h[:-1] / 2.0)
        obs_mu = rho * vol * eta
        obs_sd = pt.sqrt(pt.clip(1.0 - rho**2, 1e-9, 1.0)) * vol
        pm.Normal("obs", mu=obs_mu, sigma=obs_sd, observed=returns)
    return model


def build_sv_t_leverage(returns: np.ndarray, T: int) -> pm.Model:
    """Diagnostic-only leverage SV with a Student-t orthogonal innovation.

    Conditional on eta_t the orthogonal residual is Student-t. The marginal
    epsilon_t is a normal-t convolution, not exactly a Student-t distribution.
    This legacy specification is retained only to reproduce audit diagnostics;
    it is excluded from the frozen v5 primary family.
    """
    with pm.Model() as model:
        _mu, _phi, _sigma, h, eta = _common_parameters(T)
        nu = _nu_prior()
        rho = pm.Uniform("rho", lower=-1.0, upper=1.0)
        vol = pt.exp(h[:-1] / 2.0)
        obs_mu = rho * vol * eta
        obs_scale = (
            pt.sqrt(pt.clip(1.0 - rho**2, 1e-9, 1.0))
            * vol
            * _pt_unit_variance_t_scale(nu)
        )
        pm.StudentT("obs", nu=nu, mu=obs_mu, sigma=obs_scale, observed=returns)
    return model


# All builders stay addressable for reproducible diagnostics. Production imports
# PRIMARY_SV_VARIANTS explicitly and therefore cannot select leverage models.
MODEL_BUILDERS = {
    "SV-Gaussian": build_sv_gaussian,
    "SV-t": build_sv_t,
    "SV-Leverage": build_sv_leverage,
    "SV-t-Leverage": build_sv_t_leverage,
}


def _structural_diagnostics(trace, chains: int) -> dict:
    var_names = [v for v in STRUCTURAL_DIAGNOSTIC_VARS if v in trace.posterior]
    if not var_names:
        return {
            "max_rhat": np.nan,
            "min_ess": np.nan,
            "n_divergences": np.nan,
            "rhat_by_var": {},
            "ess_by_var": {},
            "converged": False,
        }

    n_chains = int(trace.posterior.sizes.get("chain", chains))
    rhat_by_var = {}
    if n_chains > 1:
        rhat = az.rhat(trace, var_names=var_names)
        for var in rhat.data_vars:
            values = np.asarray(rhat[var].values, dtype=float)
            if np.isfinite(values).any():
                rhat_by_var[var] = float(np.nanmax(values))
        max_rhat = max(rhat_by_var.values()) if rhat_by_var else np.nan
    else:
        max_rhat = np.nan

    ess = az.ess(trace, var_names=var_names, method="bulk")
    ess_by_var = {}
    for var in ess.data_vars:
        values = np.asarray(ess[var].values, dtype=float)
        if np.isfinite(values).any():
            ess_by_var[var] = float(np.nanmin(values))
    min_ess = min(ess_by_var.values()) if ess_by_var else np.nan

    if "diverging" in trace.sample_stats:
        n_divergences = int(trace.sample_stats["diverging"].sum().values)
    else:
        n_divergences = 0

    if n_chains > 1:
        converged = (
            np.isfinite(max_rhat)
            and max_rhat < RHAT_THRESHOLD
            and np.isfinite(min_ess)
            and min_ess > MIN_STRUCTURAL_ESS
            and n_divergences == 0
        )
    else:
        converged = (
            np.isfinite(min_ess)
            and min_ess > FAST_MIN_ESS
            and n_divergences == 0
        )
    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_divergences": n_divergences,
        "rhat_by_var": rhat_by_var,
        "ess_by_var": ess_by_var,
        "converged": bool(converged),
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
    """Fit one SV specification and return structural convergence diagnostics."""
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(MODEL_BUILDERS)}")
    if fast_mode:
        chains, draws, tune = 1, 500, 500

    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("returns must be a finite one-dimensional array")
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
        diagnostics = _structural_diagnostics(trace, chains)
        return {
            "trace": trace,
            **diagnostics,
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
            "rhat_by_var": {},
            "ess_by_var": {},
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
    """Apply the strict rolling MCMC escalation; never accept a weak trace."""
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
            "rhat_by_var": fit.get("rhat_by_var", {}),
            "ess_by_var": fit.get("ess_by_var", {}),
            "error": fit.get("error"),
        })
        final_fit = fit
        if fit.get("converged", False):
            break
    if final_fit is None:
        raise ValueError("at least one MCMC attempt is required")
    final_fit = dict(final_fit)
    final_fit["mcmc_attempts"] = records
    final_fit["accepted_attempt"] = next(
        (r["attempt"] for r in records if r["converged"]), None
    )
    return final_fit


def initialize_filter_state(fit_result: dict) -> dict:
    """Create particles whose h value is the next forecast-date state."""
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
    """Draw eta_t and propose h_{t+1}, retaining h_t for today's forecast."""
    eta = rng.normal(size=len(state["h"]))
    h_next = (
        state["mu"]
        + state["phi"] * (state["h"] - state["mu"])
        + state["sigma_eta"] * eta
    )
    transition = {k: v for k, v in state.items()}
    transition["eta"] = eta
    transition["h_next"] = h_next
    return transition


def _predictive_returns(transition: dict, n_predictive: int, rng: np.random.Generator) -> np.ndarray:
    """Draw r_t from h_t jointly with proposed eta_t for diagnostic leverage."""
    n_particles = len(transition["h"])
    idx = rng.integers(0, n_particles, size=int(n_predictive))
    h = transition["h"][idx]
    eta = transition["eta"][idx]
    vol = np.exp(h / 2.0)
    variant = transition["variant"]
    if variant in ("SV-t", "SV-t-Leverage"):
        nu = transition["nu"][idx]
        xi = rng.standard_t(nu) * _np_unit_variance_t_scale(nu)
    else:
        xi = rng.normal(size=len(idx))
    if variant in ("SV-Leverage", "SV-t-Leverage"):
        rho = transition["rho"][idx]
        eps = rho * eta + np.sqrt(np.clip(1.0 - rho**2, 1e-12, 1.0)) * xi
    else:
        eps = xi
    return transition["mean_return"] + vol * eps


def _observation_loglik(transition: dict, actual_return: float) -> np.ndarray:
    """Likelihood of r_t given h_t and proposed eta_t for particle weighting."""
    x = float(actual_return) - transition["mean_return"]
    h = transition["h"]
    eta = transition["eta"]
    vol = np.exp(h / 2.0)
    variant = transition["variant"]
    if variant in ("SV-Leverage", "SV-t-Leverage"):
        rho = transition["rho"]
        loc = rho * vol * eta
        base_scale = np.sqrt(np.clip(1.0 - rho**2, 1e-12, 1.0)) * vol
    else:
        loc = np.zeros_like(vol)
        base_scale = vol
    if variant in ("SV-t", "SV-t-Leverage"):
        nu = transition["nu"]
        scale = base_scale * _np_unit_variance_t_scale(nu)
        return stats.t.logpdf(x, df=nu, loc=loc, scale=np.maximum(scale, 1e-12))
    return stats.norm.logpdf(x, loc=loc, scale=np.maximum(base_scale, 1e-12))


def update_filter_state(transition: dict, actual_return: float, rng: np.random.Generator) -> tuple[dict, float]:
    """Condition on r_t, resample, and advance particles from h_t to h_{t+1}."""
    if "h_next" not in transition:
        raise ValueError("transition is missing h_next; call _transition_filter_state first")
    logw = _observation_loglik(transition, actual_return)
    finite = np.isfinite(logw)
    if not finite.any():
        weights = np.full(len(logw), 1.0 / len(logw))
    else:
        floor = np.nanmax(logw[finite]) - 1_000.0
        logw = np.where(finite, logw, floor)
        logw -= np.max(logw)
        weights = np.exp(logw)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(logw), 1.0 / len(logw))
        else:
            weights /= total
    filter_ess = float(1.0 / np.sum(weights**2))
    idx = rng.choice(len(weights), size=len(weights), replace=True, p=weights)
    new_state = {
        "variant": transition["variant"],
        "mean_return": transition["mean_return"],
    }
    for key in ("mu", "phi", "sigma_eta", "nu", "rho"):
        if key in transition:
            new_state[key] = transition[key][idx]
    new_state["h"] = transition["h_next"][idx]
    return new_state, filter_ess


def predictive_var_es(r_pred: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute positive-loss VaR and ES from predictive returns."""
    q = float(np.quantile(r_pred, alpha))
    var = -q
    tail = r_pred[r_pred <= q]
    es = -float(tail.mean()) if len(tail) else var
    return float(var), float(es)


def forecast_sv_var_es(fit_result: dict, alpha: float, n_predictive: int = 20_000, random_seed: int = 42) -> tuple:
    """One-step forecast after a fresh fit using its h_T forecast-date state."""
    if fit_result.get("trace") is None:
        return np.nan, np.nan
    state = initialize_filter_state(fit_result)
    rng = np.random.default_rng(random_seed)
    transition = _transition_filter_state(state, rng)
    r_pred = _predictive_returns(transition, n_predictive, rng)
    return predictive_var_es(r_pred, alpha)


def _state_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".state.npz")


def _save_filter_state(
    path: Path,
    state: dict | None,
    next_i: int,
    active_block_start: int,
    variant: str,
):
    """Persist active particle state or an explicit unavailable block safely."""
    arrays = {
        "checkpoint_schema_version": checkpoint_schema_array(),
        "model_version": np.array([MODEL_VERSION]),
        "next_i": np.array([next_i], dtype=int),
        "active_block_start": np.array([active_block_start], dtype=int),
        "variant": np.array([variant]),
        "has_state": np.array([state is not None], dtype=bool),
    }
    if state is not None:
        arrays["mean_return"] = np.array([state["mean_return"]], dtype=float)
        for key in ("mu", "phi", "sigma_eta", "nu", "rho", "h"):
            if key in state:
                arrays[key] = np.asarray(state[key])
    np.savez_compressed(path, **arrays)


def _load_filter_state(path: Path) -> tuple[dict | None, int, int, str]:
    """Load a sidecar only when its schema and SV model version match exactly."""
    with np.load(path, allow_pickle=False) as data:
        schema_values = (
            data["checkpoint_schema_version"]
            if "checkpoint_schema_version" in data.files
            else np.array([], dtype=int)
        )
        validate_checkpoint_schema(data.files, schema_values)
        if "model_version" not in data.files:
            raise RuntimeError(
                "SV checkpoint sidecar has no model version; refusing an inexact "
                "resume. Restart the checkpoint under the current pipeline."
            )
        sidecar_model_version = str(np.asarray(data["model_version"]).reshape(-1)[0])
        if sidecar_model_version != MODEL_VERSION:
            raise RuntimeError(
                f"SV checkpoint belongs to model version {sidecar_model_version!r}; "
                f"expected {MODEL_VERSION!r}. Restart the checkpoint."
            )

        variant = str(data["variant"][0])
        has_state = parse_bool_value(data["has_state"][0], missing=False)
        state = None
        if has_state:
            state = {
                "variant": variant,
                "mean_return": float(data["mean_return"][0]),
            }
            for key in ("mu", "phi", "sigma_eta", "nu", "rho", "h"):
                if key in data.files:
                    state[key] = np.asarray(data[key])
        return (
            state,
            int(data["next_i"][0]),
            int(data["active_block_start"][0]),
            variant,
        )


def _resume_failed_refit_count(existing: pd.DataFrame) -> int:
    """Count failed scheduled refits without ever using string truthiness."""
    if "refit" not in existing:
        return 0
    refit_mask = parse_bool_series(existing["refit"], missing=False)
    refit_rows = existing.loc[refit_mask]
    if len(refit_rows) == 0 or "mcmc_converged" not in refit_rows:
        return 0
    converged = parse_bool_series(refit_rows["mcmc_converged"], missing=False)
    return int((~converged).sum())


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
    """Walk-forward VaR/ES with fixed scheduled refits and daily state filtering."""
    if alphas is None:
        alphas = [0.01, 0.05]
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"Unknown variant: {variant}")
    if len(returns) <= window:
        raise ValueError("Series length must exceed rolling window")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    if mcmc_attempts is None:
        mcmc_attempts = DEFAULT_ROLLING_MCMC_ATTEMPTS

    ret_array = returns.to_numpy(dtype=float)
    dates = returns.index
    n_forecasts = len(ret_array) - window
    results: list[dict] = []
    n_failed_refits = 0
    filter_state = None
    active_block_start = 0
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
                    "refusing an inexact resume. Remove the partial checkpoint to restart cleanly."
                )
            filter_state, state_next_i, active_block_start, state_variant = _load_filter_state(sidecar)
            if state_next_i != len(existing):
                raise RuntimeError("Checkpoint CSV/state sidecar are out of sync")
            if state_variant != variant:
                raise RuntimeError("Checkpoint state belongs to a different SV variant")
            results = existing.to_dict("records")
            start_i = len(existing)
            n_failed_refits = _resume_failed_refit_count(existing)

    for i in range(start_i, n_forecasts):
        actual_return = ret_array[i + window]
        row = {
            "date": dates[i + window],
            "actual_return": actual_return,
            "estimation_failed": False,
            "refit": False,
            "mcmc_converged": np.nan,
        }

        scheduled_refit = (i % refit_every) == 0
        if scheduled_refit:
            active_block_start = i
            train = ret_array[i : i + window]
            fit = fit_sv_adaptive(
                train,
                variant=variant,
                target_accept=target_accept,
                random_seed=random_seed + i,
                attempts=mcmc_attempts,
            )
            row["refit"] = True
            row["mcmc_converged"] = bool(fit.get("converged", False))
            row["mcmc_attempt"] = fit.get("accepted_attempt", np.nan)
            row["mcmc_max_rhat"] = fit.get("max_rhat", np.nan)
            row["mcmc_min_ess"] = fit.get("min_ess", np.nan)
            row["mcmc_divergences"] = fit.get("n_divergences", np.nan)
            row["mcmc_attempts_json"] = str(fit.get("mcmc_attempts", []))
            if not fit.get("converged", False):
                n_failed_refits += 1
                filter_state = None
            else:
                filter_state = initialize_filter_state(fit)

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
            filter_state, filter_ess = update_filter_state(
                transition, actual_return, step_rng
            )
            row["filter_ess"] = filter_ess

        results.append(row)

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            _save_filter_state(
                sidecar, filter_state, i + 1, active_block_start, variant
            )

    df = pd.DataFrame(results).set_index("date")
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(checkpoint_path)
        _save_filter_state(
            sidecar, filter_state, n_forecasts, active_block_start, variant
        )

    if n_failed_refits:
        print(
            f"WARNING [{variant}]: {n_failed_refits} scheduled MCMC refits failed "
            "after escalation; their complete forecast blocks remain unavailable"
        )
    return df
