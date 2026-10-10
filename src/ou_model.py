"""Exact Ornstein-Uhlenbeck benchmark aligned to retained log-return pairs.

For X_t = log(P_t),

    dX_t = kappa (theta - X_t) dt + sigma dW_t

has exact transition

    X_{t+dt}|X_t ~ N(theta + (X_t-theta)e^{-kappa dt},
                     sigma^2(1-e^{-2 kappa dt})/(2 kappa)).

The rolling implementation estimates the likelihood on the exact transition
pairs underlying each retained log return.  If r_t = log(P_t/P_{t-1}), then
X_{t-1} = log(P_t) - r_t.  Recovering the conditioning state this way remains
correct even when the cleaned return index skips dates after missing prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

OU_MODEL_VERSION = "ou-exact-v5-return-aligned-pairs"
MEAN_REVERSION_DETECTION_KAPPA = 0.01


def ou_transition_moments(
    x_prev: np.ndarray | float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: np.ndarray | float = 1 / 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditional mean/variance, supporting scalar or vector ``dt``."""
    if kappa <= 0 or sigma <= 0:
        raise ValueError("kappa and sigma must be positive")
    x_prev = np.asarray(x_prev, dtype=float)
    dt_arr = np.asarray(dt, dtype=float)
    if not np.isfinite(dt_arr).all() or np.any(dt_arr <= 0):
        raise ValueError("dt must be finite and positive")
    decay = np.exp(-kappa * dt_arr)
    mean = theta + (x_prev - theta) * decay
    variance = sigma**2 * (-np.expm1(-2.0 * kappa * dt_arr)) / (2.0 * kappa)
    mean, variance = np.broadcast_arrays(mean, variance)
    return mean.astype(float), variance.astype(float)


def _failure_result() -> dict:
    return {
        "kappa": np.nan,
        "theta": np.nan,
        "sigma": np.nan,
        "log_likelihood": np.nan,
        "optimizer_converged": False,
        "converged": False,
        "mean_reversion_detected": False,
        "half_life_years": np.nan,
        "model_version": OU_MODEL_VERSION,
    }


def _initial_guess_transitions(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    dt: np.ndarray | float,
) -> tuple[float, float, float]:
    theta0 = float(np.mean(np.concatenate([x_prev, x_next])))
    xp = x_prev - x_prev.mean()
    xn = x_next - x_next.mean()
    denom = float(np.sum(xp**2))
    a0 = float(np.sum(xp * xn) / denom) if denom > 0 else 0.99
    a0 = float(np.clip(a0, 1e-8, 0.9999999))
    dt_arr = np.asarray(dt, dtype=float)
    dt_typical = float(np.median(dt_arr)) if dt_arr.ndim else float(dt_arr)
    kappa0 = float(max(-np.log(a0) / dt_typical, 1e-7))
    mean0, _ = ou_transition_moments(x_prev, kappa0, theta0, 1.0, dt)
    residual = x_next - mean0
    innovation_var = float(np.var(residual, ddof=1)) if len(residual) > 1 else 1e-8
    unit_var = -np.expm1(-2.0 * kappa0 * dt_arr) / (2.0 * kappa0)
    scale_factor = float(np.mean(unit_var))
    sigma0 = float(np.sqrt(max(innovation_var / max(scale_factor, 1e-16), 1e-12)))
    return kappa0, theta0, sigma0


def fit_ou_transitions(
    x_prev: np.ndarray,
    x_next: np.ndarray,
    dt: np.ndarray | float = 1 / 252,
) -> dict:
    """Exact-transition MLE for arbitrary observed transition pairs."""
    x_prev = np.asarray(x_prev, dtype=float)
    x_next = np.asarray(x_next, dtype=float)
    if (
        x_prev.ndim != 1
        or x_next.ndim != 1
        or len(x_prev) != len(x_next)
        or len(x_prev) < 29
        or not np.isfinite(x_prev).all()
        or not np.isfinite(x_next).all()
    ):
        return _failure_result()
    dt_arr = np.asarray(dt, dtype=float)
    if dt_arr.ndim > 0 and dt_arr.size not in (1, len(x_prev)):
        return _failure_result()
    if not np.isfinite(dt_arr).all() or np.any(dt_arr <= 0):
        return _failure_result()

    kappa0, theta0, sigma0 = _initial_guess_transitions(x_prev, x_next, dt)

    def nll(params: np.ndarray) -> float:
        log_kappa, theta, log_sigma = params
        kappa = float(np.exp(log_kappa))
        sigma = float(np.exp(log_sigma))
        try:
            mean, variance = ou_transition_moments(
                x_prev, kappa, float(theta), sigma, dt
            )
        except ValueError:
            return 1e100
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            return 1e100
        value = -float(
            np.sum(stats.norm.logpdf(x_next, loc=mean, scale=np.sqrt(variance)))
        )
        return value if np.isfinite(value) else 1e100

    try:
        result = optimize.minimize(
            nll,
            x0=np.array([np.log(kappa0), theta0, np.log(sigma0)]),
            method="L-BFGS-B",
            bounds=[(-16.0, 12.0), (None, None), (-20.0, 10.0)],
            options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 10000},
        )
        log_kappa, theta, log_sigma = result.x
        kappa = float(np.exp(log_kappa))
        sigma = float(np.exp(log_sigma))
        log_likelihood = float(-result.fun)
        optimizer_converged = bool(
            result.success
            and np.isfinite(log_likelihood)
            and np.isfinite(kappa)
            and kappa > 0
            and np.isfinite(sigma)
            and sigma > 0
        )
        if not optimizer_converged:
            return _failure_result()
        return {
            "kappa": kappa,
            "theta": float(theta),
            "sigma": sigma,
            "log_likelihood": log_likelihood,
            "optimizer_converged": True,
            "converged": True,
            "mean_reversion_detected": bool(kappa > MEAN_REVERSION_DETECTION_KAPPA),
            "half_life_years": float(np.log(2.0) / kappa),
            "model_version": OU_MODEL_VERSION,
        }
    except Exception:
        return _failure_result()


def fit_ou(prices: np.ndarray, dt: np.ndarray | float = 1 / 252) -> dict:
    """Compatibility fit for a contiguous vector of log-price levels."""
    x = np.asarray(prices, dtype=float)
    if x.ndim != 1 or len(x) < 30 or not np.isfinite(x).all():
        return _failure_result()
    dt_pairs = dt
    if np.asarray(dt).ndim > 0 and np.asarray(dt).size == len(x):
        dt_pairs = np.asarray(dt)[1:]
    return fit_ou_transitions(x[:-1], x[1:], dt_pairs)


def forecast_ou_var_es(
    kappa: float,
    theta: float,
    sigma: float,
    current_price: float,
    alpha: float,
    dt: float = 1 / 252,
) -> tuple[float, float]:
    """One-step positive-loss VaR/ES for the exact Gaussian OU log return."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1)")
    mean_next, variance = ou_transition_moments(
        current_price, kappa, theta, sigma, dt
    )
    mu_r = float(mean_next - current_price)
    sigma_r = float(np.sqrt(variance))
    z = float(stats.norm.ppf(alpha))
    var = -(mu_r + sigma_r * z)
    es = -(mu_r - sigma_r * stats.norm.pdf(z) / alpha)
    return float(var), float(es)


def aligned_ou_transition_pairs(
    prices: pd.Series,
    returns: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact (log P_{t-1}, log P_t) pairs for retained returns.

    Only P_t must remain present in ``prices``.  P_{t-1} is reconstructed from
    the retained log return itself, which is exact by definition and avoids the
    compressed-index error that occurs when missing-return dates are removed.
    """
    if not prices.index.is_monotonic_increasing or not returns.index.is_monotonic_increasing:
        raise ValueError("prices and returns must have sorted indices")
    current = prices.reindex(returns.index).to_numpy(dtype=float)
    observed = returns.to_numpy(dtype=float)
    if (
        not np.isfinite(current).all()
        or np.any(current <= 0)
        or not np.isfinite(observed).all()
    ):
        raise ValueError("retained returns must map to positive finite current prices")
    x_next = np.log(current)
    x_prev = x_next - observed
    if not np.isfinite(x_prev).all():
        raise ValueError("reconstructed OU conditioning states are non-finite")
    return x_prev, x_next


def rolling_ou_var_es(
    prices: pd.Series,
    returns: pd.Series,
    window: int = 1000,
    alphas: list | None = None,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    """Rolling exact-OU forecasts using the same transition window as returns."""
    if alphas is None:
        alphas = [0.01, 0.05]
    if len(returns) <= window:
        raise ValueError(f"Series length ({len(returns)}) must exceed window ({window})")

    x_prev, x_next = aligned_ou_transition_pairs(prices, returns)
    r = returns.to_numpy(dtype=float)
    dates = returns.index
    n_forecasts = len(r) - window
    rows = []
    n_failures = 0
    n_weak = 0

    for i in range(n_forecasts):
        target = i + window
        fitted = fit_ou_transitions(x_prev[i:target], x_next[i:target], dt)
        row = {
            "date": dates[target],
            "actual_return": r[target],
            "estimation_failed": False,
            "ou_model_version": OU_MODEL_VERSION,
            "ou_kappa": fitted["kappa"],
            "ou_half_life_years": fitted["half_life_years"],
            "ou_mean_reversion_detected": fitted["mean_reversion_detected"],
            "ou_optimizer_converged": fitted["optimizer_converged"],
        }
        if not fitted["optimizer_converged"]:
            n_failures += 1
            row["estimation_failed"] = True
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"] = np.nan
        else:
            if not fitted["mean_reversion_detected"]:
                n_weak += 1
            for alpha in alphas:
                var, es = forecast_ou_var_es(
                    fitted["kappa"], fitted["theta"], fitted["sigma"],
                    x_prev[target], alpha, dt
                )
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"] = es
        rows.append(row)

    print(
        f"  OU: {n_failures}/{n_forecasts} numerical failures; "
        f"{n_weak}/{n_forecasts} weak-mean-reversion windows retained."
    )
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dt = 1 / 252
    kappa_true, theta_true, sigma_true = 2.0, 4.5, 0.3
    n = 4000
    x = np.empty(n)
    x[0] = theta_true
    var = sigma_true**2 * (-np.expm1(-2 * kappa_true * dt)) / (2 * kappa_true)
    decay = np.exp(-kappa_true * dt)
    for t in range(1, n):
        mean = theta_true + (x[t - 1] - theta_true) * decay
        x[t] = mean + np.sqrt(var) * rng.normal()
    fitted = fit_ou(x, dt)
    assert fitted["optimizer_converged"]
