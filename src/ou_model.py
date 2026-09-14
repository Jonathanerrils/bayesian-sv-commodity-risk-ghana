"""Exact Ornstein-Uhlenbeck (OU) mean-reversion benchmark.

The model is fitted to log-price levels.  For X_t following

    dX_t = kappa * (theta - X_t) dt + sigma dW_t,

the exact discrete transition over an interval dt is

    X_{t+dt} | X_t ~ Normal(
        theta + (X_t - theta) exp(-kappa dt),
        sigma^2 (1 - exp(-2 kappa dt)) / (2 kappa),
    ).

Using the exact transition avoids the Euler-Maruyama approximation that was
previously inconsistent with the manuscript's claim of exact OU estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

OU_MODEL_VERSION = "ou-exact-v2"


def ou_transition_moments(
    x_prev: np.ndarray | float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float = 1 / 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact OU conditional mean and variance."""
    if kappa <= 0 or sigma <= 0 or dt <= 0:
        raise ValueError("kappa, sigma and dt must be positive")
    x_prev = np.asarray(x_prev, dtype=float)
    decay = np.exp(-kappa * dt)
    mean = theta + (x_prev - theta) * decay
    # expm1 is numerically stable when kappa*dt is very small.
    variance = sigma**2 * (-np.expm1(-2.0 * kappa * dt)) / (2.0 * kappa)
    return mean, np.broadcast_to(variance, np.shape(mean)).astype(float)


def _initial_guess(x: np.ndarray, dt: float) -> tuple[float, float, float]:
    """AR(1)-based starting values for exact OU maximum likelihood."""
    x0 = x[:-1]
    x1 = x[1:]
    theta0 = float(np.mean(x))
    denom = float(np.sum((x0 - x0.mean()) ** 2))
    if denom > 0:
        a0 = float(np.sum((x0 - x0.mean()) * (x1 - x1.mean())) / denom)
    else:
        a0 = 0.99
    a0 = float(np.clip(a0, 1e-6, 0.999999))
    kappa0 = float(max(-np.log(a0) / dt, 1e-3))
    residual = x1 - (theta0 + (x0 - theta0) * np.exp(-kappa0 * dt))
    innovation_var = float(np.var(residual, ddof=1)) if len(residual) > 1 else 1e-8
    scale_factor = -np.expm1(-2.0 * kappa0 * dt) / (2.0 * kappa0)
    sigma0 = float(np.sqrt(max(innovation_var / max(scale_factor, 1e-16), 1e-12)))
    return kappa0, theta0, sigma0


def fit_ou(prices: np.ndarray, dt: float = 1 / 252) -> dict:
    """Fit the exact OU transition density by numerical maximum likelihood."""
    x = np.asarray(prices, dtype=float)
    if len(x) < 30 or not np.isfinite(x).all() or dt <= 0:
        return {
            "kappa": np.nan,
            "theta": np.nan,
            "sigma": np.nan,
            "log_likelihood": np.nan,
            "converged": False,
            "model_version": OU_MODEL_VERSION,
        }

    kappa0, theta0, sigma0 = _initial_guess(x, dt)

    def neg_log_likelihood(params: np.ndarray) -> float:
        log_kappa, theta, log_sigma = params
        kappa = float(np.exp(log_kappa))
        sigma = float(np.exp(log_sigma))
        try:
            mean, variance = ou_transition_moments(
                x[:-1], kappa=kappa, theta=float(theta), sigma=sigma, dt=dt
            )
        except ValueError:
            return 1e100
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            return 1e100
        ll = stats.norm.logpdf(x[1:], loc=mean, scale=np.sqrt(variance))
        nll = -float(np.sum(ll))
        return nll if np.isfinite(nll) else 1e100

    try:
        result = optimize.minimize(
            neg_log_likelihood,
            x0=np.array([np.log(kappa0), theta0, np.log(sigma0)]),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0), (None, None), (-20.0, 10.0)],
            options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 10000},
        )
        log_kappa, theta, log_sigma = result.x
        kappa = float(np.exp(log_kappa))
        sigma = float(np.exp(log_sigma))
        log_likelihood = float(-result.fun)

        # kappa < 0.01 implies an annual half-life above roughly 69 years and
        # is treated as no economically detectable mean reversion, as before.
        converged = bool(
            result.success
            and np.isfinite(log_likelihood)
            and kappa > 0.01
            and sigma > 0
        )
        return {
            "kappa": kappa,
            "theta": float(theta),
            "sigma": sigma,
            "log_likelihood": log_likelihood,
            "converged": converged,
            "model_version": OU_MODEL_VERSION,
        }
    except Exception:
        return {
            "kappa": np.nan,
            "theta": np.nan,
            "sigma": np.nan,
            "log_likelihood": np.nan,
            "converged": False,
            "model_version": OU_MODEL_VERSION,
        }


def forecast_ou_var_es(
    kappa: float,
    theta: float,
    sigma: float,
    current_price: float,
    alpha: float,
    dt: float = 1 / 252,
) -> tuple[float, float]:
    """One-step VaR/ES for the exact OU log-return distribution.

    If X is log price, the next log return is X_{t+dt}-X_t.  Under the exact
    transition it is Gaussian with mean

        (theta - X_t) * (1 - exp(-kappa dt))

    and variance sigma^2(1-exp(-2*kappa*dt))/(2*kappa).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1)")
    mean_next, variance = ou_transition_moments(
        current_price, kappa=kappa, theta=theta, sigma=sigma, dt=dt
    )
    mu_r = float(mean_next - current_price)
    sigma_r = float(np.sqrt(variance))

    z_alpha = float(stats.norm.ppf(alpha))
    var = -(mu_r + sigma_r * z_alpha)
    es = -(mu_r - sigma_r * stats.norm.pdf(z_alpha) / alpha)
    return float(var), float(es)


def rolling_ou_var_es(
    prices: pd.Series,
    returns: pd.Series,
    window: int = 1000,
    alphas: list | None = None,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    """Rolling walk-forward exact-OU VaR/ES on log-price levels."""
    if alphas is None:
        alphas = [0.01, 0.05]

    log_prices = np.log(prices.to_numpy(dtype=float))
    return_arr = returns.to_numpy(dtype=float)
    dates = returns.index
    n = len(log_prices)
    if n != len(return_arr):
        raise ValueError("prices and returns must be aligned to the same length")
    if n <= window:
        raise ValueError(f"Series length ({n}) must exceed window ({window})")

    n_forecasts = n - window
    results = []
    n_failures = 0

    for i in range(n_forecasts):
        train_logp = log_prices[i : i + window]
        current_logp = log_prices[i + window - 1]
        actual_return = return_arr[i + window]
        forecast_date = dates[i + window]

        row = {
            "date": forecast_date,
            "actual_return": actual_return,
            "estimation_failed": False,
            "ou_model_version": OU_MODEL_VERSION,
        }
        fitted = fit_ou(train_logp, dt=dt)

        if not fitted["converged"]:
            n_failures += 1
            row["estimation_failed"] = True
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"] = np.nan
        else:
            for alpha in alphas:
                var, es = forecast_ou_var_es(
                    fitted["kappa"],
                    fitted["theta"],
                    fitted["sigma"],
                    current_logp,
                    alpha,
                    dt,
                )
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"] = es

        results.append(row)
        if (i + 1) % 250 == 0:
            print(f"  OU [{i + 1}/{n_forecasts}] failures: {n_failures}")

    df = pd.DataFrame(results).set_index("date")
    if n_failures:
        pct_fail = 100.0 * n_failures / n_forecasts
        print(
            f"  OU: {n_failures}/{n_forecasts} failures ({pct_fail:.1f}%) -- "
            "windows with no economically detectable mean reversion."
        )
    else:
        print(f"  OU: all {n_forecasts} steps converged.")
    return df


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dt = 1 / 252
    kappa_true, theta_true, sigma_true = 2.0, 4.5, 0.3
    n = 4000
    x = np.empty(n)
    x[0] = theta_true
    mean_var = sigma_true**2 * (-np.expm1(-2 * kappa_true * dt)) / (2 * kappa_true)
    sd = np.sqrt(mean_var)
    decay = np.exp(-kappa_true * dt)
    for t in range(1, n):
        mean = theta_true + (x[t - 1] - theta_true) * decay
        x[t] = mean + sd * rng.normal()

    fitted = fit_ou(x, dt=dt)
    print(fitted)
    assert fitted["converged"]
    var_01, es_01 = forecast_ou_var_es(
        fitted["kappa"], fitted["theta"], fitted["sigma"], x[-1], 0.01, dt
    )
    var_05, es_05 = forecast_ou_var_es(
        fitted["kappa"], fitted["theta"], fitted["sigma"], x[-1], 0.05, dt
    )
    assert es_01 > var_01 > var_05
    assert es_05 > var_05
