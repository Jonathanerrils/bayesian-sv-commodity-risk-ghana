import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ou_model
from ou_model import (
    aligned_ou_transition_pairs,
    fit_ou,
    forecast_ou_var_es,
    ou_transition_moments,
)


def test_exact_transition_matches_closed_form():
    x = 4.0
    kappa = 1.7
    theta = 4.5
    sigma = 0.3
    dt = 1 / 252
    mean, var = ou_transition_moments(x, kappa, theta, sigma, dt)
    expected_mean = theta + (x - theta) * np.exp(-kappa * dt)
    expected_var = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    assert np.isclose(float(mean), expected_mean)
    assert np.isclose(float(var), expected_var)


def test_return_pair_alignment_survives_missing_price_gap():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, np.nan, 101.0, 102.0, 103.0], index=dates)
    raw_returns = np.log(prices / prices.shift(1))
    returns = raw_returns.dropna()

    # Only Jan 4 and Jan 5 have valid returns.  The previous retained return
    # date is irrelevant: Jan 4 must condition on Jan 3's price (=101), not on
    # whichever date happens to precede it in the compressed return index.
    x_prev, x_next = aligned_ou_transition_pairs(prices, returns)
    assert returns.index.tolist() == [dates[3], dates[4]]
    assert np.allclose(np.exp(x_prev), [101.0, 102.0])
    assert np.allclose(np.exp(x_next), [102.0, 103.0])
    assert np.allclose(x_next - x_prev, returns.to_numpy())

    # The helper must also work if a caller already compressed prices onto the
    # retained-return index, because x_{t-1} is reconstructed from P_t and r_t.
    compressed_prices = prices.reindex(returns.index)
    xp2, xn2 = aligned_ou_transition_pairs(compressed_prices, returns)
    assert np.allclose(xp2, x_prev)
    assert np.allclose(xn2, x_next)


def test_exact_ou_mle_recovers_synthetic_parameters_reasonably():
    rng = np.random.default_rng(123)
    dt = 1 / 252
    kappa_true, theta_true, sigma_true = 3.0, 4.4, 0.25
    n = 5000
    x = np.empty(n)
    x[0] = theta_true
    decay = np.exp(-kappa_true * dt)
    sd = np.sqrt(sigma_true**2 * (1 - np.exp(-2 * kappa_true * dt)) / (2 * kappa_true))
    for t in range(1, n):
        mean = theta_true + (x[t - 1] - theta_true) * decay
        x[t] = mean + sd * rng.normal()

    fit = fit_ou(x, dt=dt)
    assert fit["optimizer_converged"]
    assert fit["converged"]
    assert fit["mean_reversion_detected"]
    assert abs(fit["theta"] - theta_true) < 0.1
    assert abs(fit["sigma"] - sigma_true) / sigma_true < 0.2
    assert abs(fit["kappa"] - kappa_true) / kappa_true < 0.35


def test_small_kappa_is_valid_estimate_not_estimation_failure(monkeypatch):
    fake = SimpleNamespace(
        success=True,
        x=np.array([np.log(0.005), 4.2, np.log(0.2)]),
        fun=-123.0,
    )
    monkeypatch.setattr(ou_model.optimize, "minimize", lambda *a, **k: fake)
    x = np.linspace(4.0, 4.3, 100)
    fit = fit_ou(x)

    assert fit["optimizer_converged"]
    assert fit["converged"]
    assert not fit["mean_reversion_detected"]
    assert np.isclose(fit["half_life_years"], np.log(2.0) / 0.005)

    var, es = forecast_ou_var_es(
        fit["kappa"], fit["theta"], fit["sigma"], current_price=x[-1], alpha=0.01
    )
    assert np.isfinite(var)
    assert np.isfinite(es)
    assert es > var


def test_exact_ou_var_es_ordering():
    var_01, es_01 = forecast_ou_var_es(2.0, 4.5, 0.25, 4.4, 0.01)
    var_05, es_05 = forecast_ou_var_es(2.0, 4.5, 0.25, 4.4, 0.05)
    assert es_01 > var_01 > var_05
    assert es_05 > var_05
