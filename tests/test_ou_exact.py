import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ou_model import fit_ou, forecast_ou_var_es, ou_transition_moments


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
    assert fit["converged"]
    assert abs(fit["theta"] - theta_true) < 0.1
    assert abs(fit["sigma"] - sigma_true) / sigma_true < 0.2
    assert abs(fit["kappa"] - kappa_true) / kappa_true < 0.35


def test_exact_ou_var_es_ordering():
    var_01, es_01 = forecast_ou_var_es(2.0, 4.5, 0.25, 4.4, 0.01)
    var_05, es_05 = forecast_ou_var_es(2.0, 4.5, 0.25, 4.4, 0.05)
    assert es_01 > var_01 > var_05
    assert es_05 > var_05
