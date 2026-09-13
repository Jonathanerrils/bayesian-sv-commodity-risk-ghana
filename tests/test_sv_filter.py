import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_model import (
    _np_unit_variance_t_scale,
    _predictive_returns,
    _transition_filter_state,
    predictive_var_es,
    update_filter_state,
)


def base_state(variant="SV-Gaussian", n=500):
    state = {
        "variant": variant,
        "mean_return": 0.0,
        "mu": np.full(n, -8.0),
        "phi": np.full(n, 0.97),
        "sigma_eta": np.full(n, 0.25),
        "h": np.full(n, -8.0),
    }
    if "t" in variant:
        state["nu"] = np.full(n, 6.0)
    if "Leverage" in variant:
        state["rho"] = np.full(n, -0.7)
    return state


def test_student_t_scale_has_unit_variance_correction():
    nu = np.array([3.0, 5.0, 10.0])
    scale = _np_unit_variance_t_scale(nu)
    assert np.all(scale > 0)
    assert np.all(scale < 1)
    assert np.allclose(scale**2 * nu / (nu - 2), 1.0)


def test_daily_filter_update_changes_next_forecast():
    state = base_state()
    rng1 = np.random.default_rng(123)
    transition1 = _transition_filter_state(state, rng1)
    pred1 = _predictive_returns(transition1, 20_000, rng1)
    var1, _ = predictive_var_es(pred1, 0.01)

    updated, _ = update_filter_state(
        transition1, actual_return=-0.12, rng=np.random.default_rng(456)
    )
    rng2 = np.random.default_rng(123)
    transition2 = _transition_filter_state(updated, rng2)
    pred2 = _predictive_returns(transition2, 20_000, rng2)
    var2, _ = predictive_var_es(pred2, 0.01)

    assert not np.isclose(var1, var2)


def test_leverage_parameter_changes_predictive_distribution():
    n = 500
    transition = base_state("SV-Leverage", n)
    transition["eta"] = np.ones(n)
    transition["h"] = np.full(n, -8.0)

    with_leverage = _predictive_returns(
        transition, 30_000, np.random.default_rng(99)
    )

    no_leverage = dict(transition)
    no_leverage["rho"] = np.zeros(n)
    without_leverage = _predictive_returns(
        no_leverage, 30_000, np.random.default_rng(99)
    )

    assert abs(with_leverage.mean() - without_leverage.mean()) > 1e-3
