import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sv_model


def _transition(n=200):
    state = {
        "variant": "SV-Gaussian",
        "mean_return": 0.0,
        "mu": np.full(n, -8.0),
        "phi": np.full(n, 0.97),
        "sigma_eta": np.full(n, 0.2),
        "h": np.linspace(-10.0, -6.0, n),
        "weights": np.full(n, 1.0 / n),
        "particle_id": np.arange(n, dtype=np.int64),
    }
    return sv_model._transition_filter_state(state, np.random.default_rng(11))


def test_high_ess_update_can_skip_resampling(monkeypatch):
    transition = _transition()
    n = len(transition["h"])
    monkeypatch.setattr(
        sv_model,
        "_observation_loglik",
        lambda transition, actual_return: np.zeros(n),
    )
    new_state, ess = sv_model.update_filter_state(
        transition,
        actual_return=0.0,
        rng=np.random.default_rng(12),
        resample_threshold=0.5,
    )
    assert np.isclose(ess, n)
    assert not new_state["resampled"]
    assert np.array_equal(new_state["particle_id"], transition["particle_id"])
    assert np.allclose(new_state["weights"], 1.0 / n)


def test_low_ess_update_resamples_and_resets_weights(monkeypatch):
    transition = _transition()
    n = len(transition["h"])
    loglik = np.full(n, -100.0)
    loglik[0] = 0.0
    monkeypatch.setattr(
        sv_model,
        "_observation_loglik",
        lambda transition, actual_return: loglik,
    )
    new_state, ess = sv_model.update_filter_state(
        transition,
        actual_return=-0.2,
        rng=np.random.default_rng(13),
        resample_threshold=0.5,
    )
    assert ess < 0.5 * n
    assert new_state["resampled"]
    assert np.allclose(new_state["weights"], 1.0 / n)
    assert np.unique(new_state["particle_id"]).size < n


def test_previous_weights_enter_next_bayes_update(monkeypatch):
    transition = _transition(n=4)
    transition["weights"] = np.array([0.7, 0.1, 0.1, 0.1])
    monkeypatch.setattr(
        sv_model,
        "_observation_loglik",
        lambda transition, actual_return: np.zeros(4),
    )
    new_state, _ = sv_model.update_filter_state(
        transition,
        actual_return=0.0,
        rng=np.random.default_rng(14),
        resample_threshold=0.01,
    )
    assert not new_state["resampled"]
    assert np.allclose(new_state["weights"], transition["weights"])


def test_predictive_mixture_uses_particle_weights(monkeypatch):
    n = 2
    transition = {
        "variant": "SV-Gaussian",
        "mean_return": 0.0,
        "h": np.array([-20.0, 0.0]),
        "eta": np.zeros(n),
        "weights": np.array([1.0, 0.0]),
    }
    pred = sv_model._predictive_returns(
        transition, 1000, np.random.default_rng(15)
    )
    assert np.std(pred) < 0.001
