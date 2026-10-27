"""Exercise adaptive checkpoint reuse without running NUTS."""
import sys
from pathlib import Path

import arviz as az
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sv_model


def test_adaptive_reuses_completed_attempt(monkeypatch, tmp_path):
    trace = az.from_dict(
        posterior={"mu": np.ones((2, 5))},
        sample_stats={"diverging": np.zeros((2, 5), dtype=bool)},
    )
    calls = []
    def fake_fit(returns, **kwargs):
        calls.append(kwargs)
        return {
            "trace": trace, "converged": True,
            "max_rhat": 1.0, "min_ess": 500.0, "n_divergences": 0,
            "rhat_by_var": {"mu": 1.0}, "ess_by_var": {"mu": 500.0},
        }
    def fake_diagnostics(trace_arg, chains):
        assert chains == 2
        return {
            "converged": True, "max_rhat": 1.0,
            "min_ess": 500.0, "n_divergences": 0,
            "rhat_by_var": {"mu": 1.0}, "ess_by_var": {"mu": 500.0},
        }
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    monkeypatch.setattr(sv_model, "_structural_diagnostics", fake_diagnostics)
    kwargs = dict(
        variant="SV-t", target_accept=0.99, random_seed=7,
        attempts=({"chains": 2, "draws": 5, "tune": 5},),
        attempt_checkpoint_dir=tmp_path,
    )
    first = sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    second = sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    assert first["accepted_attempt"] == second["accepted_attempt"] == 1
    assert len(calls) == 1


def test_adaptive_without_checkpoint_still_refits(monkeypatch):
    calls = []
    def fake_fit(returns, **kwargs):
        calls.append(kwargs)
        return {"trace": None, "converged": False, "error": "synthetic failure"}
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    kwargs = dict(attempts=({"chains": 2, "draws": 5, "tune": 5},))
    sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    assert len(calls) == 2


def test_nonconverged_completed_attempt_reused_before_escalation(monkeypatch, tmp_path):
    calls = []
    def fake_fit(returns, **kwargs):
        calls.append(kwargs["draws"])
        trace = az.from_dict(
            posterior={"mu": np.ones((2, kwargs["draws"]))},
            sample_stats={"diverging": np.zeros((2, kwargs["draws"]), dtype=bool)},
        )
        converged = kwargs["draws"] == 10
        return {
            "trace": trace, "converged": converged,
            "max_rhat": 1.0 if converged else 1.2,
            "min_ess": 500.0 if converged else 50.0,
            "n_divergences": 0,
            "rhat_by_var": {"mu": 1.0 if converged else 1.2},
            "ess_by_var": {"mu": 500.0 if converged else 50.0},
        }
    def fake_diagnostics(trace_arg, chains):
        # Model a converged second attempt and a nonconverged first attempt.
        draws = trace_arg.posterior.sizes["draw"]
        converged = draws == 10
        return {
            "converged": converged,
            "max_rhat": 1.0 if converged else 1.2,
            "min_ess": 500.0 if converged else 50.0,
            "n_divergences": 0,
            "rhat_by_var": {"mu": 1.0 if converged else 1.2},
            "ess_by_var": {"mu": 500.0 if converged else 50.0},
        }
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    monkeypatch.setattr(sv_model, "_structural_diagnostics", fake_diagnostics)
    kwargs = dict(
        variant="SV-t", target_accept=0.99, random_seed=7,
        attempts=(
            {"chains": 2, "draws": 5, "tune": 5},
            {"chains": 2, "draws": 10, "tune": 10},
        ),
        attempt_checkpoint_dir=tmp_path,
    )
    first = sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    assert first["accepted_attempt"] == 2
    second = sv_model.fit_sv_adaptive(np.array([0.1, 0.2, 0.3]), **kwargs)
    assert second["accepted_attempt"] == 2
    assert len(calls) == 2
    assert len(second["mcmc_attempts"]) == 2
