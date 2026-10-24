"""Mocked integration tests: no expensive MCMC runs."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sv_model

DATA = np.array([.1, -.2, .3, -.4])
ATTEMPTS = ({"chains": 4, "tune": 1000, "draws": 1000},
            {"chains": 4, "tune": 2000, "draws": 2000})

def test_adaptive_resume_after_interrupted_second_attempt(monkeypatch, tmp_path):
    seeds = []
    def first_run(returns, **kwargs):
        seeds.append(kwargs["random_seed"])
        if len(seeds) == 2:
            raise KeyboardInterrupt("simulated runner cancellation")
        return {"converged": False, "max_rhat": 1.03, "min_ess": 100,
                "n_divergences": 0, "rhat_by_var": {}, "ess_by_var": {}}
    monkeypatch.setattr(sv_model, "fit_sv", first_run)
    path = tmp_path / "attempt.json"
    with pytest.raises(KeyboardInterrupt):
        sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS, checkpoint_path=path)
    assert path.exists()
    def second_run(returns, **kwargs):
        seeds.append(kwargs["random_seed"])
        return {"converged": True, "trace": object(), "max_rhat": 1.0,
                "min_ess": 600, "n_divergences": 0}
    monkeypatch.setattr(sv_model, "fit_sv", second_run)
    fit = sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS, checkpoint_path=path)
    assert seeds == [42, 43, 43]
    assert fit["accepted_attempt"] == 2
    assert len(fit["mcmc_attempts"]) == 2

def test_default_path_keeps_original_attempt_seeds(monkeypatch):
    seeds = []
    def fake_fit(returns, **kwargs):
        seeds.append(kwargs["random_seed"])
        return {"converged": len(seeds) == 2, "trace": object()}
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    result = sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS)
    assert seeds == [42, 43]
    assert result["accepted_attempt"] == 2

def test_rejects_checkpoint_for_different_returns(monkeypatch, tmp_path):
    monkeypatch.setattr(sv_model, "fit_sv", lambda returns, **kwargs: {
        "converged": False, "max_rhat": 1.05, "min_ess": 100,
        "n_divergences": 0, "rhat_by_var": {}, "ess_by_var": {}})
    path = tmp_path / "attempt.json"
    sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS, checkpoint_path=path)
    with pytest.raises(ValueError, match="identity mismatch"):
        sv_model.fit_sv_adaptive(DATA + .01, attempts=ATTEMPTS, checkpoint_path=path)

def test_all_failed_preserves_last_fit_diagnostics(monkeypatch, tmp_path):
    calls = []
    def fake_fit(returns, **kwargs):
        calls.append(kwargs["random_seed"])
        return {"converged": False, "trace": None, "max_rhat": 1.08,
                "min_ess": 72, "n_divergences": 3,
                "error": "structural convergence failure"}
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    path = tmp_path / "attempt.json"
    result = sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS, checkpoint_path=path)
    assert calls == [42, 43]
    assert result["max_rhat"] == 1.08
    assert result["n_divergences"] == 3
    assert result["error"] == "structural convergence failure"
    assert result["accepted_attempt"] is None
    assert len(result["mcmc_attempts"]) == 2
    assert len(__import__("json").loads(path.read_text())["records"]) == 1

def test_nonfinite_numpy_diagnostics_checkpoint_safely(monkeypatch, tmp_path):
    calls = []
    def fake_fit(returns, **kwargs):
        calls.append(kwargs["random_seed"])
        return {"converged": False, "max_rhat": np.float64("nan"),
                "min_ess": np.float64(100), "n_divergences": np.int64(0),
                "rhat_by_var": {"mu": np.float64("nan"), "phi": np.float64(1.04)},
                "ess_by_var": {"mu": np.float64("inf"), "phi": np.int64(100)}}
    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    path = tmp_path / "attempt.json"
    sv_model.fit_sv_adaptive(DATA, attempts=ATTEMPTS, checkpoint_path=path)
    import json
    doc = json.loads(path.read_text())
    assert doc["records"][0]["rhat_by_var"]["mu"] is None
    assert doc["records"][0]["ess_by_var"]["mu"] is None
    assert doc["records"][0]["rhat_by_var"]["phi"] == 1.04
