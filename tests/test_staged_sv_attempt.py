"""Unit tests for one-attempt job boundaries; no real sampling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
import staged_sv_attempt

DATA = np.array([0.1, -0.2, 0.3])
ATTEMPTS = ({"chains": 4, "tune": 1000, "draws": 1000},
            {"chains": 4, "tune": 2000, "draws": 2000})


def call(n, path):
    return staged_sv_attempt.run_attempt(
        DATA, variant="SV-t", target_accept=.99, random_seed=42,
        attempts=ATTEMPTS, attempt_no=n, checkpoint_path=path,
        provenance="campaign-1")


def test_first_failure_then_second_success(monkeypatch, tmp_path):
    seeds = []
    def fake_fit(returns, **kw):
        seeds.append(kw["random_seed"])
        return {"converged": len(seeds) == 2, "trace": object(),
                "max_rhat": 1.04, "min_ess": 100}
    monkeypatch.setattr(staged_sv_attempt, "fit_sv", fake_fit)
    path = tmp_path / "checkpoint.json"
    first, status = call(1, path)
    assert status == "needs_escalation"
    assert path.exists()
    second, status = call(2, path)
    assert status == "complete"
    assert second["accepted_attempt"] == 2
    assert len(second["mcmc_attempts"]) == 2
    assert seeds == [42, 43]


def test_first_success_skips_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(staged_sv_attempt, "fit_sv",
                        lambda returns, **kw: {"converged": True, "trace": object()})
    path = tmp_path / "checkpoint.json"
    fit, status = call(1, path)
    assert status == "complete"
    assert fit["accepted_attempt"] == 1
    assert not path.exists()


def test_second_attempt_requires_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(staged_sv_attempt, "fit_sv",
                        lambda returns, **kw: pytest.fail("must not sample"))
    with pytest.raises(ValueError, match="preceding attempts"):
        call(2, tmp_path / "missing.json")


def test_both_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(staged_sv_attempt, "fit_sv",
                        lambda returns, **kw: {"converged": False, "trace": None})
    path = tmp_path / "checkpoint.json"
    _, first_status = call(1, path)
    final, final_status = call(2, path)
    assert first_status == "needs_escalation"
    assert final_status == "unavailable"
    assert final["accepted_attempt"] is None


def test_rejects_invalid_attempt_number(tmp_path):
    with pytest.raises(ValueError, match="attempt_no"):
        call(3, tmp_path / "checkpoint.json")
