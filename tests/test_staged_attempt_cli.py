"""Mocked staged attempt CLI tests; no PyMC sampling."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import run_staged_sv_attempt as stage


def test_stage_cli_failed_attempt_writes_outcome(monkeypatch, tmp_path):
    dates = pd.date_range("2025-01-01", periods=8)
    monkeypatch.setattr(stage, "load_all_returns", lambda verbose=False: {
        "gold": pd.Series(np.arange(8, dtype=float) / 100, index=dates)})
    calls = []
    def fake_attempt(returns, **kw):
        calls.append((len(returns), kw["random_seed"], kw["attempt_no"]))
        return {"converged": False}, "needs_escalation"
    monkeypatch.setattr(stage, "run_attempt", fake_attempt)
    monkeypatch.setattr(stage, "save_posterior_artifact",
                        lambda *args, **kw: (_ for _ in ()).throw(
                            AssertionError("should not save failed posterior")))
    monkeypatch.setattr(sys, "argv", [
        "run_staged_sv_attempt.py", "--commodity", "gold", "--model", "SV-t",
        "--start-block", "1", "--attempt", "1", "--window", "3",
        "--refit-every", "2", "--provenance", "campaign-A",
        "--output-dir", str(tmp_path)])
    stage.main()
    assert calls == [(3, 44, 1)]
    outcome = json.loads((tmp_path / "stage-outcome.json").read_text())
    assert outcome["outcome"] == "needs_escalation"
    assert outcome["block_id"] == 1
