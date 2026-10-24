"""CLI restoration smoke test: restored posterior reaches canonical shard runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import run_production_shard as runner
import staged_posterior_artifact as artifact


def test_cli_restores_posterior_without_refitting(monkeypatch, tmp_path):
    dates = pd.date_range("2025-01-01", periods=8)
    data = pd.Series(np.linspace(.01, .08, 8), index=dates)
    monkeypatch.setattr(runner, "load_all_returns",
                        lambda verbose=False: {"gold": data})
    restored = {"converged": True, "variant": "SV-t", "T": 3,
                "model_version": runner.MODEL_VERSION, "trace": object()}
    calls = []
    def fake_restore(directory, returns, **kw):
        calls.append(("restore", len(returns), kw["random_seed"]))
        return restored
    def fake_shard(*args, **kw):
        calls.append(("shard", kw["staged_fit"] is restored))
        return tmp_path / "output.csv"
    monkeypatch.setattr(artifact, "load_posterior_artifact", fake_restore)
    monkeypatch.setattr(runner, "run_sv_shard", fake_shard)
    monkeypatch.setattr(sys, "argv", [
        "run_production_shard.py", "--kind", "sv", "--commodity", "gold",
        "--model", "SV-t", "--window", "3", "--refit-every", "2",
        "--n-blocks", "1", "--posterior-artifact", str(tmp_path)])
    runner.main()
    assert calls == [("restore", 3, 42), ("shard", True)]
