import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_production_shard as shard


def test_sv_shard_records_failed_block_and_continues_next_refit(monkeypatch, tmp_path):
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    returns = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)
    monkeypatch.setattr(shard, "load_all_returns", lambda verbose=False: {"cocoa": returns})

    calls = {"n": 0}

    def fake_fit(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "converged": False,
                "accepted_attempt": None,
                "max_rhat": 1.05,
                "min_ess": 80.0,
                "n_divergences": 0,
                "rhat_by_var": {"phi": 1.05},
                "ess_by_var": {"phi": 80.0},
                "mcmc_attempts": [{"attempt": 1, "converged": False}],
            }
        return {
            "converged": True,
            "accepted_attempt": 1,
            "max_rhat": 1.005,
            "min_ess": 650.0,
            "n_divergences": 0,
            "mcmc_attempts": [
                {"attempt": 1, "chains": 4, "tune": 1000, "draws": 1000, "converged": True}
            ],
        }

    monkeypatch.setattr(shard, "fit_sv_adaptive", fake_fit)
    monkeypatch.setattr(shard, "initialize_filter_state", lambda fit: {"h": np.array([0.0])})
    monkeypatch.setattr(
        shard,
        "_transition_filter_state",
        lambda state, rng: {"h": state["h"], "h_next": state["h"]},
    )
    monkeypatch.setattr(
        shard,
        "_predictive_returns",
        lambda transition, n_predictive, rng: np.array([-0.03, -0.01, 0.01, 0.02]),
    )
    monkeypatch.setattr(shard, "predictive_var_es", lambda pred, alpha: (0.02, 0.03))
    monkeypatch.setattr(
        shard,
        "update_filter_state",
        lambda transition, actual_return, rng: ({"h": transition["h_next"]}, 1000.0),
    )

    path = shard.run_sv_shard(
        commodity="cocoa",
        variant="SV-Gaussian",
        start_block=0,
        n_blocks=2,
        window=4,
        refit_every=2,
        predictive_draws=2000,
        output_dir=tmp_path,
    )
    out = pd.read_csv(path)

    assert calls["n"] == 2
    assert out["global_i"].tolist() == [0, 1, 2, 3]

    first = out[out["block_id"] == 0]
    second = out[out["block_id"] == 1]
    assert first["estimation_failed"].astype(bool).all()
    assert first[["var_0.01", "es_0.01", "var_0.05", "es_0.05"]].isna().all().all()
    assert bool(first.iloc[0]["refit"])
    assert not bool(first.iloc[0]["mcmc_converged"])

    assert not second["estimation_failed"].astype(bool).any()
    assert np.isfinite(second[["var_0.01", "es_0.01", "var_0.05", "es_0.05"]]).all().all()
    assert bool(second.iloc[0]["mcmc_converged"])

    failures = list(tmp_path.glob("failure__cocoa__sv-gaussian__block-0.json"))
    assert len(failures) == 1
