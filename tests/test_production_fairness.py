import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sv_model
from production_runner import common_valid_dates, mcmc_policy_label, run_key


def test_fit_sv_adaptive_stops_after_first_converged(monkeypatch):
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append((kwargs["chains"], kwargs["tune"], kwargs["draws"]))
        return {
            "trace": object(),
            "converged": True,
            "max_rhat": 1.005,
            "min_ess": 500.0,
            "n_divergences": 0,
        }

    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    result = sv_model.fit_sv_adaptive(np.zeros(20), variant="SV-t")

    assert calls == [(4, 1000, 1000)]
    assert result["accepted_attempt"] == 1
    assert len(result["mcmc_attempts"]) == 1


def test_fit_sv_adaptive_escalates_and_never_accepts_weak_trace(monkeypatch):
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append((kwargs["chains"], kwargs["tune"], kwargs["draws"]))
        if len(calls) == 1:
            return {
                "trace": object(),
                "converged": False,
                "max_rhat": 1.02,
                "min_ess": 250.0,
                "n_divergences": 0,
            }
        return {
            "trace": object(),
            "converged": True,
            "max_rhat": 1.007,
            "min_ess": 550.0,
            "n_divergences": 0,
        }

    monkeypatch.setattr(sv_model, "fit_sv", fake_fit)
    result = sv_model.fit_sv_adaptive(np.zeros(20), variant="SV-t-Leverage")

    assert calls == [(4, 1000, 1000), (4, 2000, 2000)]
    assert result["accepted_attempt"] == 2
    assert result["converged"] is True
    assert result["mcmc_attempts"][0]["converged"] is False


def test_common_valid_dates_intersects_model_availability():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")

    def frame(failed_at=None, nan_at=None):
        df = pd.DataFrame(
            {
                "actual_return": [0.0] * 5,
                "var_0.01": [0.02] * 5,
                "es_0.01": [0.03] * 5,
                "var_0.05": [0.01] * 5,
                "es_0.05": [0.02] * 5,
                "estimation_failed": [False] * 5,
            },
            index=idx,
        )
        if failed_at is not None:
            df.loc[idx[failed_at], "estimation_failed"] = True
        if nan_at is not None:
            df.loc[idx[nan_at], "var_0.01"] = np.nan
        return df

    forecasts = {
        ("gold", "GARCH-t"): frame(failed_at=1),
        ("gold", "EGARCH-t"): frame(nan_at=3),
        ("gold", "SV-t"): frame(),
    }

    common = common_valid_dates(forecasts, "gold")
    assert list(common) == [idx[0], idx[2], idx[4]]


def test_run_key_namespaces_mcmc_policy():
    key = run_key(1000, 42, 20000)
    assert mcmc_policy_label() in key
    assert "4c1000t1000d" in key
    assert "4c2000t2000d" in key
