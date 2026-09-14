import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtests import (
    acerbi_szekely,
    apply_bonferroni_reporting,
    bonferroni_adjust_pvalue,
    kupiec_pof,
)


def test_kupiec_zero_violations_is_finite():
    actual = np.zeros(500)
    var = np.ones(500)
    result = kupiec_pof(actual, var, alpha=0.01)
    assert result["n_violations"] == 0
    assert np.isfinite(result["lr_stat"])
    assert 0 <= result["p_value"] <= 1


def test_acerbi_szekely_uses_each_days_es_forecast_and_reports_mc_uncertainty():
    actual = np.array([-0.04, -0.01, -0.08, 0.01])
    var = np.array([0.03, 0.03, 0.05, 0.03])
    es = np.array([0.05, 0.20, 0.10, 0.20])
    alpha = 0.25
    result = acerbi_szekely(actual, var, es, alpha, n_simulations=2_000, random_seed=7)
    hit = actual < -var
    expected = np.sum((actual * hit) / es) / (len(actual) * alpha) + 1.0
    assert np.isclose(result["z2_stat"], expected)
    assert result["mc_n"] == 2_000
    assert 1 / 2001 <= result["p_value"] <= 1
    assert 0 <= result["mc_ci_low"] <= result["mc_ci_high"] <= 1


def test_bonferroni_adjustment_preserves_raw_logic_and_caps_at_one():
    assert np.isclose(bonferroni_adjust_pvalue(0.001, 60), 0.06)
    assert bonferroni_adjust_pvalue(0.2, 60) == 1.0


def test_bonferroni_reporting_does_not_overwrite_raw_pvalues():
    raw = 0.001000123456789
    df = pd.DataFrame(
        {
            "kupiec_p": [raw],
            "cc_p": [0.02],
            "as_p": [0.0001],
            "as_mc_ci_low": [0.00008],
            "as_mc_ci_high": [0.00013],
            "kupiec_passed": [False],
            "cc_passed": [False],
            "as_passed": [False],
        }
    )
    out = apply_bonferroni_reporting(df, family_size=60)
    assert out.loc[0, "kupiec_p"] == raw
    assert np.isclose(out.loc[0, "kupiec_p_bonferroni"], raw * 60)
    assert bool(out.loc[0, "kupiec_passed_bonferroni"])
    assert out.loc[0, "bonferroni_family_size"] == 60
    assert np.isclose(out.loc[0, "bonferroni_alpha"], 0.05 / 60)
    assert out.loc[0, "as_bonferroni_decision"] == "reject"
    assert bool(out.loc[0, "as_bonferroni_decision_stable"])


def test_es_bonferroni_marks_mc_uncertain_when_interval_crosses_cutoff():
    cutoff = 0.05 / 60
    df = pd.DataFrame(
        {
            "as_p": [cutoff],
            "as_mc_ci_low": [cutoff * 0.8],
            "as_mc_ci_high": [cutoff * 1.2],
        }
    )
    out = apply_bonferroni_reporting(df, family_size=60)
    assert out.loc[0, "as_bonferroni_decision"] == "mc-uncertain"
    assert not bool(out.loc[0, "as_bonferroni_decision_stable"])
