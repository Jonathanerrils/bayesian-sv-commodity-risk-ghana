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


def test_acerbi_szekely_uses_each_days_es_forecast():
    actual = np.array([-0.04, -0.01, -0.08, 0.01])
    var = np.array([0.03, 0.03, 0.05, 0.03])
    es = np.array([0.05, 0.20, 0.10, 0.20])
    alpha = 0.25
    result = acerbi_szekely(actual, var, es, alpha, n_simulations=2_000, random_seed=7)
    hit = actual < -var
    expected = np.sum((actual * hit) / es) / (len(actual) * alpha) + 1.0
    assert np.isclose(result["z2_stat"], expected)


def test_bonferroni_adjustment_preserves_raw_logic_and_caps_at_one():
    assert np.isclose(bonferroni_adjust_pvalue(0.001, 60), 0.06)
    assert bonferroni_adjust_pvalue(0.2, 60) == 1.0


def test_bonferroni_reporting_adds_adjusted_columns_without_overwriting_raw():
    df = pd.DataFrame(
        {
            "kupiec_p": [0.001],
            "cc_p": [0.02],
            "as_p": [0.0001],
            "kupiec_passed": [False],
            "cc_passed": [False],
            "as_passed": [False],
        }
    )
    out = apply_bonferroni_reporting(df, family_size=60)
    assert out.loc[0, "kupiec_p"] == 0.001
    assert np.isclose(out.loc[0, "kupiec_p_bonferroni"], 0.06)
    assert bool(out.loc[0, "kupiec_passed_bonferroni"])
    assert out.loc[0, "bonferroni_family_size"] == 60
    assert np.isclose(out.loc[0, "bonferroni_alpha"], 0.05 / 60)
