import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtests import acerbi_szekely, kupiec_pof


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
    result = acerbi_szekely(
        actual, var, es, alpha, n_simulations=2_000, random_seed=7
    )
    hit = actual < -var
    expected = np.sum((actual * hit) / es) / (len(actual) * alpha) + 1.0
    assert np.isclose(result["z2_stat"], expected)
