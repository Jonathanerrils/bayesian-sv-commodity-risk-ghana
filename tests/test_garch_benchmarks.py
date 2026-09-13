import numpy as np
import pandas as pd

from src.garch_model import (
    _student_t_var_es_multiplier,
    fit_garch,
    forecast_var_es,
    rolling_var_es,
)
from src.production_runner import GARCH_BENCHMARK_SPECS


def test_student_t_unit_variance_quantile_and_es_are_sensible():
    q, es = _student_t_var_es_multiplier(0.01, nu=8.0)
    assert q < 0
    assert es > -q


def test_heavy_tail_benchmark_matrix_is_complete():
    assert GARCH_BENCHMARK_SPECS == {
        "GARCH": ("GARCH", "normal"),
        "GARCH-t": ("GARCH", "t"),
        "EGARCH": ("EGARCH", "normal"),
        "EGARCH-t": ("EGARCH", "t"),
    }


def test_student_t_rolling_forecasts_run_and_es_exceeds_var():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=180, freq="B")
    returns = pd.Series(rng.standard_t(df=7, size=len(idx)) * 0.01, index=idx)

    out = rolling_var_es(
        returns,
        model_type="GARCH",
        distribution="t",
        window=150,
        alphas=[0.05],
    )

    valid = out.loc[~out["estimation_failed"]]
    assert len(valid) > 0
    assert np.isfinite(valid["var_0.05"]).all()
    assert np.isfinite(valid["es_0.05"]).all()
    assert (valid["es_0.05"] >= valid["var_0.05"]).all()


def test_asymmetric_egarch_t_fit_exposes_leverage_and_nu_parameters():
    rng = np.random.default_rng(13)
    returns = rng.standard_t(df=8, size=500) * 0.01
    result = fit_garch(
        returns,
        model_type="EGARCH",
        distribution="t",
    )

    assert result is not None
    assert "gamma[1]" in result.params.index
    assert "nu" in result.params.index

    var, es = forecast_var_es(result, 0.05, distribution="t")
    assert np.isfinite(var)
    assert np.isfinite(es)
    assert es >= var
