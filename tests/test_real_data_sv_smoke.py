import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import load_all_returns
from sv_model import forecast_sv_var_es, fit_sv


def test_real_gold_data_runs_through_sv_fit_and_forecast():
    """Lightweight integration smoke test on committed real project data.

    This is intentionally *not* a production-quality posterior. It verifies
    the data loader -> PyMC model -> posterior -> one-step predictive path in a
    clean CI environment without spending the compute required by the real
    rolling experiment.
    """
    returns = load_all_returns(verbose=False)["gold"].iloc[-120:].to_numpy()

    fit = fit_sv(
        returns,
        variant="SV-Gaussian",
        chains=1,
        draws=30,
        tune=30,
        target_accept=0.9,
        random_seed=123,
        fast_mode=False,
    )

    assert fit["trace"] is not None, fit.get("error")
    var, es = forecast_sv_var_es(
        fit,
        alpha=0.05,
        n_predictive=2_000,
        random_seed=321,
    )
    assert np.isfinite(var)
    assert np.isfinite(es)
    assert es >= var
