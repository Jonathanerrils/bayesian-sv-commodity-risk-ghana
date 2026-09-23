import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from sv_model import forecast_sv_var_es


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Trace:
    def __init__(self, posterior):
        self.posterior = {name: _Values(values) for name, values in posterior.items()}


def make_fit(variant, rho=None, n=1000):
    posterior = {
        "mu": np.full((1, n), -8.0),
        "phi": np.full((1, n), 0.96),
        "sigma_eta": np.full((1, n), 0.25),
        "h": np.full((1, n, 2), -8.0),
    }
    if variant in ("SV-t", "SV-t-Leverage"):
        posterior["nu"] = np.full((1, n), 8.0)
    if variant in ("SV-Leverage", "SV-t-Leverage"):
        posterior["rho"] = np.full((1, n), rho)

    return {
        "trace": _Trace(posterior),
        "variant": variant,
        "mean_return": 0.0,
    }


class SVLeverageForecastTests(unittest.TestCase):
    def test_gaussian_leverage_rho_zero_matches_base_sv(self):
        base = forecast_sv_var_es(make_fit("SV-Gaussian"), 0.05)
        leverage = forecast_sv_var_es(make_fit("SV-Leverage", rho=0.0), 0.05)
        self.assertEqual(base, leverage)

    def test_student_t_leverage_rho_zero_matches_sv_t(self):
        base = forecast_sv_var_es(make_fit("SV-t"), 0.05)
        leverage = forecast_sv_var_es(make_fit("SV-t-Leverage", rho=0.0), 0.05)
        self.assertEqual(base, leverage)

    def test_nonzero_rho_changes_gaussian_leverage_forecast(self):
        rho_zero = forecast_sv_var_es(make_fit("SV-Leverage", rho=0.0), 0.05)
        rho_negative = forecast_sv_var_es(
            make_fit("SV-Leverage", rho=-0.8), 0.05
        )
        self.assertNotEqual(rho_zero, rho_negative)

    def test_nonzero_rho_changes_student_t_leverage_forecast(self):
        rho_zero = forecast_sv_var_es(
            make_fit("SV-t-Leverage", rho=0.0), 0.05
        )
        rho_negative = forecast_sv_var_es(
            make_fit("SV-t-Leverage", rho=-0.8), 0.05
        )
        self.assertNotEqual(rho_zero, rho_negative)


if __name__ == "__main__":
    unittest.main()
