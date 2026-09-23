import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_sv_refit import _divergence_geometry


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Trace:
    def __init__(self):
        # 1 chain x 4 draws
        self.posterior = {
            "mu": _Values([[-9.0, -9.1, -8.9, -9.0]]),
            "phi": _Values([[0.95, 0.99, 0.96, 0.97]]),
            "sigma_eta": _Values([[0.20, 0.05, 0.22, 0.21]]),
            "nu": _Values([[8.0, 3.0, 9.0, 10.0]]),
            "eta": _Values(
                [[
                    [0.1, -0.2, 0.3],
                    [4.0, -5.0, 3.0],
                    [0.2, 0.1, -0.2],
                    [0.0, 0.2, -0.1],
                ]]
            ),
            "h": _Values(
                [[
                    [-9.0, -8.8, -8.7],
                    [-9.1, -6.0, -5.5],
                    [-8.9, -8.8, -8.9],
                    [-9.0, -9.1, -9.0],
                ]]
            ),
        }
        self.sample_stats = {
            "diverging": _Values([[False, True, False, False]]),
            "tree_depth": _Values([[8, 12, 7, 8]]),
        }


def test_divergence_geometry_reports_counts_and_structural_metrics():
    result = _divergence_geometry({"trace": _Trace()})

    assert result["available"]
    assert result["n_total"] == 4
    assert result["n_divergent"] == 1
    assert np.isclose(result["divergence_rate"], 0.25)

    phi = result["metrics"]["phi"]
    assert phi["divergent"]["n"] == 1
    assert np.isclose(phi["divergent"]["mean"], 0.99)
    assert phi["nondivergent"]["n"] == 3


def test_divergence_geometry_summarizes_latent_extremes():
    result = _divergence_geometry({"trace": _Trace()})

    eta_max = result["metrics"]["eta_max_abs"]
    assert np.isclose(eta_max["divergent"]["mean"], 5.0)

    h_range = result["metrics"]["h_range"]
    assert h_range["divergent"]["mean"] > h_range["nondivergent"]["mean"]


def test_divergence_geometry_handles_missing_trace():
    result = _divergence_geometry({"trace": None})

    assert not result["available"]
    assert result["n_divergent"] == 0
    assert result["metrics"] == {}
