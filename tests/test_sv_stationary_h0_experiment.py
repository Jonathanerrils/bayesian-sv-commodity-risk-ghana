import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_model import MODEL_VERSION, build_sv_t


def test_stationary_h0_experiment_is_explicitly_versioned():
    assert MODEL_VERSION == "sv-filter-exp-stationary-h0-centered"


def test_sv_t_uses_direct_h0_parameterization():
    returns = np.zeros(8, dtype=float)
    model = build_sv_t(returns, len(returns))
    named = set(model.named_vars)
    assert "h0" in named
    assert "h0_std" not in named
    assert "eta" in named
    assert "phi" in named
    assert "sigma_eta" in named
