import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_model import MODEL_VERSION, build_sv_t


def test_stationary_scale_tau_experiment_is_explicitly_versioned():
    assert MODEL_VERSION == "sv-filter-exp-stationary-scale-tau"


def test_tau_parameterization_preserves_noncentered_h0_structure():
    model = build_sv_t(np.zeros(8, dtype=float), 8)
    named = set(model.named_vars)
    free = {rv.name for rv in model.free_RVs}

    assert "stationary_sd" in free
    assert "h0_std" in free
    assert "sigma_eta" in named
    assert "sigma_eta" not in free
    assert "sigma_eta_transformed_prior" in named
    assert "eta" in free
    assert "phi_raw" in free
