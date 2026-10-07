import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sv_model import (
    MODEL_VERSION,
    RESAMPLE_ESS_FRACTION,
    build_sv_gaussian,
    build_sv_t,
)


def _free_names(model):
    return {rv.name for rv in model.free_RVs}


def test_production_candidate_version_and_ess_threshold_are_frozen():
    assert MODEL_VERSION == "sv-filter-v6-gaussian-logitphi-tau-ess50"
    assert RESAMPLE_ESS_FRACTION == 0.5


def test_gaussian_uses_logit_phi_with_original_sigma_eta_geometry():
    model = build_sv_gaussian(np.zeros(8, dtype=float), 8)
    named = set(model.named_vars)
    free = _free_names(model)
    assert "phi_logit" in free
    assert "phi_raw" in named
    assert "phi_raw" not in free
    assert "phi_raw_transformed_prior" in named
    assert "sigma_eta" in free
    assert "stationary_sd" not in named
    assert "sigma_eta_transformed_prior" not in named
    assert "h0_std" in free


def test_student_t_uses_stationary_scale_tau_geometry():
    model = build_sv_t(np.zeros(8, dtype=float), 8)
    named = set(model.named_vars)
    free = _free_names(model)
    assert "stationary_sd" in free
    assert "sigma_eta" in named
    assert "sigma_eta" not in free
    assert "sigma_eta_transformed_prior" in named
    assert "h0_std" in free
