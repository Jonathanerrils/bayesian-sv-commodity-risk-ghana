"""Posterior transfer smoke tests; synthetic InferenceData, no MCMC."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az
import numpy as np
import pytest

from staged_posterior_artifact import save_posterior_artifact, load_posterior_artifact
from sv_model import MODEL_VERSION

DATA = np.array([.1, -.2, .3])
ATTEMPTS = ({"chains": 4, "tune": 1000, "draws": 1000},
            {"chains": 4, "tune": 2000, "draws": 2000})


def params(provenance="campaign-A"):
    return dict(variant="SV-t", target_accept=.99, random_seed=42,
                attempts=ATTEMPTS, provenance=provenance)


def test_round_trip_full_posterior(tmp_path):
    trace = az.from_dict(posterior={
        "mu": np.zeros((2, 3)), "phi": np.ones((2, 3)) * .8,
        "sigma_eta": np.ones((2, 3)) * .1,
        "h": np.zeros((2, 3, len(DATA)))})
    fit = {"converged": True, "trace": trace, "variant": "SV-t",
           "model_version": MODEL_VERSION, "T": len(DATA),
           "mean_return": 0., "max_rhat": 1.0, "min_ess": 500,
           "n_divergences": 0, "accepted_attempt": 1,
           "mcmc_attempts": [{"attempt": 1, "converged": True}]}
    save_posterior_artifact(tmp_path, fit, DATA, **params())
    restored = load_posterior_artifact(tmp_path, DATA, **params())
    assert restored["accepted_attempt"] == 1
    np.testing.assert_array_equal(restored["trace"].posterior["phi"].values,
                                  trace.posterior["phi"].values)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_posterior_artifact(tmp_path, DATA, **params("campaign-B"))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_posterior_artifact(tmp_path, DATA + .01, **params())


def test_failed_fit_cannot_publish_posterior(tmp_path):
    with pytest.raises(ValueError, match="only converged"):
        save_posterior_artifact(tmp_path, {"converged": False, "trace": None},
                                DATA, **params())


def test_corrupted_posterior_is_rejected(tmp_path):
    trace = az.from_dict(posterior={"mu": np.zeros((2, 3))})
    fit = {"converged": True, "trace": trace, "variant": "SV-t",
           "model_version": MODEL_VERSION, "T": len(DATA),
           "mean_return": 0., "max_rhat": 1.0, "min_ess": 500,
           "n_divergences": 0, "accepted_attempt": 1}
    save_posterior_artifact(tmp_path, fit, DATA, **params())
    with (tmp_path / "posterior.nc").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_posterior_artifact(tmp_path, DATA, **params())
