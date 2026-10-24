"""Publication retry safety, using synthetic posterior data without MCMC."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import arviz as az
import numpy as np
import pytest
from staged_posterior_artifact import save_posterior_artifact, load_posterior_artifact
from sv_model import MODEL_VERSION, DEFAULT_ROLLING_MCMC_ATTEMPTS

DATA=np.array([.01,-.02,.03])
OPTIONS=dict(variant="SV-t",target_accept=.99,random_seed=42,
             attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,provenance="retry-test")

def fit():
    return dict(converged=True,trace=az.from_dict(posterior={"mu":np.zeros((2,3))}),
                variant="SV-t",model_version=MODEL_VERSION,T=3,
                mean_return=float(DATA.mean()),max_rhat=1.0,min_ess=600,
                n_divergences=0,accepted_attempt=1)

def test_repeated_publish_keeps_original_bytes(tmp_path):
    save_posterior_artifact(tmp_path,fit(),DATA,**OPTIONS)
    original=(tmp_path/"posterior.nc").read_bytes()
    save_posterior_artifact(tmp_path,fit(),DATA,**OPTIONS)
    assert (tmp_path/"posterior.nc").read_bytes()==original
    assert load_posterior_artifact(tmp_path,DATA,**OPTIONS)["converged"]

def test_partial_publication_is_rejected(tmp_path):
    (tmp_path/"posterior.nc").write_bytes(b"partial")
    with pytest.raises(ValueError,match="incomplete posterior publication"):
        save_posterior_artifact(tmp_path,fit(),DATA,**OPTIONS)

def test_retry_with_different_training_data_is_rejected(tmp_path):
    save_posterior_artifact(tmp_path,fit(),DATA,**OPTIONS)
    with pytest.raises(ValueError,match="identity mismatch"):
        save_posterior_artifact(tmp_path,fit(),DATA+.01,**OPTIONS)
