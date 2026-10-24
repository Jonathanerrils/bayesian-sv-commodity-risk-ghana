"""Reject incompatible or tampered staged posterior artifacts before forecast use."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import arviz as az
import numpy as np
import pytest
from staged_posterior_artifact import save_posterior_artifact,load_posterior_artifact
from sv_model import MODEL_VERSION,DEFAULT_ROLLING_MCMC_ATTEMPTS


def test_transfer_rejects_wrong_seed_training_and_tampering(tmp_path):
    training=np.array([.01,-.02,.03])
    trace=az.from_dict(posterior={
        "mu":np.zeros((2,3)),"phi":np.full((2,3),.7),
        "sigma_eta":np.full((2,3),.1),"h":np.zeros((2,3,3)),
        "nu":np.full((2,3),9.)})
    fit={"converged":True,"trace":trace,"variant":"SV-t",
         "model_version":MODEL_VERSION,"T":3,
         "mean_return":float(training.mean()),"max_rhat":1.,
         "min_ess":600,"n_divergences":0,"accepted_attempt":1,
         "mcmc_attempts":[{"attempt":1,"converged":True}]}
    args=dict(variant="SV-t",target_accept=.99,random_seed=42,
              attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,provenance="run-A")
    save_posterior_artifact(tmp_path,fit,training,**args)
    assert load_posterior_artifact(tmp_path,training,**args)["converged"]
    with pytest.raises(ValueError,match="identity mismatch"):
        load_posterior_artifact(tmp_path,training,**{**args,"random_seed":43})
    with pytest.raises(ValueError,match="identity mismatch"):
        load_posterior_artifact(tmp_path,training,**{**args,"provenance":"run-B"})
    with pytest.raises(ValueError,match="identity mismatch"):
        load_posterior_artifact(tmp_path,training+0.001,**args)
    with (tmp_path/"posterior.nc").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError,match="checksum mismatch"):
        load_posterior_artifact(tmp_path,training,**args)
