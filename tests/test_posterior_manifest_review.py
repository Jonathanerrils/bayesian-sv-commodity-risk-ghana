"""Review guards for posterior publication and restoration metadata."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import arviz as az
import numpy as np
import pytest
from staged_posterior_artifact import save_posterior_artifact,load_posterior_artifact
from sv_model import MODEL_VERSION,DEFAULT_ROLLING_MCMC_ATTEMPTS


def test_rejects_manifest_with_missing_convergence_evidence(tmp_path):
    data=np.array([.01,-.02,.03])
    trace=az.from_dict(posterior={"mu":np.zeros((2,3))})
    fit={"converged":True,"trace":trace,"variant":"SV-t",
         "model_version":MODEL_VERSION,"T":3,"mean_return":float(data.mean()),
         "max_rhat":1.0,"min_ess":600,"n_divergences":0,
         "accepted_attempt":1,"mcmc_attempts":[{"attempt":1,"converged":True}]}
    options=dict(variant="SV-t",target_accept=.99,random_seed=42,
                 attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,provenance="review")
    save_posterior_artifact(tmp_path,fit,data,**options)
    manifest=tmp_path/"manifest.json"
    doc=json.loads(manifest.read_text())
    doc["max_rhat"]=None
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="convergence|diagnostic"):
        load_posterior_artifact(tmp_path,data,**options)
