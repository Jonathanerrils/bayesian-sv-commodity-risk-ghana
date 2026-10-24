"""Synthetic equivalence test for restored posterior forecasts; no MCMC sampling."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import arviz as az
import numpy as np
import pandas as pd
import run_production_shard as runner
from staged_posterior_artifact import save_posterior_artifact,load_posterior_artifact
from sv_model import MODEL_VERSION,DEFAULT_ROLLING_MCMC_ATTEMPTS

def test_restored_posterior_produces_identical_forecast_rows(monkeypatch,tmp_path):
    window=3
    data=pd.Series(np.array([.01,-.02,.03,.015,-.01,.02,.005]),
                   index=pd.date_range("2025-01-01",periods=7))
    monkeypatch.setattr(runner,"load_all_returns",lambda verbose=False:{"gold":data})
    trace=az.from_dict(posterior={
        "mu":np.zeros((2,3)),"phi":np.full((2,3),.7),
        "sigma_eta":np.full((2,3),.1),
        "h":np.zeros((2,3,window)),"nu":np.full((2,3),9.)})
    fit={"converged":True,"trace":trace,"variant":"SV-t",
         "model_version":MODEL_VERSION,"T":window,
         "mean_return":float(data.iloc[:window].mean()),
         "max_rhat":1.,"min_ess":600,"n_divergences":0,
         "accepted_attempt":1,
         "mcmc_attempts":[{"attempt":1,"chains":4,"tune":1000,
                           "draws":1000,"converged":True}]}
    args=("gold","SV-t",0,1,window,2,20)
    direct=runner.run_sv_shard(*args,tmp_path/"direct",staged_fit=fit)
    options=dict(variant="SV-t",target_accept=runner.DEFAULT_TARGET_ACCEPT,
                 random_seed=42,attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
                 provenance="synthetic-equivalence")
    training=data.iloc[:window].to_numpy()
    save_posterior_artifact(tmp_path/"artifact",fit,training,**options)
    restored=load_posterior_artifact(tmp_path/"artifact",training,**options)
    transferred=runner.run_sv_shard(*args,tmp_path/"restored",staged_fit=restored)
    left=pd.read_csv(direct)
    right=pd.read_csv(transferred)
    pd.testing.assert_frame_equal(left,right,check_dtype=False)
    assert len(left)==2
    assert left["global_i"].tolist()==[0,1]
    assert not left["estimation_failed"].any()
