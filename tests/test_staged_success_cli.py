"""Synthetic CLI successful posterior handoff without running an MCMC sampler."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import arviz as az
import numpy as np
import pandas as pd
import run_staged_sv_attempt as stage
import run_production_shard as shard
from staged_posterior_artifact import load_posterior_artifact
from sv_model import MODEL_VERSION,DEFAULT_ROLLING_MCMC_ATTEMPTS


def test_cli_success_artifact_can_generate_forecast_without_refit(monkeypatch,tmp_path):
    series=pd.Series(np.array([.01,-.02,.03,.015,-.01,.02,.005]),
                     index=pd.date_range("2025-01-01",periods=7))
    monkeypatch.setattr(stage,"load_all_returns",lambda verbose=False:{"gold":series})
    monkeypatch.setattr(shard,"load_all_returns",lambda verbose=False:{"gold":series})
    window=3
    trace=az.from_dict(posterior={
        "mu":np.zeros((2,3)),"phi":np.full((2,3),.7),
        "sigma_eta":np.full((2,3),.1),
        "h":np.zeros((2,3,window)),"nu":np.full((2,3),9.)})
    def fake_attempt(training,**kwargs):
        assert kwargs["attempt_no"]==1
        return {"converged":True,"trace":trace,"variant":"SV-t",
                "model_version":MODEL_VERSION,"T":window,
                "mean_return":float(np.mean(training)),
                "max_rhat":1.,"min_ess":600,"n_divergences":0,
                "accepted_attempt":1,"mcmc_attempts":[
                    {"attempt":1,"chains":4,"tune":1000,"draws":1000,
                     "converged":True}]},"complete"
    monkeypatch.setattr(stage,"run_attempt",fake_attempt)
    monkeypatch.setattr(sys,"argv",[
        "run_staged_sv_attempt.py","--commodity","gold","--model","SV-t",
        "--start-block","0","--attempt","1","--window","3",
        "--refit-every","2","--provenance","synthetic-success",
        "--output-dir",str(tmp_path)])
    stage.main()
    assert json.loads((tmp_path/"stage-outcome.json").read_text())["outcome"]=="complete"
    training=series.iloc[:window].to_numpy()
    restored=load_posterior_artifact(
        tmp_path/"posterior",training,variant="SV-t",
        target_accept=stage.DEFAULT_TARGET_ACCEPT,random_seed=42,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,provenance="synthetic-success")
    monkeypatch.setattr(shard,"fit_sv_adaptive",lambda *args,**kwargs:
                        (_ for _ in ()).throw(AssertionError("refit prohibited")))
    path=shard.run_sv_shard("gold","SV-t",0,1,3,2,20,
                             tmp_path/"forecast",staged_fit=restored)
    result=pd.read_csv(path)
    assert len(result)==2
    assert not result["estimation_failed"].any()
