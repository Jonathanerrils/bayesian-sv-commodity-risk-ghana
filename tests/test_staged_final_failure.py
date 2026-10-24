"""Final nonconverged staged fits retain the complete forecast calendar."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pandas as pd
import staged_sv_attempt as stage
import run_production_shard as shard


def test_final_failure_generates_unavailable_forecasts(monkeypatch,tmp_path):
    returns=pd.Series(np.array([.01,-.02,.03,.01,.02,-.01,.015]),
                      index=pd.date_range("2025-01-01",periods=7))
    monkeypatch.setattr(shard,"load_all_returns",lambda verbose=False:{"gold":returns})
    attempts=({"chains":4,"tune":1000,"draws":1000},
              {"chains":4,"tune":2000,"draws":2000})
    seeds=[]
    def rejected_fit(data,**kwargs):
        seeds.append(kwargs["random_seed"])
        return {"converged":False,"trace":None,"variant":"SV-t",
                "model_version":stage.MODEL_VERSION,"T":len(data),
                "max_rhat":1.05,"min_ess":100,"n_divergences":0,
                "error":"synthetic nonconvergence"}
    monkeypatch.setattr(stage,"fit_sv",rejected_fit)
    training=returns.iloc[:3].to_numpy()
    checkpoint=tmp_path/"failed.json"
    _,first=stage.run_attempt(training,variant="SV-t",target_accept=.99,
                              random_seed=42,attempts=attempts,attempt_no=1,
                              checkpoint_path=checkpoint,provenance="failure-test")
    assert first=="needs_escalation"
    fit,last=stage.run_attempt(training,variant="SV-t",target_accept=.99,
                               random_seed=42,attempts=attempts,attempt_no=2,
                               checkpoint_path=checkpoint,provenance="failure-test")
    assert last=="unavailable"
    assert seeds==[42,43]
    assert fit["accepted_attempt"] is None
    assert len(fit["mcmc_attempts"])==2
    assert fit["mcmc_attempts"][0]["chains"]==4
    assert fit["mcmc_attempts"][0]["tune"]==1000
    path=shard.run_sv_shard("gold","SV-t",0,1,3,2,20,
                            tmp_path/"forecasts",staged_fit=fit)
    out=pd.read_csv(path)
    assert len(out)==2
    assert out["global_i"].tolist()==[0,1]
    assert out["estimation_failed"].all()
    assert out["var_0.01"].isna().all()
    assert out["es_0.01"].isna().all()
    assert out["mcmc_attempts_json"].notna().iloc[0]
