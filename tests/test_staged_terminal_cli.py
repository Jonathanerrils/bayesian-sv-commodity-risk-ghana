"""CLI integration for terminal staged failure, using synthetic data only."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import run_staged_sv_attempt as stage
import run_production_shard as shard
from sv_model import MODEL_VERSION


def test_cli_terminal_failure_writes_complete_forecast_block(monkeypatch,tmp_path):
    returns=pd.Series(np.arange(8,dtype=float)/100,
                      index=pd.date_range("2025-01-01",periods=8))
    monkeypatch.setattr(stage,"load_all_returns",lambda verbose=False:{"gold":returns})
    monkeypatch.setattr(shard,"load_all_returns",lambda verbose=False:{"gold":returns})
    def final_failure(training,**kwargs):
        assert kwargs["attempt_no"]==2
        return {"converged":False,"accepted_attempt":None,
                "variant":"SV-t","model_version":MODEL_VERSION,"T":len(training),
                "mcmc_attempts":[{"attempt":1,"chains":4,"tune":1000,"draws":1000},
                                 {"attempt":2,"chains":4,"tune":2000,"draws":2000}],
                "max_rhat":1.05,"min_ess":100,"n_divergences":0,
                "error":"synthetic failure"},"unavailable"
    monkeypatch.setattr(stage,"run_attempt",final_failure)
    monkeypatch.setattr(sys,"argv",[
        "run_staged_sv_attempt.py","--commodity","gold","--model","SV-t",
        "--start-block","1","--attempt","2","--window","3",
        "--refit-every","2","--provenance","synthetic-terminal",
        "--output-dir",str(tmp_path)])
    stage.main()
    outcome=json.loads((tmp_path/"stage-outcome.json").read_text())
    assert outcome["outcome"]=="unavailable"
    csvs=list((tmp_path/"forecasts").glob("sv__*.csv"))
    assert len(csvs)==1
    rows=pd.read_csv(csvs[0])
    assert len(rows)==2
    assert rows["global_i"].tolist()==[2,3]
    assert rows["estimation_failed"].all()
    for col in [c for c in rows if c.startswith(("var_","es_"))]:
        assert rows[col].isna().all()
