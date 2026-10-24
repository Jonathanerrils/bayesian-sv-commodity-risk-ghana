import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from attempt_checkpoint import fingerprint, read_failed_attempts, write_failed_attempts

DATA=np.array([.1,-.2,.3])
ATTEMPTS=({"chains":4,"tune":1000,"draws":1000},{"chains":4,"tune":2000,"draws":2000})
def identity():
    return fingerprint(DATA,"SV-t",.99,42,ATTEMPTS)
def record():
    return {"attempt":1,"configuration":identity()["attempts"][0],
            "converged":False,"max_rhat":1.04,"min_ess":200,"n_divergences":0}
def test_round_trip(tmp_path):
    p=tmp_path/"attempt.json"
    write_failed_attempts(p,identity(),[record()])
    assert read_failed_attempts(p,identity())==[record()]
def test_changed_data_rejected(tmp_path):
    p=tmp_path/"attempt.json"
    write_failed_attempts(p,identity(),[record()])
    other=fingerprint(DATA+.01,"SV-t",.99,42,ATTEMPTS)
    with pytest.raises(ValueError,match="identity"):
        read_failed_attempts(p,other)
def test_changed_seed_rejected(tmp_path):
    p=tmp_path/"attempt.json"
    write_failed_attempts(p,identity(),[record()])
    with pytest.raises(ValueError,match="identity"):
        read_failed_attempts(p,fingerprint(DATA,"SV-t",.99,43,ATTEMPTS))
def test_converged_attempt_cannot_be_saved(tmp_path):
    r=record();r["converged"]=True
    with pytest.raises(ValueError,match="cannot checkpoint"):
        write_failed_attempts(tmp_path/"attempt.json",identity(),[r])
def test_tampered_checkpoint_rejected(tmp_path):
    p=tmp_path/"attempt.json"
    write_failed_attempts(p,identity(),[record()])
    d=json.loads(p.read_text());d["records"][0]["converged"]=True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="invalid"):
        read_failed_attempts(p,identity())
def test_missing_checkpoint_is_empty(tmp_path):
    assert read_failed_attempts(tmp_path/"missing.json",identity())==[]

def test_model_version_mismatch_rejected(tmp_path):
    p=tmp_path/"attempt.json"
    base=fingerprint(DATA,"SV-t",.99,42,ATTEMPTS,model_version="v1")
    changed=fingerprint(DATA,"SV-t",.99,42,ATTEMPTS,model_version="v2")
    write_failed_attempts(p,base,[record()])
    with pytest.raises(ValueError,match="identity"):
        read_failed_attempts(p,changed)

def test_provenance_mismatch_rejected(tmp_path):
    p=tmp_path/"attempt.json"
    base=fingerprint(DATA,"SV-t",.99,42,ATTEMPTS,provenance="run-A")
    changed=fingerprint(DATA,"SV-t",.99,42,ATTEMPTS,provenance="run-B")
    write_failed_attempts(p,base,[record()])
    with pytest.raises(ValueError,match="identity"):
        read_failed_attempts(p,changed)

def test_empty_checkpoint_is_not_published(tmp_path):
    p = tmp_path / "attempt.json"
    with pytest.raises(ValueError, match="empty"):
        write_failed_attempts(p, identity(), [])
    assert not p.exists()
