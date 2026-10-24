"""Attempt-boundary metadata checkpointing for SV MCMC.

Only completed *nonconverged* attempts can be skipped on restart. This module
never claims to resume an interrupted NUTS chain or reconstruct posterior draws.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

# Production checkpointing must additionally pin model-version and run provenance.
# A JSON record documents a completed rejected attempt; it is NOT a saved
# posterior, and no trace may be reconstructed from this record.

def fingerprint(returns, variant, target_accept, seed, attempts, *, model_version=None, provenance=None):
    a = np.asarray(returns, dtype="<f8")
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("returns must be finite 1D array")
    configs = [{"chains": int(x["chains"]), "tune": int(x["tune"]),
                "draws": int(x["draws"])} for x in attempts]
    if not configs:
        raise ValueError("attempts must be nonempty")
    if provenance is not None and (not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError("provenance must be a nonempty string")
    return {"schema": 2, "model_version": model_version,
            "provenance": provenance, "data_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "n": len(a), "variant": variant, "target_accept": float(target_accept),
            "seed": int(seed), "attempts": configs}

def read_failed_attempts(path, identity):
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("identity") != identity:
        raise ValueError("checkpoint identity mismatch")
    records = doc.get("records")
    if not isinstance(records, list) or len(records) > len(identity["attempts"]):
        raise ValueError("invalid checkpoint records")
    for i, record in enumerate(records):
        if (record.get("attempt") != i + 1 or record.get("converged") is not False
                or record.get("configuration") != identity["attempts"][i]):
            raise ValueError("invalid checkpoint attempt")
    return records

def write_failed_attempts(path, identity, records):
    # Validate before writing. A converged trace cannot be reconstructed from JSON.
    path = Path(path)
    if len(records) > len(identity["attempts"]):
        raise ValueError("too many attempts")
    for i, record in enumerate(records):
        if (record.get("attempt") != i + 1 or record.get("converged") is not False
                or record.get("configuration") != identity["attempts"][i]):
            raise ValueError("cannot checkpoint incomplete or converged attempt")
    # A checkpoint is meaningful only after at least one completed rejection.
    if not records:
        raise ValueError("cannot write an empty attempt checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump({"identity": identity, "records": records}, stream,
                      sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
