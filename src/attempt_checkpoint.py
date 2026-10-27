"""Fail-closed storage for completed MCMC attempts (not mid-chain resume)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm


def fingerprint(returns, *, variant, model_version, target_accept, seed, config):
    values = np.ascontiguousarray(np.asarray(returns, dtype="<f8"))
    payload = {
        "variant": variant, "model_version": model_version,
        "target_accept": float(target_accept), "seed": int(seed),
        "config": {k: int(config[k]) for k in ("chains", "draws", "tune")},
        "data_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "data_length": int(values.size),
        "pymc_version": pm.__version__,
        "arviz_version": az.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_attempt(directory: Path, key: str, fit: dict):
    """Publish a completed successful sampling attempt atomically."""
    if fit.get("trace") is None or fit.get("error"):
        return False
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (key + ".nc")
    temp = directory / (key + ".partial.nc")
    try:
        trace = fit["trace"].copy()
        trace.attrs["attempt_checkpoint_key"] = key
        az.to_netcdf(trace, str(temp))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return True


def load_attempt(directory: Path, key: str):
    """Only return a fully published inference dataset."""
    path = Path(directory) / (key + ".nc")
    if not path.is_file():
        return None
    try:
        trace = az.from_netcdf(str(path))
        if trace.attrs.get("attempt_checkpoint_key") != key:
            raise ValueError("Checkpoint fingerprint mismatch")
        if "posterior" not in trace.groups() or "sample_stats" not in trace.groups():
            raise ValueError("Checkpoint missing posterior or sampling diagnostics")
        posterior = trace.posterior
        stats = trace.sample_stats
        if not posterior.data_vars or "diverging" not in stats:
            raise ValueError("Checkpoint missing structural samples or divergence diagnostics")
        if posterior.sizes.get("chain", 0) < 1 or posterior.sizes.get("draw", 0) < 1:
            raise ValueError("Checkpoint contains no posterior draws")
        if stats.sizes.get("chain") != posterior.sizes["chain"] or stats.sizes.get("draw") != posterior.sizes["draw"]:
            raise ValueError("Checkpoint diagnostic dimensions disagree with posterior")
        return trace
    except Exception as exc:
        raise RuntimeError(f"Invalid completed-attempt checkpoint: {path}") from exc
