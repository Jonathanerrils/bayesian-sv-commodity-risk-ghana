"""Durable converged SV posterior transfer between isolated jobs.

A successful fit needs its full posterior, not just a JSON convergence record.
The manifest binds the posterior to a validated training-data and run identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import arviz as az

from attempt_checkpoint import fingerprint
from sv_model import MODEL_VERSION


def save_posterior_artifact(directory, fit, returns, *, variant, target_accept,
                            random_seed, attempts, provenance):
    if not fit.get("converged") or fit.get("trace") is None:
        raise ValueError("only converged fits with a posterior can be published")
    if fit.get("variant") != variant or fit.get("model_version") != MODEL_VERSION:
        raise ValueError("posterior fit identity mismatch")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(returns, variant, target_accept, random_seed, attempts,
                           model_version=MODEL_VERSION, provenance=provenance)
    trace_path = directory / "posterior.nc"
    manifest_path = directory / "manifest.json"
    # A completed publication is immutable. A retry with the same identity
    # is idempotent; an incomplete or mismatched publication must be rejected.
    if trace_path.exists() or manifest_path.exists():
        if not (trace_path.exists() and manifest_path.exists()):
            raise ValueError("incomplete posterior publication; use a clean directory")
        load_posterior_artifact(
            directory, returns, variant=variant, target_accept=target_accept,
            random_seed=random_seed, attempts=attempts, provenance=provenance)
        return directory
    temporary_trace = directory / "posterior.nc.tmp"
    az.to_netcdf(fit["trace"], temporary_trace)
    os.replace(temporary_trace, trace_path)
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    metadata = {
        "identity": identity,
        "posterior_sha256": trace_sha256,
        "accepted_attempt": fit.get("accepted_attempt"),
        "T": fit.get("T"),
        "mean_return": fit.get("mean_return"),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "mcmc_attempts": fit.get("mcmc_attempts", []),
    }
    from sv_model import _checkpoint_number
    for key in ("mean_return", "max_rhat", "min_ess", "n_divergences"):
        metadata[key] = _checkpoint_number(metadata[key])
    if metadata["accepted_attempt"] not in range(1, len(attempts) + 1):
        raise ValueError("invalid accepted attempt")
    temporary_manifest = directory / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(metadata, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temporary_manifest, directory / "manifest.json")
    return directory


def load_posterior_artifact(directory, returns, *, variant, target_accept,
                            random_seed, attempts, provenance):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = fingerprint(returns, variant, target_accept, random_seed, attempts,
                           model_version=MODEL_VERSION, provenance=provenance)
    if metadata.get("identity") != expected:
        raise ValueError("posterior artifact identity mismatch")
    accepted = metadata.get("accepted_attempt")
    if type(accepted) is not int or not 1 <= accepted <= len(attempts):
        raise ValueError("invalid accepted attempt")
    if metadata.get("T") != len(returns):
        raise ValueError("posterior training length mismatch")
    for key in ("max_rhat", "min_ess", "n_divergences"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("posterior convergence diagnostic missing or invalid: " + key)
    trace_path = directory / "posterior.nc"
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != metadata.get("posterior_sha256"):
        raise ValueError("posterior artifact checksum mismatch")
    trace = az.from_netcdf(trace_path)
    return {
        "trace": trace, "converged": True, "variant": variant,
        "model_version": MODEL_VERSION, "T": len(returns),
        "mean_return": metadata["mean_return"],
        "max_rhat": metadata["max_rhat"], "min_ess": metadata["min_ess"],
        "n_divergences": metadata["n_divergences"],
        "mcmc_attempts": metadata["mcmc_attempts"],
        "accepted_attempt": accepted,
    }
