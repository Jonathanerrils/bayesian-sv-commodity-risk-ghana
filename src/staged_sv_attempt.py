"""Execute exactly one configured MCMC attempt, without implicit escalation.

This is the testable core for a future separate-job recovery runner. It does
not serialize a converged posterior or publish forecast-block outputs.
"""
from __future__ import annotations

from pathlib import Path

from attempt_checkpoint import fingerprint, read_failed_attempts, write_failed_attempts
from sv_model import MODEL_VERSION, fit_sv


def run_attempt(returns, *, variant, target_accept, random_seed, attempts,
                attempt_no, checkpoint_path=None, provenance=None):
    """Return (fit, outcome); publish metadata only for rejected earlier attempts."""
    attempts = tuple(attempts)
    if not attempts or not 1 <= attempt_no <= len(attempts):
        raise ValueError("attempt_no outside configured schedule")
    if attempt_no > 1 and checkpoint_path is None:
        raise ValueError("later attempts require a prior checkpoint")
    identity = fingerprint(returns, variant, target_accept, random_seed, attempts,
                           model_version=MODEL_VERSION, provenance=provenance)
    records = read_failed_attempts(checkpoint_path, identity) if checkpoint_path else []
    if len(records) != attempt_no - 1:
        raise ValueError("checkpoint does not contain exactly the preceding attempts")
    cfg = attempts[attempt_no - 1]
    fit = fit_sv(returns, variant=variant, chains=int(cfg["chains"]),
                 tune=int(cfg["tune"]), draws=int(cfg["draws"]),
                 target_accept=target_accept,
                 random_seed=random_seed + attempt_no - 1, fast_mode=False)
    fit = dict(fit)
    fit["accepted_attempt"] = attempt_no if fit.get("converged", False) else None
    normalized_records = [
        {**record.get("configuration", {}),
         **{key: value for key, value in record.items() if key != "configuration"}}
        for record in records
    ]
    fit["mcmc_attempts"] = normalized_records + [{
        "attempt": attempt_no, **identity["attempts"][attempt_no - 1],
        "converged": bool(fit.get("converged", False)),
        "max_rhat": fit.get("max_rhat"),
        "min_ess": fit.get("min_ess"),
        "n_divergences": fit.get("n_divergences"),
        "rhat_by_var": fit.get("rhat_by_var", {}),
        "ess_by_var": fit.get("ess_by_var", {}),
        "error": fit.get("error"),
    }]
    if fit.get("converged", False):
        return fit, "complete"
    if attempt_no == len(attempts):
        return fit, "unavailable"
    from sv_model import _checkpoint_number, _checkpoint_diagnostics
    records.append({
        "attempt": attempt_no, "configuration": identity["attempts"][attempt_no - 1],
        "converged": False,
        "max_rhat": _checkpoint_number(fit.get("max_rhat")),
        "min_ess": _checkpoint_number(fit.get("min_ess")),
        "n_divergences": _checkpoint_number(fit.get("n_divergences")),
        "rhat_by_var": _checkpoint_diagnostics(fit.get("rhat_by_var", {})),
        "ess_by_var": _checkpoint_diagnostics(fit.get("ess_by_var", {})),
        "error": str(fit.get("error")) if fit.get("error") is not None else None,
    })
    if checkpoint_path is None:
        raise ValueError("checkpoint_path required before escalation")
    write_failed_attempts(Path(checkpoint_path), identity, records)
    return fit, "needs_escalation"
