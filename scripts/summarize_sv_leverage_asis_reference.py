"""Summarize stochvol ASIS reference chains with the project's diagnostics.

This intentionally does not declare production convergence because the
stochvol sigma^2 prior is only a reference approximation to the project's
Half-Cauchy prior.  It answers whether a dedicated ASIS/AWOL leverage sampler
mixes well on the same data/model structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

RHAT_THRESHOLD = 1.01
MIN_STRUCTURAL_ESS = 400
PARAMETERS = ("mu", "phi", "sigma", "rho")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["oil", "synthetic"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    chain_files = [args.input_dir / f"{args.case}_chain_{i}.csv" for i in range(1, 5)]
    chains = [pd.read_csv(path) for path in chain_files]
    lengths = {len(df) for df in chains}
    if len(lengths) != 1:
        raise RuntimeError(f"ASIS chains have unequal lengths: {sorted(lengths)}")

    posterior = {
        name: np.stack([df[name].to_numpy(dtype=float) for df in chains], axis=0)
        for name in PARAMETERS
    }
    idata = az.from_dict(posterior=posterior)
    rhat = az.rhat(idata, var_names=list(PARAMETERS))
    ess = az.ess(idata, var_names=list(PARAMETERS), method="bulk")

    rhat_by_var = {
        name: float(np.nanmax(np.asarray(rhat[name].values, dtype=float)))
        for name in PARAMETERS
    }
    ess_by_var = {
        name: float(np.nanmin(np.asarray(ess[name].values, dtype=float)))
        for name in PARAMETERS
    }
    max_rhat = max(rhat_by_var.values())
    min_ess = min(ess_by_var.values())
    reference_mixing_pass = bool(
        np.isfinite(max_rhat)
        and max_rhat < RHAT_THRESHOLD
        and np.isfinite(min_ess)
        and min_ess > MIN_STRUCTURAL_ESS
    )

    metadata_path = args.input_dir / f"{args.case}_metadata.csv"
    metadata = pd.read_csv(metadata_path).iloc[0].to_dict()

    posterior_summary = {}
    truth_map = {
        "mu": metadata.get("truth_mu"),
        "phi": metadata.get("truth_phi"),
        "sigma": metadata.get("truth_sigma"),
        "rho": metadata.get("truth_rho"),
    }
    for name in PARAMETERS:
        values = posterior[name].reshape(-1)
        q025, q50, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        truth = truth_map[name]
        truth_finite = pd.notna(truth)
        posterior_summary[name] = {
            "mean": float(values.mean()),
            "median": float(q50),
            "q025": float(q025),
            "q975": float(q975),
            "truth": float(truth) if truth_finite else None,
            "contains_true_95pct": bool(q025 <= float(truth) <= q975) if truth_finite else None,
        }

    payload = {
        "kind": "reference_asis_leverage_diagnostic",
        "case": args.case,
        "reference_only": True,
        "production_acceptance": False,
        "reason_not_production_acceptance": (
            "stochvol uses an inverse-gamma reference prior for sigma^2 rather "
            "than the frozen Half-Cauchy prior on sigma_eta"
        ),
        "reference_mixing_pass": reference_mixing_pass,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "rhat_by_var": rhat_by_var,
        "ess_by_var": ess_by_var,
        "posterior_summary": posterior_summary,
        "metadata": metadata,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"asis_reference__{args.case}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    # This workflow is diagnostic. A mixing failure should surface as a failed
    # job, but a pass remains reference evidence rather than production proof.
    if not reference_mixing_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
