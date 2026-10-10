"""Summarize Gaussian stochvol ASIS chains and exact-prior reweighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

RHAT_THRESHOLD = 1.01
MIN_STRUCTURAL_ESS = 400
PARAMETERS = ("mu", "phi", "sigma")


def weighted_quantile(values, quantiles, weights):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(quantiles, cw, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commodity", choices=["cocoa", "gold", "oil"], required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    dfs = [pd.read_csv(args.input_dir / f"{args.commodity}_chain_{i}.csv") for i in range(1, 5)]
    posterior = {
        k: np.stack([df[k].to_numpy(float) for df in dfs], axis=0)
        for k in PARAMETERS
    }
    idata = az.from_dict(posterior=posterior)
    rhat = az.rhat(idata, var_names=list(PARAMETERS))
    ess = az.ess(idata, var_names=list(PARAMETERS), method="bulk")
    rhat_by_var = {k: float(np.asarray(rhat[k]).max()) for k in PARAMETERS}
    ess_by_var = {k: float(np.asarray(ess[k]).min()) for k in PARAMETERS}
    max_rhat = max(rhat_by_var.values())
    min_ess = min(ess_by_var.values())
    native_mixing_pass = bool(max_rhat < RHAT_THRESHOLD and min_ess > MIN_STRUCTURAL_ESS)

    all_df = pd.concat(dfs, ignore_index=True)
    logw = all_df["log_prior_ratio_sigma"].to_numpy(float)
    lw_smooth, pareto_k = az.psislw(logw)
    w = np.exp(lw_smooth - np.max(lw_smooth))
    w /= w.sum()
    weighted_ess = float(1.0 / np.sum(w**2))

    native_summary = {}
    reweighted_summary = {}
    for k in PARAMETERS:
        x = all_df[k].to_numpy(float)
        native_summary[k] = {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "q025": float(np.quantile(x, 0.025)),
            "q975": float(np.quantile(x, 0.975)),
        }
        q025, q50, q975 = weighted_quantile(x, [0.025, 0.5, 0.975], w)
        reweighted_summary[k] = {
            "mean": float(np.sum(w * x)),
            "median": float(q50),
            "q025": float(q025),
            "q975": float(q975),
        }

    # Common rule of thumb: k < 0.7 usable, k < 0.5 preferred.
    reweighting_reliable = bool(np.isfinite(pareto_k) and pareto_k < 0.7 and weighted_ess > 400)

    metadata = pd.read_csv(args.input_dir / f"{args.commodity}_metadata.csv").iloc[0].to_dict()
    payload = {
        "kind": "gaussian_sv_asis_reference",
        "commodity": args.commodity,
        "reference_only": True,
        "production_acceptance": False,
        "native_mixing_pass": native_mixing_pass,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "rhat_by_var": rhat_by_var,
        "ess_by_var": ess_by_var,
        "importance_reweighting": {
            "pareto_k": float(pareto_k),
            "weighted_ess": weighted_ess,
            "reliable_under_k_0_7_rule": reweighting_reliable,
            "target_prior": "HalfCauchy(scale=0.5) on sigma_eta",
        },
        "native_posterior_summary": native_summary,
        "reweighted_posterior_summary": reweighted_summary,
        "metadata": metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"asis_gaussian__{args.commodity}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not native_mixing_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
