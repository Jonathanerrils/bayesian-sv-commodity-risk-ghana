"""Compare always-resample and ESS-triggered SV filtering on one fitted block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from production_runner import COMMODITIES, SV_VARIANTS
from sv_model import (
    DEFAULT_ROLLING_MCMC_ATTEMPTS,
    _observation_loglik,
    _predictive_returns,
    _transition_filter_state,
    fit_sv_adaptive,
    initialize_filter_state,
    predictive_var_es,
    update_filter_state,
)


def _always_resample_update(transition, actual_return, rng):
    logw = _observation_loglik(transition, actual_return)
    finite = np.isfinite(logw)
    if not finite.any():
        weights = np.full(len(logw), 1.0 / len(logw))
    else:
        floor = np.nanmax(logw[finite]) - 1_000.0
        logw = np.where(finite, logw, floor)
        logw -= np.max(logw)
        weights = np.exp(logw)
        total = weights.sum()
        weights = (
            weights / total
            if np.isfinite(total) and total > 0.0
            else np.full(len(logw), 1.0 / len(logw))
        )
    ess = float(1.0 / np.sum(weights**2))
    idx = rng.choice(len(weights), size=len(weights), replace=True, p=weights)
    new = {
        "variant": transition["variant"],
        "mean_return": transition["mean_return"],
    }
    for key in ("mu", "phi", "sigma_eta", "nu", "rho", "particle_id"):
        if key in transition:
            new[key] = transition[key][idx]
    new["h"] = transition["h_next"][idx]
    new["weights"] = np.full(len(weights), 1.0 / len(weights))
    new["resampled"] = True
    return new, ess


def _unique_fraction(state):
    return float(np.unique(state["particle_id"]).size / len(state["particle_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--variant", choices=SV_VARIANTS, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--predictive-draws", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    returns = load_all_returns(verbose=False)[args.commodity].to_numpy(dtype=float)
    start_i = args.block * args.refit_every
    train_end = start_i + args.window
    block_end = train_end + args.refit_every
    if block_end > len(returns):
        raise ValueError("requested block is incomplete")

    fit = fit_sv_adaptive(
        returns[start_i:train_end],
        variant=args.variant,
        target_accept=args.target_accept,
        random_seed=42 + start_i,
        attempts=DEFAULT_ROLLING_MCMC_ATTEMPTS,
    )
    if not fit.get("converged", False):
        raise RuntimeError("comparison requires a converged structural fit")

    base = initialize_filter_state(fit)
    always = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in base.items()}
    adaptive = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in base.items()}

    rows = []
    for offset, actual_return in enumerate(returns[train_end:block_end]):
        global_i = start_i + offset

        rng_t_a = np.random.default_rng(42 * 1_000_003 + global_i)
        rng_t_b = np.random.default_rng(42 * 1_000_003 + global_i)
        trans_a = _transition_filter_state(always, rng_t_a)
        trans_b = _transition_filter_state(adaptive, rng_t_b)

        pred_rng_a = np.random.default_rng(900_000_000 + global_i)
        pred_rng_b = np.random.default_rng(900_000_000 + global_i)
        pred_a = _predictive_returns(trans_a, args.predictive_draws, pred_rng_a)
        pred_b = _predictive_returns(trans_b, args.predictive_draws, pred_rng_b)

        row = {
            "step": offset + 1,
            "global_i": global_i,
            "always_unique_before": _unique_fraction(always),
            "adaptive_unique_before": _unique_fraction(adaptive),
        }
        for alpha in (0.01, 0.05):
            va, ea = predictive_var_es(pred_a, alpha)
            vb, eb = predictive_var_es(pred_b, alpha)
            row[f"always_var_{alpha}"] = va
            row[f"adaptive_var_{alpha}"] = vb
            row[f"always_es_{alpha}"] = ea
            row[f"adaptive_es_{alpha}"] = eb
            row[f"abs_var_diff_{alpha}"] = abs(vb - va)
            row[f"abs_es_diff_{alpha}"] = abs(eb - ea)

        update_rng_a = np.random.default_rng(700_000_000 + global_i)
        update_rng_b = np.random.default_rng(700_000_000 + global_i)
        always, ess_a = _always_resample_update(trans_a, actual_return, update_rng_a)
        adaptive, ess_b = update_filter_state(
            trans_b,
            actual_return,
            update_rng_b,
            resample_threshold=args.resample_threshold,
        )
        row["always_filter_ess"] = ess_a
        row["adaptive_filter_ess"] = ess_b
        row["adaptive_resampled"] = bool(adaptive["resampled"])
        row["always_unique_after"] = _unique_fraction(always)
        row["adaptive_unique_after"] = _unique_fraction(adaptive)
        rows.append(row)

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"filter_compare__{args.commodity}__{args.variant.lower().replace(' ', '-')}__block-{args.block}"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "commodity": args.commodity,
        "variant": args.variant,
        "block": args.block,
        "target_accept": args.target_accept,
        "resample_threshold": args.resample_threshold,
        "accepted_attempt": fit.get("accepted_attempt"),
        "mcmc_max_rhat": fit.get("max_rhat"),
        "mcmc_min_ess": fit.get("min_ess"),
        "mcmc_divergences": fit.get("n_divergences"),
        "n_steps": int(len(df)),
        "adaptive_resample_count": int(df["adaptive_resampled"].sum()),
        "always_final_unique_fraction": float(df.iloc[-1]["always_unique_after"]),
        "adaptive_final_unique_fraction": float(df.iloc[-1]["adaptive_unique_after"]),
    }
    for alpha in (0.01, 0.05):
        summary[f"mean_abs_var_diff_{alpha}"] = float(df[f"abs_var_diff_{alpha}"].mean())
        summary[f"max_abs_var_diff_{alpha}"] = float(df[f"abs_var_diff_{alpha}"].max())
        summary[f"mean_abs_es_diff_{alpha}"] = float(df[f"abs_es_diff_{alpha}"].mean())
        summary[f"max_abs_es_diff_{alpha}"] = float(df[f"abs_es_diff_{alpha}"].max())
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
