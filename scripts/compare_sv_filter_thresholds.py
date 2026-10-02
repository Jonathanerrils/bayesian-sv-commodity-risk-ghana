"""Compare daily resampling with multiple ESS-trigger thresholds using one SV fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def clone_state(state):
    return {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}


def always_resample_update(transition, actual_return, rng):
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
    new = {"variant": transition["variant"], "mean_return": transition["mean_return"]}
    for key in ("mu", "phi", "sigma_eta", "nu", "rho", "particle_id"):
        if key in transition:
            new[key] = transition[key][idx]
    new["h"] = transition["h_next"][idx]
    new["weights"] = np.full(len(weights), 1.0 / len(weights))
    new["resampled"] = True
    return new, ess


def unique_fraction(state):
    return float(np.unique(state["particle_id"]).size / len(state["particle_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", choices=COMMODITIES, required=True)
    parser.add_argument("--variant", choices=SV_VARIANTS, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--predictive-draws", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds]
    if not thresholds or any(not (0.0 < x <= 1.0) for x in thresholds):
        raise ValueError("all thresholds must lie in (0, 1]")

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
    always = clone_state(base)
    adaptive = {threshold: clone_state(base) for threshold in thresholds}

    metrics = {
        threshold: {
            "resample_count": 0,
            "var_diff_0.01": [], "es_diff_0.01": [],
            "var_diff_0.05": [], "es_diff_0.05": [],
        }
        for threshold in thresholds
    }

    for offset, actual_return in enumerate(returns[train_end:block_end]):
        global_i = start_i + offset
        trans_seed = 42 * 1_000_003 + global_i

        trans_always = _transition_filter_state(always, np.random.default_rng(trans_seed))
        pred_always = _predictive_returns(
            trans_always,
            args.predictive_draws,
            np.random.default_rng(900_000_000 + global_i),
        )
        baseline = {
            alpha: predictive_var_es(pred_always, alpha) for alpha in (0.01, 0.05)
        }

        always, _ = always_resample_update(
            trans_always,
            actual_return,
            np.random.default_rng(700_000_000 + global_i),
        )

        for threshold in thresholds:
            trans = _transition_filter_state(
                adaptive[threshold], np.random.default_rng(trans_seed)
            )
            pred = _predictive_returns(
                trans,
                args.predictive_draws,
                np.random.default_rng(900_000_000 + global_i),
            )
            for alpha in (0.01, 0.05):
                var, es = predictive_var_es(pred, alpha)
                base_var, base_es = baseline[alpha]
                metrics[threshold][f"var_diff_{alpha}"].append(abs(var - base_var))
                metrics[threshold][f"es_diff_{alpha}"].append(abs(es - base_es))

            adaptive[threshold], _ = update_filter_state(
                trans,
                actual_return,
                np.random.default_rng(700_000_000 + global_i),
                resample_threshold=threshold,
            )
            metrics[threshold]["resample_count"] += int(
                adaptive[threshold].get("resampled", False)
            )

    threshold_results = {}
    for threshold in thresholds:
        m = metrics[threshold]
        row = {
            "resample_count": int(m["resample_count"]),
            "final_unique_fraction": unique_fraction(adaptive[threshold]),
        }
        for alpha in (0.01, 0.05):
            for kind in ("var", "es"):
                values = np.asarray(m[f"{kind}_diff_{alpha}"], dtype=float)
                row[f"mean_abs_{kind}_diff_{alpha}"] = float(values.mean())
                row[f"max_abs_{kind}_diff_{alpha}"] = float(values.max())
        threshold_results[str(threshold)] = row

    payload = {
        "commodity": args.commodity,
        "variant": args.variant,
        "block": args.block,
        "target_accept": args.target_accept,
        "accepted_attempt": fit.get("accepted_attempt"),
        "mcmc_max_rhat": fit.get("max_rhat"),
        "mcmc_min_ess": fit.get("min_ess"),
        "mcmc_divergences": fit.get("n_divergences"),
        "n_steps": args.refit_every,
        "always_final_unique_fraction": unique_fraction(always),
        "threshold_results": threshold_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.variant.lower().replace(" ", "-")
    path = args.output_dir / f"filter_thresholds__{args.commodity}__{slug}__block-{args.block}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
