"""Single-refit W=1000 smoke test for the repaired production SV pipeline.

This is intentionally not a rolling backtest. It validates that the actual
production window can be fitted under the adaptive MCMC policy before spending
compute on the full three-commodity rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_all_returns  # noqa: E402
from sv_model import (  # noqa: E402
    _predictive_returns,
    _transition_filter_state,
    fit_sv_adaptive,
    initialize_filter_state,
    predictive_var_es,
    update_filter_state,
)

ARTIFACT_DIR = ROOT / "w1000_smoke_artifacts"
WINDOW = 1000
N_PREDICTIVE = 20_000
ALPHAS = (0.01, 0.05)
BASE_SEED = 20_260_913


def _run_variant(train: np.ndarray, actual: float, variant: str, seed: int) -> dict:
    fit = fit_sv_adaptive(
        train,
        variant=variant,
        target_accept=0.95,
        random_seed=seed,
    )

    result = {
        "variant": variant,
        "converged": bool(fit.get("converged", False)),
        "accepted_attempt": fit.get("accepted_attempt"),
        "max_rhat": float(fit.get("max_rhat", np.nan)),
        "min_ess": float(fit.get("min_ess", np.nan)),
        "n_divergences": int(fit.get("n_divergences", -1))
        if np.isfinite(fit.get("n_divergences", np.nan))
        else None,
        "chains": fit.get("chains"),
        "draws": fit.get("draws"),
        "tune": fit.get("tune"),
        "attempts": fit.get("mcmc_attempts", []),
    }

    if fit.get("trace") is None or not fit.get("converged", False):
        result["smoke_pass"] = False
        return result

    state = initialize_filter_state(fit)
    rng = np.random.default_rng(seed * 1_000_003)
    transition = _transition_filter_state(state, rng)
    r_pred = _predictive_returns(transition, N_PREDICTIVE, rng)

    result["predictive_mean"] = float(np.mean(r_pred))
    result["predictive_sd"] = float(np.std(r_pred))
    result["actual_return"] = float(actual)

    for alpha in ALPHAS:
        var, es = predictive_var_es(r_pred, alpha)
        result[f"var_{alpha}"] = float(var)
        result[f"es_{alpha}"] = float(es)

    updated_state, filter_ess = update_filter_state(transition, float(actual), rng)
    before_h = float(np.mean(transition["h"]))
    after_h = float(np.mean(updated_state["h"]))
    result["predictive_h_mean"] = before_h
    result["filtered_h_mean"] = after_h
    result["observation_update_delta_h"] = after_h - before_h
    result["filter_ess"] = float(filter_ess)

    finite_fields = [
        result["predictive_mean"],
        result["predictive_sd"],
        result["filter_ess"],
        result["var_0.01"],
        result["es_0.01"],
        result["var_0.05"],
        result["es_0.05"],
    ]
    result["smoke_pass"] = bool(
        result["converged"]
        and result["n_divergences"] == 0
        and all(np.isfinite(finite_fields))
        and result["predictive_sd"] > 0
        and result["filter_ess"] > 1
        and result["es_0.01"] >= result["var_0.01"]
        and result["es_0.05"] >= result["var_0.05"]
    )
    return result


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_all_returns(verbose=False)["gold"].dropna()
    sample = gold.iloc[-(WINDOW + 1):]
    if len(sample) != WINDOW + 1:
        raise RuntimeError("Not enough gold observations for W=1000 smoke")

    train = sample.iloc[:WINDOW].to_numpy(dtype=float)
    actual = float(sample.iloc[-1])
    summary = {
        "commodity": "gold",
        "window": WINDOW,
        "train_start": sample.index[0].isoformat(),
        "forecast_date": sample.index[-1].isoformat(),
        "n_predictive": N_PREDICTIVE,
        "models": {},
    }

    for i, variant in enumerate(("SV-t", "SV-t-Leverage"), start=1):
        summary["models"][variant] = _run_variant(
            train,
            actual,
            variant,
            BASE_SEED + i * 100,
        )

    summary["overall_pass"] = bool(
        all(v.get("smoke_pass", False) for v in summary["models"].values())
    )

    path = ARTIFACT_DIR / "w1000_smoke_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["overall_pass"]:
        raise SystemExit("W=1000 production-window smoke failed")


if __name__ == "__main__":
    main()
