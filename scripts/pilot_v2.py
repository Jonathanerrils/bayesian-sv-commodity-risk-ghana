"""Empirical validity pilot for the repaired v2 forecasting pipeline.

This is deliberately small enough for CI. It is not a substitute for the
production W=1000 experiment. The pilot validates both forecast mechanics and
the candidate adaptive rolling-MCMC policy on committed real data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_all_returns  # noqa: E402
from garch_model import rolling_var_es  # noqa: E402
from sv_model import (  # noqa: E402
    _predictive_returns,
    _transition_filter_state,
    fit_sv,
    initialize_filter_state,
    predictive_var_es,
    update_filter_state,
)

ARTIFACT_DIR = ROOT / "pilot_artifacts"
WINDOW = 180
FORECAST_DAYS = 20
N_PREDICTIVE = 5_000
ALPHAS = (0.01, 0.05)
BASE_SEED = 20_260_913
TARGET_ACCEPT = 0.95

# Empirically selected candidate policy:
# * 1x500 and 2x500 were inadequate.
# * 4x1000 passes SV-t but the leverage variant narrowly misses the strict
#   R-hat/ESS gate because of mu/phi mixing.
# * Rather than weakening diagnostics or forcing every refit to the most
#   expensive setting, retry only failed refits at 4x2000.
MCMC_ATTEMPTS = (
    {"chains": 4, "tune": 1_000, "draws": 1_000},
    {"chains": 4, "tune": 2_000, "draws": 2_000},
)


def _max_equal_run(values, decimals: int = 8) -> int:
    x = np.round(np.asarray(values, dtype=float), decimals)
    if len(x) == 0:
        return 0
    best = current = 1
    for prev, cur in zip(x[:-1], x[1:]):
        if np.isfinite(prev) and np.isfinite(cur) and cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def _legacy_flatness(variant: str, dates: pd.DatetimeIndex) -> dict:
    path = ROOT / "checkpoints" / f"gold_{variant}.csv"
    if not path.exists():
        return {"available": False}
    legacy = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    aligned = legacy.reindex(dates).dropna(subset=["var_0.01"])
    if aligned.empty:
        return {"available": False}
    vals = aligned["var_0.01"].to_numpy(dtype=float)
    return {
        "available": True,
        "n_aligned": int(len(aligned)),
        "unique_var_1pct_rounded_8dp": int(len(np.unique(np.round(vals, 8)))),
        "max_equal_var_1pct_run_rounded_8dp": _max_equal_run(vals),
    }


def _scalar_diagnostics(trace) -> dict:
    scalar_vars = [v for v in trace.posterior.data_vars if v not in ("h", "z")]
    rhat = az.rhat(trace, var_names=scalar_vars)
    ess = az.ess(trace, var_names=scalar_vars)
    diagnostics = {}
    for name in scalar_vars:
        diagnostics[name] = {
            "max_rhat": float(np.nanmax(np.asarray(rhat[name].values, dtype=float))),
            "min_ess": float(np.nanmin(np.asarray(ess[name].values, dtype=float))),
        }
    return diagnostics


def _fit_adaptive(train: np.ndarray, variant: str, seed_offset: int) -> tuple[dict, list]:
    attempts = []
    final_fit = None
    for attempt_no, cfg in enumerate(MCMC_ATTEMPTS, start=1):
        fit = fit_sv(
            train,
            variant=variant,
            chains=cfg["chains"],
            draws=cfg["draws"],
            tune=cfg["tune"],
            fast_mode=False,
            target_accept=TARGET_ACCEPT,
            random_seed=BASE_SEED + seed_offset + attempt_no - 1,
        )
        record = {
            "attempt": attempt_no,
            **cfg,
            "converged": bool(fit.get("converged", False)),
            "max_rhat": float(fit.get("max_rhat", np.nan)),
            "min_ess": float(fit.get("min_ess", np.nan)),
            "n_divergences": int(fit.get("n_divergences", -1))
            if np.isfinite(fit.get("n_divergences", np.nan))
            else None,
            "error": fit.get("error"),
        }
        if fit.get("trace") is not None:
            record["scalar_diagnostics"] = _scalar_diagnostics(fit["trace"])
        attempts.append(record)
        final_fit = fit
        if fit.get("converged", False):
            break
    return final_fit, attempts


def _run_sv_variant(
    sample: pd.Series,
    variant: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, dict]:
    train = sample.iloc[:WINDOW].to_numpy(dtype=float)
    test = sample.iloc[WINDOW:]
    fit, attempts = _fit_adaptive(train, variant, seed_offset)

    fit_summary = {
        "converged": bool(fit.get("converged", False)),
        "max_rhat": float(fit.get("max_rhat", np.nan)),
        "min_ess": float(fit.get("min_ess", np.nan)),
        "n_divergences": int(fit.get("n_divergences", -1))
        if np.isfinite(fit.get("n_divergences", np.nan))
        else None,
        "trace_available": fit.get("trace") is not None,
        "error": fit.get("error"),
        "mcmc_attempts": attempts,
        "accepted_attempt": next(
            (a["attempt"] for a in attempts if a["converged"]), None
        ),
    }
    if fit.get("trace") is None:
        return pd.DataFrame(), fit_summary

    fit_summary["scalar_diagnostics"] = _scalar_diagnostics(fit["trace"])
    state = initialize_filter_state(fit)
    rows = []
    for j, (date, actual) in enumerate(test.items()):
        rng = np.random.default_rng((BASE_SEED + seed_offset) * 1_000_003 + j)
        transition = _transition_filter_state(state, rng)
        predictive_h_mean = float(np.mean(transition["h"]))
        predictive_h_sd = float(np.std(transition["h"]))
        r_pred = _predictive_returns(transition, N_PREDICTIVE, rng)

        row = {
            "date": date,
            "actual_return": float(actual),
            "predictive_h_mean": predictive_h_mean,
            "predictive_h_sd": predictive_h_sd,
        }
        for alpha in ALPHAS:
            var, es = predictive_var_es(r_pred, alpha)
            row[f"var_{alpha}"] = var
            row[f"es_{alpha}"] = es

        state, filter_ess = update_filter_state(transition, float(actual), rng)
        filtered_h_mean = float(np.mean(state["h"]))
        row["filter_ess"] = float(filter_ess)
        row["filtered_h_mean"] = filtered_h_mean
        row["observation_update_delta_h"] = filtered_h_mean - predictive_h_mean
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    finite_cols = [f"var_{a}" for a in ALPHAS] + [f"es_{a}" for a in ALPHAS]
    finite_ok = bool(np.isfinite(df[finite_cols].to_numpy()).all())
    es_order_ok = bool(
        all((df[f"es_{a}"] >= df[f"var_{a}"]).all() for a in ALPHAS)
    )
    update_abs = np.abs(df["observation_update_delta_h"].to_numpy(dtype=float))
    filter_ess = df["filter_ess"].to_numpy(dtype=float)
    var1 = df["var_0.01"].to_numpy(dtype=float)

    summary = {
        **fit_summary,
        "n_forecasts": int(len(df)),
        "finite_forecasts": finite_ok,
        "es_ge_var": es_order_ok,
        "min_filter_ess": float(np.min(filter_ess)),
        "median_filter_ess": float(np.median(filter_ess)),
        "mean_abs_observation_update_delta_h": float(np.mean(update_abs)),
        "max_abs_observation_update_delta_h": float(np.max(update_abs)),
        "filtered_h_std_over_pilot": float(np.std(df["filtered_h_mean"])),
        "unique_var_1pct_rounded_8dp": int(len(np.unique(np.round(var1, 8)))),
        "max_equal_var_1pct_run_rounded_8dp": _max_equal_run(var1),
        "legacy_comparison": _legacy_flatness(variant, df.index),
    }
    summary["pilot_pass"] = bool(
        summary["converged"]
        and summary["n_divergences"] == 0
        and finite_ok
        and es_order_ok
        and summary["min_filter_ess"] > 1.0
        and summary["filtered_h_std_over_pilot"] > 1e-5
        and summary["max_abs_observation_update_delta_h"] > 1e-5
        and summary["max_equal_var_1pct_run_rounded_8dp"] < len(df)
    )
    return df, summary


def _run_benchmarks(sample: pd.Series) -> dict:
    out = {}
    for label, model_type in (("GARCH-t", "GARCH"), ("EGARCH-t", "EGARCH")):
        df = rolling_var_es(
            sample,
            model_type=model_type,
            distribution="t",
            window=WINDOW,
            alphas=list(ALPHAS),
        )
        df.to_csv(ARTIFACT_DIR / f"gold_{label}_pilot.csv")
        valid = df.loc[~df["estimation_failed"]]
        out[label] = {
            "n_forecasts": int(len(df)),
            "n_valid": int(len(valid)),
            "failure_rate": float(df["estimation_failed"].mean()),
            "finite_forecasts": bool(
                len(valid) > 0
                and np.isfinite(
                    valid[["var_0.01", "es_0.01", "var_0.05", "es_0.05"]]
                    .to_numpy()
                ).all()
            ),
            "es_ge_var": bool(
                len(valid) > 0
                and (valid["es_0.01"] >= valid["var_0.01"]).all()
                and (valid["es_0.05"] >= valid["var_0.05"]).all()
            ),
        }
        out[label]["pilot_pass"] = bool(
            out[label]["failure_rate"] <= 0.05
            and out[label]["finite_forecasts"]
            and out[label]["es_ge_var"]
        )
    return out


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_all_returns(verbose=False)["gold"]
    sample = gold.iloc[-(WINDOW + FORECAST_DAYS):].copy()
    if len(sample) != WINDOW + FORECAST_DAYS:
        raise RuntimeError("Not enough gold observations for the pilot")

    summary = {
        "commodity": "gold",
        "window": WINDOW,
        "forecast_days": FORECAST_DAYS,
        "n_predictive": N_PREDICTIVE,
        "mcmc_attempt_policy": list(MCMC_ATTEMPTS),
        "target_accept": TARGET_ACCEPT,
        "sample_start": sample.index.min().isoformat(),
        "sample_end": sample.index.max().isoformat(),
        "models": {},
    }

    for offset, variant in enumerate(("SV-t", "SV-t-Leverage"), start=1):
        df, model_summary = _run_sv_variant(sample, variant, seed_offset=offset * 100)
        summary["models"][variant] = model_summary
        if not df.empty:
            df.to_csv(ARTIFACT_DIR / f"gold_{variant}_pilot.csv")

    summary["models"].update(_run_benchmarks(sample))
    summary["overall_pass"] = bool(
        all(m.get("pilot_pass", False) for m in summary["models"].values())
    )

    with open(ARTIFACT_DIR / "pilot_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["overall_pass"]:
        raise SystemExit("Empirical v2 pilot failed one or more validity gates")


if __name__ == "__main__":
    main()
