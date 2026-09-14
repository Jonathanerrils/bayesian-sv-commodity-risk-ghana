"""Validate and assemble distributed production shards into canonical outputs."""

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
from production_runner import (
    ALL_MODELS,
    BENCHMARK_MODELS,
    COMMODITIES,
    SV_VARIANTS,
    compile_results,
    run_key,
    setup_logging,
)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df or df["date"].isna().any():
        raise RuntimeError(f"Invalid date column in shard: {path}")
    return df


def _candidate_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def assemble(
    input_dir: Path,
    window: int,
    refit_every: int,
    predictive_draws: int,
    output_dir: Path,
) -> dict:
    returns_all = load_all_returns(verbose=False)
    csvs = _candidate_csvs(input_dir)
    if not csvs:
        raise RuntimeError(f"No shard CSVs found under {input_dir}")

    forecasts: dict[tuple[str, str], pd.DataFrame] = {}
    diagnostics: dict = {
        "run_key": run_key(window, refit_every, predictive_draws),
        "window": window,
        "refit_every": refit_every,
        "predictive_draws": predictive_draws,
        "models_expected": ALL_MODELS,
        "commodities_expected": COMMODITIES,
        "inputs": [str(p) for p in csvs],
        "series": {},
    }

    for commodity in COMMODITIES:
        n_expected = len(returns_all[commodity]) - window
        if n_expected <= 0:
            raise RuntimeError(f"{commodity}: no forecasts possible at W={window}")
        expected_i = np.arange(n_expected, dtype=int)
        expected_dates = pd.DatetimeIndex(returns_all[commodity].index[window:])
        expected_actual = returns_all[commodity].iloc[window:].to_numpy(dtype=float)

        for model in ALL_MODELS:
            if model in BENCHMARK_MODELS:
                matches = [
                    p for p in csvs
                    if p.name.startswith(f"benchmark__{commodity}__")
                    and _read_csv(p)["model"].eq(model).all()
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one benchmark shard for {commodity}/{model}; "
                        f"found {len(matches)}"
                    )
                merged = _read_csv(matches[0]).copy()
            else:
                pieces = []
                for p in csvs:
                    if not p.name.startswith(f"sv__{commodity}__"):
                        continue
                    df = _read_csv(p)
                    if df["model"].eq(model).all():
                        pieces.append(df)
                if not pieces:
                    raise RuntimeError(f"No SV shards found for {commodity}/{model}")
                merged = pd.concat(pieces, ignore_index=True)

            required = {
                "date",
                "actual_return",
                "global_i",
                "commodity",
                "model",
                "var_0.01",
                "es_0.01",
                "var_0.05",
                "es_0.05",
            }
            missing_cols = required.difference(merged.columns)
            if missing_cols:
                raise RuntimeError(
                    f"{commodity}/{model}: missing columns {sorted(missing_cols)}"
                )
            if not merged["commodity"].eq(commodity).all() or not merged["model"].eq(model).all():
                raise RuntimeError(f"{commodity}/{model}: shard labels are inconsistent")
            if merged["global_i"].duplicated().any():
                dup = merged.loc[merged["global_i"].duplicated(), "global_i"].tolist()[:10]
                raise RuntimeError(f"{commodity}/{model}: duplicate global_i values {dup}")

            merged = merged.sort_values("global_i").reset_index(drop=True)
            got_i = merged["global_i"].to_numpy(dtype=int)
            if len(merged) != n_expected or not np.array_equal(got_i, expected_i):
                missing = sorted(set(expected_i).difference(got_i))[:20]
                extra = sorted(set(got_i).difference(expected_i))[:20]
                raise RuntimeError(
                    f"{commodity}/{model}: incomplete coverage; got {len(merged)}/{n_expected}, "
                    f"missing={missing}, extra={extra}"
                )

            got_dates = pd.DatetimeIndex(merged["date"])
            if not got_dates.equals(expected_dates):
                raise RuntimeError(f"{commodity}/{model}: forecast dates do not match source data")
            actual = merged["actual_return"].to_numpy(dtype=float)
            if not np.allclose(actual, expected_actual, rtol=0.0, atol=1e-12, equal_nan=True):
                raise RuntimeError(f"{commodity}/{model}: actual returns disagree with source data")

            if model in SV_VARIANTS:
                expected_refits = list(range(0, n_expected, refit_every))
                refit_i = merged.loc[merged["refit"].fillna(False).astype(bool), "global_i"].astype(int).tolist()
                if refit_i != expected_refits:
                    raise RuntimeError(
                        f"{commodity}/{model}: refit schedule mismatch; "
                        f"expected {expected_refits}, got {refit_i}"
                    )
                refit_rows = merged[merged["refit"].fillna(False).astype(bool)]
                if refit_rows["mcmc_attempt"].isna().any():
                    raise RuntimeError(f"{commodity}/{model}: refit missing accepted MCMC attempt")
                if not (pd.to_numeric(refit_rows["mcmc_max_rhat"]) < 1.01).all():
                    raise RuntimeError(f"{commodity}/{model}: accepted refit violates R-hat gate")
                if not (pd.to_numeric(refit_rows["mcmc_min_ess"]) > 400).all():
                    raise RuntimeError(f"{commodity}/{model}: accepted refit violates ESS gate")
                if not (pd.to_numeric(refit_rows["mcmc_divergences"]) == 0).all():
                    raise RuntimeError(f"{commodity}/{model}: accepted refit has divergences")

            forecast_cols = ["var_0.01", "es_0.01", "var_0.05", "es_0.05"]
            finite_fraction = float(
                np.isfinite(merged[forecast_cols].to_numpy(dtype=float)).mean()
            )
            if finite_fraction < 1.0:
                raise RuntimeError(
                    f"{commodity}/{model}: non-finite primary forecasts remain "
                    f"(finite fraction={finite_fraction:.6f})"
                )

            fc = merged.drop(columns=["commodity", "model", "shard_kind"], errors="ignore")
            fc = fc.set_index("date")
            forecasts[(commodity, model)] = fc

            key = f"{commodity}/{model}"
            diagnostics["series"][key] = {
                "n_forecasts": int(len(fc)),
                "start": str(fc.index.min().date()),
                "end": str(fc.index.max().date()),
                "finite_fraction": finite_fraction,
            }
            if model in SV_VARIANTS:
                refits = merged[merged["refit"].fillna(False).astype(bool)]
                diagnostics["series"][key].update(
                    {
                        "n_refits": int(len(refits)),
                        "fallback_refits": int((pd.to_numeric(refits["mcmc_attempt"]) > 1).sum()),
                        "max_rhat": float(pd.to_numeric(refits["mcmc_max_rhat"]).max()),
                        "min_ess": float(pd.to_numeric(refits["mcmc_min_ess"]).min()),
                        "total_divergences": int(pd.to_numeric(refits["mcmc_divergences"]).sum()),
                        "min_filter_ess": float(pd.to_numeric(merged["filter_ess"]).min()),
                        "median_filter_ess": float(pd.to_numeric(merged["filter_ess"]).median()),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "assembly.log")
    primary = compile_results(
        forecasts,
        window=window,
        refit_every=refit_every,
        predictive_draws=predictive_draws,
        logger=logger,
    )

    # Copy canonical assembled forecast series into the production artifact.
    forecast_dir = output_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    for (commodity, model), fc in forecasts.items():
        fc.to_csv(forecast_dir / f"{commodity}__{model.replace(' ', '_')}.csv")

    diagnostics["primary_rows"] = int(len(primary))
    diagnostics["status"] = "validated_complete"
    (output_dir / "production_validation_summary.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble validated production shards")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--predictive-draws", type=int, default=20_000)
    args = parser.parse_args()

    diagnostics = assemble(
        args.input_dir,
        args.window,
        args.refit_every,
        args.predictive_draws,
        args.output_dir,
    )
    print(json.dumps({"status": diagnostics["status"], "run_key": diagnostics["run_key"]}, indent=2))


if __name__ == "__main__":
    main()
