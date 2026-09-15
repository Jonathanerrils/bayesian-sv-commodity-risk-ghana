"""Validate and assemble distributed production shards into canonical outputs.

The fixed evaluation calendar is preserved for every model. A row is valid
only when all primary forecasts are finite and ``estimation_failed`` is false;
a failed row must be explicitly flagged and contain no finite primary risk
forecast. Model availability is reported as an outcome alongside native and
common-date backtests.
"""

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
    common_valid_dates,
    compile_results,
    run_key,
    setup_logging,
)

FORECAST_COLS = ["var_0.01", "es_0.01", "var_0.05", "es_0.05"]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df or df["date"].isna().any():
        raise RuntimeError(f"Invalid date column in shard: {path}")
    return df


def _candidate_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def _bool_series(series: pd.Series, *, na_value: bool, label: str) -> pd.Series:
    """Parse bool/object/string CSV columns without treating 'False' as truthy."""
    def parse(value):
        if pd.isna(value):
            return na_value
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise RuntimeError(f"{label}: cannot parse boolean value {value!r}")
    return series.map(parse).astype(bool)


def _validate_forecast_state(merged: pd.DataFrame, label: str) -> tuple[pd.Series, pd.Series]:
    values = merged[FORECAST_COLS].apply(pd.to_numeric, errors="coerce")
    all_finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    any_finite = np.isfinite(values.to_numpy(dtype=float)).any(axis=1)
    failed = _bool_series(
        merged["estimation_failed"], na_value=True,
        label=f"{label}/estimation_failed",
    ).to_numpy()
    inconsistent_valid = (~failed) & (~all_finite)
    inconsistent_failed = failed & any_finite
    if inconsistent_valid.any():
        bad = merged.loc[inconsistent_valid, "global_i"].astype(int).tolist()[:10]
        raise RuntimeError(f"{label}: unflagged non-finite forecasts at global_i={bad}")
    if inconsistent_failed.any():
        bad = merged.loc[inconsistent_failed, "global_i"].astype(int).tolist()[:10]
        raise RuntimeError(f"{label}: failed rows contain finite forecasts at global_i={bad}")
    return pd.Series(~failed & all_finite, index=merged.index), pd.Series(failed, index=merged.index)


def assemble(input_dir: Path, window: int, refit_every: int, predictive_draws: int, output_dir: Path) -> dict:
    returns_all = load_all_returns(verbose=False)
    csvs = _candidate_csvs(input_dir)
    if not csvs:
        raise RuntimeError(f"No shard CSVs found under {input_dir}")

    forecasts: dict[tuple[str, str], pd.DataFrame] = {}
    availability_rows: list[dict] = []
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
                matches = []
                for p in csvs:
                    if not p.name.startswith(f"benchmark__{commodity}__"):
                        continue
                    candidate = _read_csv(p)
                    if "model" in candidate and candidate["model"].eq(model).all():
                        matches.append((p, candidate))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one benchmark shard for {commodity}/{model}; found {len(matches)}"
                    )
                merged = matches[0][1].copy()
            else:
                pieces = []
                for p in csvs:
                    if not p.name.startswith(f"sv__{commodity}__"):
                        continue
                    df = _read_csv(p)
                    if "model" in df and df["model"].eq(model).all():
                        pieces.append(df)
                if not pieces:
                    raise RuntimeError(f"No SV shards found for {commodity}/{model}")
                merged = pd.concat(pieces, ignore_index=True)

            required = {
                "date", "actual_return", "global_i", "commodity", "model",
                "estimation_failed", *FORECAST_COLS,
            }
            missing_cols = required.difference(merged.columns)
            if missing_cols:
                raise RuntimeError(f"{commodity}/{model}: missing columns {sorted(missing_cols)}")
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
                    f"{commodity}/{model}: incomplete calendar coverage; got {len(merged)}/{n_expected}, "
                    f"missing={missing}, extra={extra}"
                )
            if not pd.DatetimeIndex(merged["date"]).equals(expected_dates):
                raise RuntimeError(f"{commodity}/{model}: forecast dates do not match source data")
            if not np.allclose(
                merged["actual_return"].to_numpy(dtype=float), expected_actual,
                rtol=0.0, atol=1e-12, equal_nan=True,
            ):
                raise RuntimeError(f"{commodity}/{model}: actual returns disagree with source data")

            valid_mask, failed_mask = _validate_forecast_state(merged, f"{commodity}/{model}")

            if model in SV_VARIANTS:
                if "refit" not in merged:
                    raise RuntimeError(f"{commodity}/{model}: missing refit column")
                refit_flags = _bool_series(
                    merged["refit"], na_value=False,
                    label=f"{commodity}/{model}/refit",
                )
                expected_refits = list(range(0, n_expected, refit_every))
                refit_rows = merged[refit_flags].copy()
                refit_i = refit_rows["global_i"].astype(int).tolist()
                if refit_i != expected_refits:
                    raise RuntimeError(
                        f"{commodity}/{model}: refit schedule mismatch; expected {expected_refits}, got {refit_i}"
                    )
                if "mcmc_converged" not in refit_rows:
                    raise RuntimeError(f"{commodity}/{model}: missing mcmc_converged diagnostics")
                converged = _bool_series(
                    refit_rows["mcmc_converged"], na_value=False,
                    label=f"{commodity}/{model}/mcmc_converged",
                )
                good = refit_rows[converged]
                bad = refit_rows[~converged]
                if len(good):
                    if good["mcmc_attempt"].isna().any():
                        raise RuntimeError(f"{commodity}/{model}: converged refit missing accepted attempt")
                    if not (pd.to_numeric(good["mcmc_max_rhat"]) < 1.01).all():
                        raise RuntimeError(f"{commodity}/{model}: accepted refit violates R-hat gate")
                    if not (pd.to_numeric(good["mcmc_min_ess"]) > 400).all():
                        raise RuntimeError(f"{commodity}/{model}: accepted refit violates ESS gate")
                    if not (pd.to_numeric(good["mcmc_divergences"]) == 0).all():
                        raise RuntimeError(f"{commodity}/{model}: accepted refit has divergences")
                failed_flags = _bool_series(
                    merged["estimation_failed"], na_value=True,
                    label=f"{commodity}/{model}/estimation_failed",
                )
                for _, row in bad.iterrows():
                    block_id = int(row["block_id"])
                    block_mask = merged["block_id"] == block_id
                    if not failed_flags[block_mask].all():
                        raise RuntimeError(
                            f"{commodity}/{model}: failed refit block {block_id} contains unflagged rows"
                        )

            fc = merged.drop(columns=["commodity", "model", "shard_kind"], errors="ignore").set_index("date")
            forecasts[(commodity, model)] = fc
            n_valid = int(valid_mask.sum())
            n_failed = int(failed_mask.sum())
            availability_rows.append({
                "commodity": commodity,
                "model": model,
                "n_total": int(n_expected),
                "n_valid": n_valid,
                "n_failed": n_failed,
                "availability_rate": float(n_valid / n_expected),
                "failure_rate": float(n_failed / n_expected),
            })

            key = f"{commodity}/{model}"
            diagnostics["series"][key] = {
                "n_forecasts": int(n_expected),
                "n_valid": n_valid,
                "n_failed": n_failed,
                "availability_rate": float(n_valid / n_expected),
                "start": str(fc.index.min().date()),
                "end": str(fc.index.max().date()),
            }
            if model in SV_VARIANTS:
                refit_flags = _bool_series(merged["refit"], na_value=False, label=f"{key}/refit")
                refits = merged[refit_flags].copy()
                converged = _bool_series(refits["mcmc_converged"], na_value=False, label=f"{key}/mcmc_converged")
                good = refits[converged]
                diagnostics["series"][key].update({
                    "n_refits": int(len(refits)),
                    "converged_refits": int(converged.sum()),
                    "failed_refits": int((~converged).sum()),
                    "fallback_refits": int((pd.to_numeric(good["mcmc_attempt"], errors="coerce") > 1).sum()),
                    "max_rhat_accepted": float(pd.to_numeric(good["mcmc_max_rhat"], errors="coerce").max()) if len(good) else None,
                    "min_ess_accepted": float(pd.to_numeric(good["mcmc_min_ess"], errors="coerce").min()) if len(good) else None,
                    "total_divergences_accepted": int(pd.to_numeric(good["mcmc_divergences"], errors="coerce").sum()) if len(good) else 0,
                    "min_filter_ess_valid": float(pd.to_numeric(merged.loc[valid_mask, "filter_ess"], errors="coerce").min()) if n_valid else None,
                    "median_filter_ess_valid": float(pd.to_numeric(merged.loc[valid_mask, "filter_ess"], errors="coerce").median()) if n_valid else None,
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(availability_rows).sort_values(["commodity", "model"]).to_csv(
        output_dir / "forecast_availability.csv", index=False
    )

    common_rows = []
    for commodity in COMMODITIES:
        dates = common_valid_dates(forecasts, commodity)
        n_total = len(returns_all[commodity]) - window
        common_rows.append({
            "commodity": commodity,
            "n_total_calendar": int(n_total),
            "n_common_valid": int(len(dates)),
            "common_valid_fraction": float(len(dates) / n_total),
        })
    pd.DataFrame(common_rows).to_csv(output_dir / "common_date_coverage.csv", index=False)

    logger = setup_logging(output_dir / "assembly.log")
    primary = compile_results(
        forecasts, window=window, refit_every=refit_every,
        predictive_draws=predictive_draws, logger=logger,
    )

    forecast_dir = output_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    for (commodity, model), fc in forecasts.items():
        fc.to_csv(forecast_dir / f"{commodity}__{model.replace(' ', '_')}.csv")

    diagnostics["primary_rows"] = int(len(primary))
    diagnostics["status"] = "validated_complete_with_availability_reporting"
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
        args.input_dir, args.window, args.refit_every,
        args.predictive_draws, args.output_dir,
    )
    print(json.dumps({"status": diagnostics["status"], "run_key": diagnostics["run_key"]}, indent=2))


if __name__ == "__main__":
    main()
