"""Audit distributed SV production artifacts and build an exact recovery matrix.

This tool is infrastructure-only.  It never refits a model and never changes
forecast values.  It classifies each planned SV block as one of:

- complete
- statistical-failure
- infrastructure-missing
- duplicate/conflicting

Complete/statistical-failure blocks can be normalized into one-block shards for
reuse.  Only infrastructure-missing blocks are emitted into the recovery
matrix.  Duplicate/conflicting evidence makes the audit unsafe for automatic
recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FORECAST_COLS = ["var_0.01", "es_0.01", "var_0.05", "es_0.05"]
TERMINAL = {"complete", "statistical-failure"}
CSV_RE = re.compile(r"^sv__(?P<commodity>[^_]+)__(?P<slug>.+)__blocks-\d{4}-\d{4}\.csv$")


def _bool(value, *, default: bool | None = None) -> bool:
    if pd.isna(value):
        if default is None:
            raise ValueError("missing boolean")
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _slug(model: str) -> str:
    return model.lower().replace(" ", "-").replace("_", "-")


@dataclass(frozen=True)
class BlockKey:
    commodity: str
    model: str
    block_id: int


def planned_blocks(plan: dict) -> list[BlockKey]:
    keys: list[BlockKey] = []
    for commodity, rows in plan["sv"].items():
        for row in rows:
            for block_id in range(
                int(row["start_block"]),
                int(row["start_block"]) + int(row["n_blocks"]),
            ):
                keys.append(BlockKey(commodity, str(row["model"]), block_id))
    if len(keys) != len(set(keys)):
        raise RuntimeError("production plan contains duplicate SV block assignments")
    return sorted(keys, key=lambda k: (k.commodity, k.model, k.block_id))


def _expected_range(plan: dict, key: BlockKey, refit_every: int) -> range:
    n_forecasts = int(plan["metadata"][key.commodity]["n_forecasts"])
    start = key.block_id * refit_every
    return range(start, min(start + refit_every, n_forecasts))


def _source_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("sv__*.csv") if p.is_file())


def _candidate_blocks(root: Path) -> dict[BlockKey, list[tuple[Path, pd.DataFrame]]]:
    out: dict[BlockKey, list[tuple[Path, pd.DataFrame]]] = {}
    for path in _source_csvs(root):
        df = pd.read_csv(path)
        required = {"commodity", "model", "block_id", "global_i", "estimation_failed"}
        if not required.issubset(df.columns):
            continue
        for block_id, block in df.groupby("block_id", sort=False):
            if block.empty:
                continue
            key = BlockKey(
                str(block["commodity"].iloc[0]),
                str(block["model"].iloc[0]),
                int(block_id),
            )
            out.setdefault(key, []).append((path, block.copy()))
    return out


def _strict_complete(block: pd.DataFrame, expected_i: list[int]) -> tuple[str | None, str]:
    got = sorted(pd.to_numeric(block["global_i"], errors="raise").astype(int).tolist())
    if got != expected_i or len(got) != len(set(got)):
        return None, f"incomplete row coverage: expected {expected_i[0]}..{expected_i[-1]}, got {got[:3]}..{got[-3:] if got else []}"

    block = block.sort_values("global_i")
    first = block.iloc[0]
    if "refit" not in block or not _bool(first["refit"], default=False):
        return None, "block does not begin with a refit row"

    failed = np.array([_bool(v, default=True) for v in block["estimation_failed"]])
    values = block[FORECAST_COLS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))

    mcmc_converged = _bool(first.get("mcmc_converged", np.nan), default=False)
    if mcmc_converged:
        if failed.any() or not finite.all():
            return None, "converged block contains failed/non-finite forecast rows"
        rhat = float(first.get("mcmc_max_rhat", np.nan))
        ess = float(first.get("mcmc_min_ess", np.nan))
        div = float(first.get("mcmc_divergences", np.nan))
        if not (np.isfinite(rhat) and rhat < 1.01):
            return None, f"converged block violates R-hat gate: {rhat}"
        if not (np.isfinite(ess) and ess > 400):
            return None, f"converged block violates ESS gate: {ess}"
        if not (np.isfinite(div) and div == 0):
            return None, f"converged block violates divergence gate: {div}"
        return "complete", "full block with accepted strict-gate refit"

    if not failed.all() or finite.any():
        return None, "nonconverged block is not fail-closed on every row"
    return "statistical-failure", "full block explicitly unavailable after strict-gate failure"


def audit(
    plan: dict,
    input_dir: Path,
    *,
    refit_every: int,
    target_accept: float,
    filter_threshold: float,
    source_run_id: str,
    source_head_sha: str,
    backfill_filter_threshold: bool,
    backfill_validated_source_sha: str | None,
    normalized_dir: Path | None = None,
) -> dict:
    if backfill_filter_threshold:
        if not backfill_validated_source_sha:
            raise RuntimeError("backfill requires --backfill-validated-source-sha")
        if source_head_sha != backfill_validated_source_sha:
            raise RuntimeError(
                "refusing ESS50 provenance backfill for unvalidated source SHA: "
                f"{source_head_sha} != {backfill_validated_source_sha}"
            )

    candidates = _candidate_blocks(input_dir)
    rows = []
    recovery = {"cocoa": [], "gold": [], "oil": []}
    duplicates = []

    if normalized_dir is not None:
        normalized_dir.mkdir(parents=True, exist_ok=True)

    for key in planned_blocks(plan):
        expected = list(_expected_range(plan, key, refit_every))
        evidences = candidates.get(key, [])
        terminal = []
        partial_reasons = []

        for path, block in evidences:
            status, reason = _strict_complete(block, expected)
            if status is None:
                partial_reasons.append({"path": str(path), "reason": reason})
            else:
                terminal.append((path, block, status, reason))

        if len(terminal) > 1:
            status = "duplicate/conflicting"
            reason = "multiple terminal artifacts cover the same planned block"
            duplicates.append({
                "commodity": key.commodity,
                "model": key.model,
                "block_id": key.block_id,
                "sources": [str(x[0]) for x in terminal],
            })
        elif len(terminal) == 1:
            path, block, status, reason = terminal[0]

            observed_target = pd.to_numeric(block.get("target_accept"), errors="coerce")
            if observed_target.isna().any() or not np.allclose(
                observed_target.to_numpy(dtype=float), target_accept, rtol=0.0, atol=1e-12
            ):
                status = "duplicate/conflicting"
                reason = f"target_accept provenance mismatch; expected {target_accept}"
            else:
                if "filter_resample_threshold" not in block:
                    if not backfill_filter_threshold:
                        status = "duplicate/conflicting"
                        reason = "missing filter_resample_threshold provenance"
                    else:
                        block["filter_resample_threshold"] = float(filter_threshold)
                        block["filter_resample_threshold_provenance"] = (
                            f"backfilled from validated source run {source_run_id} "
                            f"at commit {source_head_sha}"
                        )
                else:
                    observed_filter = pd.to_numeric(
                        block["filter_resample_threshold"], errors="coerce"
                    )
                    if observed_filter.isna().any() or not np.allclose(
                        observed_filter.to_numpy(dtype=float),
                        filter_threshold,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        status = "duplicate/conflicting"
                        reason = (
                            "filter_resample_threshold provenance mismatch; "
                            f"expected {filter_threshold}"
                        )

            if status in TERMINAL and normalized_dir is not None:
                block = block.sort_values("global_i").copy()
                block["source_run_id"] = str(source_run_id)
                block["source_head_sha"] = str(source_head_sha)
                out = normalized_dir / (
                    f"sv__{key.commodity}__{_slug(key.model)}__blocks-"
                    f"{key.block_id:04d}-{key.block_id:04d}.csv"
                )
                block.to_csv(out, index=False)
        else:
            status = "infrastructure-missing"
            reason = (
                "no complete terminal block artifact"
                if not partial_reasons
                else "only partial artifact evidence exists"
            )

        if status == "infrastructure-missing":
            recovery.setdefault(key.commodity, []).append(
                {
                    "commodity": key.commodity,
                    "model": key.model,
                    "start_block": key.block_id,
                    "n_blocks": 1,
                }
            )

        rows.append(
            {
                "commodity": key.commodity,
                "model": key.model,
                "block_id": key.block_id,
                "status": status,
                "reason": reason,
                "expected_start_i": expected[0],
                "expected_end_i": expected[-1],
                "source_count": len(evidences),
                "terminal_source_count": len(terminal),
                "sources": [str(p) for p, _ in evidences],
                "partial_evidence": partial_reasons,
            }
        )

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    return {
        "source_run_id": str(source_run_id),
        "source_head_sha": str(source_head_sha),
        "target_accept": float(target_accept),
        "filter_resample_threshold": float(filter_threshold),
        "counts": counts,
        "safe_for_automatic_recovery": not duplicates,
        "blocks": rows,
        "recovery_matrix": recovery,
        "duplicate_conflicts": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--recovery-json", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-head-sha", required=True)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--filter-resample-threshold", type=float, default=0.5)
    parser.add_argument("--backfill-filter-threshold", action="store_true")
    parser.add_argument("--backfill-validated-source-sha")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = audit(
        plan,
        args.input_dir,
        refit_every=args.refit_every,
        target_accept=args.target_accept,
        filter_threshold=args.filter_resample_threshold,
        source_run_id=args.source_run_id,
        source_head_sha=args.source_head_sha,
        backfill_filter_threshold=args.backfill_filter_threshold,
        backfill_validated_source_sha=args.backfill_validated_source_sha,
        normalized_dir=args.normalized_dir,
    )

    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.recovery_json.write_text(
        json.dumps(result["recovery_matrix"], indent=2), encoding="utf-8"
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as fh:
            for commodity in ("cocoa", "gold", "oil"):
                matrix = result["recovery_matrix"].get(commodity, [])
                fh.write(
                    f"recovery_{commodity}="
                    f"{json.dumps(matrix, separators=(',', ':'))}\n"
                )
                fh.write(f"has_{commodity}={'true' if matrix else 'false'}\n")
        fh = None
    print(json.dumps({"counts": result["counts"], "safe": result["safe_for_automatic_recovery"]}, indent=2))

    if not result["safe_for_automatic_recovery"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
