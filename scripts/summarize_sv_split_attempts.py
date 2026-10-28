"""Combine split SV attempt artifacts without confusing statistics and infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_status(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid job status {item!r}; expected name=status")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _case_key(payload: dict) -> tuple[str, str, int]:
    return (
        str(payload["commodity"]),
        str(payload["variant"]),
        int(payload["block"]),
    )


def summarize_attempts(
    artifact_dir: Path,
    expected_cases: list[tuple[str, str, int]],
    job_status: dict[str, str],
) -> dict:
    records = []
    for path in artifact_dir.rglob("sv_refit__*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        records.append(payload)

    summary = {"cases": [], "all_converged": True, "infrastructure_complete": True}

    for commodity, variant, block in expected_cases:
        case_records = [r for r in records if _case_key(r) == (commodity, variant, block)]
        by_attempt = {
            int(r["requested_attempt"]): r
            for r in case_records
            if r.get("requested_attempt") is not None
        }
        attempts = []
        accepted = None

        for attempt in (1, 2):
            job_name = f"{commodity}_{variant}_attempt{attempt}".replace("-", "_")
            result = job_status.get(job_name, "unknown")
            record = by_attempt.get(attempt)

            if record is not None:
                status = "converged" if bool(record.get("converged")) else "statistical_failure"
                attempt_row = {
                    "attempt": attempt,
                    "execution_status": status,
                    "job_result": result,
                    "converged": bool(record.get("converged")),
                    "fit_seed": record.get("fit_seed"),
                    "max_rhat": record.get("max_rhat"),
                    "min_ess": record.get("min_ess"),
                    "n_divergences": record.get("n_divergences"),
                    "geometry_available": bool(
                        record.get("divergence_geometry", {}).get("available", False)
                    ),
                    "artifact": record.get("_path"),
                    "error": record.get("error"),
                }
                if accepted is None and attempt_row["converged"]:
                    accepted = attempt
            elif result == "skipped":
                attempt_row = {
                    "attempt": attempt,
                    "execution_status": "skipped",
                    "job_result": result,
                    "converged": None,
                }
            else:
                attempt_row = {
                    "attempt": attempt,
                    "execution_status": "infrastructure_incomplete",
                    "job_result": result,
                    "converged": None,
                }
                summary["infrastructure_complete"] = False
            attempts.append(attempt_row)

        case_converged = accepted is not None
        if not case_converged:
            summary["all_converged"] = False

        summary["cases"].append(
            {
                "commodity": commodity,
                "variant": variant,
                "block": block,
                "attempts": attempts,
                "accepted_attempt": accepted,
                "strict_gate_satisfied": case_converged,
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-status", action="append", default=[])
    args = parser.parse_args()

    statuses = _parse_status(args.job_status)
    expected = [("cocoa", "SV-t", 0), ("gold", "SV-t", 0)]
    summary = summarize_attempts(args.artifact_dir, expected, statuses)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not summary["infrastructure_complete"]:
        raise SystemExit(2)
    if not summary["all_converged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
