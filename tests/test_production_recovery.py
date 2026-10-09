import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_production_recovery import audit, planned_blocks


VALIDATED_SHA = "f121fc79a203a8712ac0efb1310dccd438e0b614"


def _plan():
    rows = {}
    metadata = {}
    for commodity in ("cocoa", "gold", "oil"):
        metadata[commodity] = {"n_forecasts": 84}
        rows[commodity] = []
        for model in ("SV-Gaussian", "SV-t"):
            rows[commodity].extend(
                [
                    {"commodity": commodity, "model": model, "start_block": 0, "n_blocks": 1},
                    {"commodity": commodity, "model": model, "start_block": 1, "n_blocks": 1},
                ]
            )
    return {
        "sv": rows,
        "metadata": metadata,
        "config": {"window": 1000, "refit_every": 42, "blocks_per_shard": 1},
    }


def _write_block(
    root: Path,
    commodity: str,
    model: str,
    block_id: int,
    *,
    status: str = "complete",
    n_rows: int = 42,
    suffix: str = "",
    include_threshold: bool = True,
):
    start = block_id * 42
    rows = []
    for offset in range(n_rows):
        first = offset == 0
        failed = status == "statistical-failure"
        row = {
            "date": f"2026-01-{(offset % 28) + 1:02d}",
            "actual_return": 0.001,
            "estimation_failed": failed,
            "refit": first,
            "filter_ess": np.nan if failed else 1000.0,
            "global_i": start + offset,
            "block_id": block_id,
            "commodity": commodity,
            "model": model,
            "shard_kind": "sv",
            "target_accept": 0.99,
            "mcmc_converged": (not failed) if first else np.nan,
            "mcmc_attempt": 1 if first and not failed else np.nan,
            "mcmc_max_rhat": 1.005 if first else np.nan,
            "mcmc_min_ess": 600.0 if first else np.nan,
            "mcmc_divergences": 0 if first else np.nan,
            "var_0.01": np.nan if failed else 0.03,
            "es_0.01": np.nan if failed else 0.04,
            "var_0.05": np.nan if failed else 0.02,
            "es_0.05": np.nan if failed else 0.025,
        }
        if include_threshold:
            row["filter_resample_threshold"] = 0.5
        rows.append(row)
    slug = model.lower().replace(" ", "-")
    path = root / f"sv__{commodity}__{slug}__blocks-{block_id:04d}-{block_id:04d}{suffix}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_plan_has_exactly_one_assignment_per_block():
    keys = planned_blocks(_plan())
    assert len(keys) == 12
    assert len(set(keys)) == 12
    for commodity in ("cocoa", "gold", "oil"):
        for model in ("SV-Gaussian", "SV-t"):
            assert [k.block_id for k in keys if k.commodity == commodity and k.model == model] == [0, 1]


def test_recovery_reruns_only_infrastructure_missing_and_reuses_statistical_failure(tmp_path):
    plan = _plan()
    source = tmp_path / "source"
    normalized = tmp_path / "normalized"
    source.mkdir()

    # Cocoa/Gaussian block 0 is complete but from the known v7 run that omitted
    # the threshold provenance column. Block 1 is only partial and must recover.
    _write_block(source, "cocoa", "SV-Gaussian", 0, include_threshold=False)
    _write_block(source, "cocoa", "SV-Gaussian", 1, n_rows=20)

    # Cocoa/SV-t block 0 is a legitimate statistical failure and must NOT rerun.
    _write_block(source, "cocoa", "SV-t", 0, status="statistical-failure")
    _write_block(source, "cocoa", "SV-t", 1)

    # Fill all remaining planned blocks completely.
    for commodity in ("gold", "oil"):
        for model in ("SV-Gaussian", "SV-t"):
            for block in (0, 1):
                _write_block(source, commodity, model, block)

    result = audit(
        plan,
        source,
        refit_every=42,
        target_accept=0.99,
        filter_threshold=0.5,
        source_run_id="37871158815",
        source_head_sha=VALIDATED_SHA,
        backfill_filter_threshold=True,
        backfill_validated_source_sha=VALIDATED_SHA,
        normalized_dir=normalized,
    )

    by_key = {
        (r["commodity"], r["model"], r["block_id"]): r["status"]
        for r in result["blocks"]
    }
    assert by_key[("cocoa", "SV-Gaussian", 0)] == "complete"
    assert by_key[("cocoa", "SV-Gaussian", 1)] == "infrastructure-missing"
    assert by_key[("cocoa", "SV-t", 0)] == "statistical-failure"
    assert result["recovery_matrix"]["cocoa"] == [
        {"commodity": "cocoa", "model": "SV-Gaussian", "start_block": 1, "n_blocks": 1}
    ]

    reused = pd.read_csv(
        normalized / "sv__cocoa__sv-gaussian__blocks-0000-0000.csv"
    )
    assert np.allclose(reused["filter_resample_threshold"], 0.5)
    assert reused["filter_resample_threshold_provenance"].str.contains(
        "37871158815"
    ).all()


def test_duplicate_terminal_artifacts_fail_closed(tmp_path):
    plan = _plan()
    source = tmp_path / "source"
    source.mkdir()

    for commodity in ("cocoa", "gold", "oil"):
        for model in ("SV-Gaussian", "SV-t"):
            for block in (0, 1):
                _write_block(source, commodity, model, block)

    _write_block(
        source,
        "gold",
        "SV-t",
        1,
        suffix="__duplicate",
    )

    result = audit(
        plan,
        source,
        refit_every=42,
        target_accept=0.99,
        filter_threshold=0.5,
        source_run_id="test",
        source_head_sha=VALIDATED_SHA,
        backfill_filter_threshold=False,
        backfill_validated_source_sha=None,
    )
    row = next(
        r for r in result["blocks"]
        if r["commodity"] == "gold" and r["model"] == "SV-t" and r["block_id"] == 1
    )
    assert row["status"] == "duplicate/conflicting"
    assert not result["safe_for_automatic_recovery"]


def test_backfill_rejects_unvalidated_source_sha(tmp_path):
    plan = _plan()
    source = tmp_path / "source"
    source.mkdir()
    _write_block(source, "cocoa", "SV-Gaussian", 0, include_threshold=False)

    try:
        audit(
            plan,
            source,
            refit_every=42,
            target_accept=0.99,
            filter_threshold=0.5,
            source_run_id="bad",
            source_head_sha="not-validated",
            backfill_filter_threshold=True,
            backfill_validated_source_sha=VALIDATED_SHA,
        )
    except RuntimeError as exc:
        assert "unvalidated source SHA" in str(exc)
    else:
        raise AssertionError("unvalidated backfill should fail closed")
