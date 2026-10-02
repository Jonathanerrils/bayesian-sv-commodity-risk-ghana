import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_safety import (
    SV_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_schema_array,
    parse_bool_series,
    parse_bool_value,
    validate_checkpoint_schema,
)
from sv_model import MODEL_VERSION, _load_filter_state, _resume_failed_refit_count, _save_filter_state


def test_false_string_is_not_truthy():
    assert parse_bool_value("False") is False
    assert parse_bool_value(" true ") is True
    assert parse_bool_value("0") is False
    assert parse_bool_value("1") is True


def test_bool_series_parses_mixed_checkpoint_values():
    series = pd.Series([True, False, "True", "False", 1, 0, np.nan, None])
    parsed = parse_bool_series(series, missing=False)
    assert parsed.dtype == bool
    assert parsed.tolist() == [True, False, True, False, True, False, False, False]


def test_bool_parser_rejects_ambiguous_values():
    with pytest.raises(ValueError):
        parse_bool_value("maybe")
    with pytest.raises(ValueError):
        parse_bool_value(2)


def test_checkpoint_schema_roundtrip():
    files = ["checkpoint_schema_version", "variant", "has_state"]
    assert validate_checkpoint_schema(files, checkpoint_schema_array()) == SV_CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_schema_missing_is_rejected():
    with pytest.raises(RuntimeError, match="predates schema versioning"):
        validate_checkpoint_schema(["variant", "has_state"], np.array([SV_CHECKPOINT_SCHEMA_VERSION]))


def test_checkpoint_schema_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="Incompatible SV checkpoint schema"):
        validate_checkpoint_schema(
            ["checkpoint_schema_version"],
            np.array([SV_CHECKPOINT_SCHEMA_VERSION - 1]),
        )


def test_core_sidecar_roundtrip_writes_schema_and_model_version(tmp_path):
    path = tmp_path / "state.npz"
    state = {
        "variant": "SV-t",
        "mean_return": 0.001,
        "mu": np.array([-9.0, -8.9]),
        "phi": np.array([0.96, 0.97]),
        "sigma_eta": np.array([0.2, 0.21]),
        "nu": np.array([7.0, 8.0]),
        "h": np.array([-8.7, -8.6]),
    }
    _save_filter_state(path, state, next_i=17, active_block_start=0, variant="SV-t")

    with np.load(path, allow_pickle=False) as data:
        assert int(data["checkpoint_schema_version"][0]) == SV_CHECKPOINT_SCHEMA_VERSION
        assert str(data["model_version"][0]) == MODEL_VERSION

    restored, next_i, block_start, variant = _load_filter_state(path)
    assert next_i == 17
    assert block_start == 0
    assert variant == "SV-t"
    assert restored is not None
    assert np.allclose(restored["h"], state["h"])


def test_core_sidecar_rejects_missing_model_version(tmp_path):
    path = tmp_path / "state.npz"
    np.savez_compressed(
        path,
        checkpoint_schema_version=checkpoint_schema_array(),
        next_i=np.array([1]),
        active_block_start=np.array([0]),
        variant=np.array(["SV-t"]),
        has_state=np.array([False]),
    )
    with pytest.raises(RuntimeError, match="no model version"):
        _load_filter_state(path)


def test_core_sidecar_rejects_model_version_mismatch(tmp_path):
    path = tmp_path / "state.npz"
    np.savez_compressed(
        path,
        checkpoint_schema_version=checkpoint_schema_array(),
        model_version=np.array(["old-sv-version"]),
        next_i=np.array([1]),
        active_block_start=np.array([0]),
        variant=np.array(["SV-t"]),
        has_state=np.array([False]),
    )
    with pytest.raises(RuntimeError, match="belongs to model version"):
        _load_filter_state(path)


def test_resume_failed_refit_count_handles_false_strings_correctly():
    existing = pd.DataFrame(
        {
            "refit": ["True", "False", "True", "False"],
            "mcmc_converged": ["False", "False", "True", "False"],
        }
    )
    assert _resume_failed_refit_count(existing) == 1
