import numpy as np
import pandas as pd
import pytest

from checkpoint_safety import (
    SV_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_schema_array,
    parse_bool_series,
    parse_bool_value,
    validate_checkpoint_schema,
)


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
