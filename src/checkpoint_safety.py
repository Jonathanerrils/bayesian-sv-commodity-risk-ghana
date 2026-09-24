"""Safety helpers for versioned SV checkpoint/resume handling.

These helpers deliberately fail closed.  A checkpoint created by an older or
unknown schema must never be interpreted as compatible merely because the file
can be opened, and string values such as ``"False"`` must never be converted
with Python truthiness (where any non-empty string is truthy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SV_CHECKPOINT_SCHEMA_VERSION = 2

_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "f", ""})


def parse_bool_value(value, *, missing: bool = False) -> bool:
    """Parse a scalar Boolean checkpoint value without string truthiness.

    Parameters
    ----------
    value:
        A bool, numeric 0/1 value, common true/false string, or missing value.
    missing:
        Value returned for missing values (NaN/None/pd.NA).

    Raises
    ------
    ValueError
        If the value is ambiguous rather than silently coercible.
    """
    if value is None or value is pd.NA:
        return bool(missing)
    try:
        if pd.isna(value):
            return bool(missing)
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(f"Boolean checkpoint integer must be 0 or 1, got {value!r}")

    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
        raise ValueError(f"Boolean checkpoint number must be 0 or 1, got {value!r}")

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise ValueError(f"Unrecognised Boolean checkpoint value: {value!r}")

    raise ValueError(f"Unsupported Boolean checkpoint value: {value!r}")


def parse_bool_series(series: pd.Series, *, missing: bool = False) -> pd.Series:
    """Return an actual bool Series using :func:`parse_bool_value` per row."""
    return series.map(lambda value: parse_bool_value(value, missing=missing)).astype(bool)


def checkpoint_schema_array() -> np.ndarray:
    """Canonical NumPy representation stored in the state sidecar."""
    return np.array([SV_CHECKPOINT_SCHEMA_VERSION], dtype=np.int64)


def validate_checkpoint_schema(files, schema_values) -> int:
    """Validate and return the sidecar schema version.

    ``files`` is normally ``np.load(...).files`` and ``schema_values`` is the
    loaded ``checkpoint_schema_version`` array.  Missing, malformed, or stale
    schemas are rejected instead of guessed.
    """
    if "checkpoint_schema_version" not in files:
        raise RuntimeError(
            "SV checkpoint sidecar predates schema versioning; refusing an "
            "inexact resume. Restart the checkpoint under the current pipeline."
        )
    values = np.asarray(schema_values).reshape(-1)
    if len(values) != 1:
        raise RuntimeError("Malformed SV checkpoint schema version")
    try:
        version = int(values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Malformed SV checkpoint schema version") from exc
    if version != SV_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Incompatible SV checkpoint schema {version}; expected "
            f"{SV_CHECKPOINT_SCHEMA_VERSION}. Restart the checkpoint."
        )
    return version
