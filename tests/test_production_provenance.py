import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_production_run import _validate_sv_provenance


def _frame(target_accept=0.99, threshold=0.5):
    return pd.DataFrame(
        {
            "target_accept": [target_accept, target_accept],
            "filter_resample_threshold": [threshold, threshold],
        }
    )


def test_validated_v7_provenance_passes():
    _validate_sv_provenance(_frame(), target_accept=0.99, label="cocoa/SV-t")


@pytest.mark.parametrize(
    "frame, message",
    [
        (_frame(target_accept=0.95), "target_accept provenance mismatch"),
        (_frame(threshold=1.0), "filter resample threshold provenance mismatch"),
        (
            pd.DataFrame({"filter_resample_threshold": [0.5]}),
            "missing target_accept provenance",
        ),
        (
            pd.DataFrame({"target_accept": [0.99]}),
            "missing filter_resample_threshold provenance",
        ),
    ],
)
def test_mismatched_v7_provenance_fails_closed(frame, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_sv_provenance(frame, target_accept=0.99, label="cocoa/SV-t")
