import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_runner import (
    ALL_MODELS,
    ALPHAS,
    BONFERRONI_FAMILY_SIZE,
    COMMODITIES,
    OU_MODEL_VERSION,
    PIPELINE_VERSION,
    run_key,
)


def test_run_key_changes_with_window_and_refit():
    a = run_key(1000, 42, 20000)
    b = run_key(750, 42, 20000)
    c = run_key(1000, 21, 20000)
    assert a != b
    assert a != c
    assert "w1000" in a
    assert "r42" in a
    assert PIPELINE_VERSION in a
    assert OU_MODEL_VERSION in a


def test_bonferroni_family_tracks_full_production_comparison_cells():
    assert BONFERRONI_FAMILY_SIZE == len(ALL_MODELS) * len(COMMODITIES) * len(ALPHAS)
    assert BONFERRONI_FAMILY_SIZE == 60
