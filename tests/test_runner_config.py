import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_runner import (
    ALL_MODELS,
    ALPHAS,
    BONFERRONI_FAMILY_SIZE,
    COMMODITIES,
    GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE,
    OU_MODEL_VERSION,
    PER_TEST_BONFERRONI_FAMILY_SIZE,
    PIPELINE_VERSION,
    SV_VARIANTS,
    run_key,
)


def test_run_key_changes_with_window_and_refit_and_versions():
    a = run_key(1000, 42, 20000)
    b = run_key(750, 42, 20000)
    c = run_key(1000, 21, 20000)
    assert a != b
    assert a != c
    assert "w1000" in a
    assert "r42" in a
    assert PIPELINE_VERSION in a
    assert OU_MODEL_VERSION in a
    assert "pipeline-v5" in a


def test_v5_primary_model_family_is_frozen():
    assert SV_VARIANTS == ["SV-Gaussian", "SV-t"]
    assert len(ALL_MODELS) == 8


def test_bonferroni_families_are_fixed_before_results():
    cells = len(ALL_MODELS) * len(COMMODITIES) * len(ALPHAS)
    assert cells == 48
    assert PER_TEST_BONFERRONI_FAMILY_SIZE == cells
    assert BONFERRONI_FAMILY_SIZE == 48
    assert GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE == cells * 3
    assert GLOBAL_PRIMARY_BONFERRONI_FAMILY_SIZE == 144
