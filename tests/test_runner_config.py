import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_runner import run_key


def test_run_key_changes_with_window_and_refit():
    a = run_key(1000, 42, 20000)
    b = run_key(750, 42, 20000)
    c = run_key(1000, 21, 20000)
    assert a != b
    assert a != c
    assert "w1000" in a
    assert "r42" in a
