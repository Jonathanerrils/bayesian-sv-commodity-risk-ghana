import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attempt_checkpoint import fingerprint, load_attempt, save_attempt


def test_fingerprint_changes_with_data_and_configuration():
    x = np.array([0.1, 0.2, 0.3])
    kwargs = dict(variant="SV-t", model_version="v1", target_accept=0.99,
                  seed=42, config=dict(chains=4, draws=1000, tune=1000))
    key = fingerprint(x, **kwargs)
    assert key == fingerprint(x.copy(), **kwargs)
    assert key != fingerprint(x + 0.01, **kwargs)
    assert key != fingerprint(x, **{**kwargs, "seed": 43})
    assert key != fingerprint(x, **{**kwargs, "model_version": "v2"})


def test_complete_attempt_roundtrip(tmp_path):
    trace = az.from_dict(posterior={"mu": np.ones((2, 4))}, sample_stats={"diverging": np.zeros((2, 4), dtype=bool)})
    key = "abc"
    assert save_attempt(tmp_path, key, {"trace": trace})
    restored = load_attempt(tmp_path, key)
    assert restored is not None
    np.testing.assert_array_equal(restored.posterior["mu"], trace.posterior["mu"])
    assert not (tmp_path / "abc.partial.nc").exists()


def test_failed_attempt_is_not_published(tmp_path):
    assert not save_attempt(tmp_path, "bad", {"trace": None, "error": "failure"})
    assert load_attempt(tmp_path, "bad") is None


def test_corrupt_attempt_fails_closed(tmp_path):
    (tmp_path / "broken.nc").write_text("invalid")
    with pytest.raises(RuntimeError, match="Invalid completed-attempt checkpoint"):
        load_attempt(tmp_path, "broken")
