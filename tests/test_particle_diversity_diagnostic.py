import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_particle_diversity import trace_filter_block


def _state(n=100):
    return {
        "variant": "SV-Gaussian",
        "mean_return": 0.0,
        "mu": np.full(n, -9.0),
        "phi": np.full(n, 0.97),
        "sigma_eta": np.full(n, 0.2),
        "h": np.full(n, -9.0),
        "particle_id": np.arange(n, dtype=np.int64),
    }


def test_filter_block_records_initial_and_every_update():
    actual = np.array([-0.01, 0.02, -0.03])
    out = trace_filter_block(_state(), actual, start_i=5, random_seed=42)

    assert len(out) == 4
    assert out.iloc[0]["step"] == 0
    assert np.isnan(out.iloc[0]["filter_ess"])
    assert out.iloc[0]["particle_unique_fraction"] == 1.0
    assert out.iloc[-1]["step"] == 3


def test_particle_diversity_is_nonincreasing_across_block():
    actual = np.array([-0.08, -0.05, 0.01, -0.10, 0.02])
    out = trace_filter_block(_state(), actual, start_i=0, random_seed=42)

    fractions = out["particle_unique_fraction"].to_numpy()
    assert np.all(np.diff(fractions) <= 1e-12)
    assert np.all((fractions > 0.0) & (fractions <= 1.0))
    assert np.isfinite(out.iloc[1:]["filter_ess"]).all()
