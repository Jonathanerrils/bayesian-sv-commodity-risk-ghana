import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.assemble_production_run import assemble
from scripts.build_production_matrix import build
from production_runner import BENCHMARK_MODELS, COMMODITIES, SV_VARIANTS


def test_w1000_matrix_covers_every_refit_block_once_per_variant():
    matrix = build(window=1000, refit_every=42, blocks_per_shard=4)

    assert len(matrix["benchmark"]) == len(COMMODITIES) * len(BENCHMARK_MODELS)
    assert {
        (row["commodity"], row["model"]) for row in matrix["benchmark"]
    } == {
        (commodity, model)
        for commodity in COMMODITIES
        for model in BENCHMARK_MODELS
    }

    for commodity in COMMODITIES:
        rows = matrix["sv"][commodity]
        assert 0 < len(rows) <= 256
        n_refit_blocks = matrix["metadata"][commodity]["n_refit_blocks"]
        for variant in SV_VARIANTS:
            variant_rows = [row for row in rows if row["model"] == variant]
            covered = []
            for row in variant_rows:
                assert 1 <= row["n_blocks"] <= 4
                covered.extend(
                    range(row["start_block"], row["start_block"] + row["n_blocks"])
                )
            assert covered == list(range(n_refit_blocks))


def test_assembler_refuses_empty_input(tmp_path):
    with pytest.raises(RuntimeError, match="No shard CSVs"):
        assemble(
            input_dir=tmp_path,
            window=1000,
            refit_every=42,
            predictive_draws=20_000,
            output_dir=tmp_path / "out",
        )


def test_assembler_rejects_mismatched_sv_target_accept(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import scripts.assemble_production_run as assembly

    dates = pd.date_range("2026-01-01", periods=6, freq="B")
    returns = pd.Series(np.linspace(-0.02, 0.02, len(dates)), index=dates)
    monkeypatch.setattr(
        assembly,
        "load_all_returns",
        lambda verbose=False: {commodity: returns for commodity in COMMODITIES},
    )
    monkeypatch.setattr(assembly, "ALL_MODELS", ["SV-Gaussian"])
    monkeypatch.setattr(assembly, "BENCHMARK_MODELS", [])
    monkeypatch.setattr(assembly, "SV_VARIANTS", ["SV-Gaussian"])

    shard = pd.DataFrame({
        "date": dates[4:],
        "actual_return": returns.iloc[4:].to_numpy(),
        "global_i": [0, 1],
        "commodity": ["cocoa", "cocoa"],
        "model": ["SV-Gaussian", "SV-Gaussian"],
        "estimation_failed": [False, False],
        "refit": [True, False],
        "block_id": [0, 0],
        "target_accept": [0.95, 0.95],
        "mcmc_converged": [True, np.nan],
        "mcmc_attempt": [1, np.nan],
        "mcmc_max_rhat": [1.005, np.nan],
        "mcmc_min_ess": [600.0, np.nan],
        "mcmc_divergences": [0, np.nan],
        "filter_ess": [500.0, 450.0],
        "var_0.01": [0.03, 0.03],
        "es_0.01": [0.04, 0.04],
        "var_0.05": [0.02, 0.02],
        "es_0.05": [0.03, 0.03],
    })
    path = tmp_path / "sv__cocoa__sv-gaussian__blocks-0000-0000.csv"
    shard.to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="target_accept provenance mismatch"):
        assemble(
            input_dir=tmp_path,
            window=4,
            refit_every=42,
            predictive_draws=20_000,
            output_dir=tmp_path / "out",
            target_accept=0.99,
        )
