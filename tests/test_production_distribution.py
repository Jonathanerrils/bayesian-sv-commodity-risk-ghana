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
