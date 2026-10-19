from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_primary_production_is_hard_fixed_to_one_block_per_job():
    text = (ROOT / ".github" / "workflows" / "w1000-production.yml").read_text(
        encoding="utf-8"
    )
    assert 'BLOCKS_PER_SHARD: "1"' in text
    assert "blocks_per_shard:" not in text
    assert '--blocks-per-shard "$BLOCKS_PER_SHARD"' in text


def test_recovery_defaults_to_audit_only_and_reruns_single_blocks():
    text = (ROOT / ".github" / "workflows" / "w1000-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "execute_recovery:" in text
    assert "default: false" in text
    assert text.count("--n-blocks 1") == 3
    assert "refusing recovery while source run" in text
    assert 'FILTER_RESAMPLE_THRESHOLD: "0.5"' in text
    assert 'TARGET_ACCEPT: "0.99"' in text
