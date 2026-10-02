"""Build GitHub Actions matrices for the distributed production run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_all_returns
from production_runner import BENCHMARK_MODELS, COMMODITIES, SV_VARIANTS


def build(window: int, refit_every: int, blocks_per_shard: int) -> dict:
    if blocks_per_shard < 1:
        raise ValueError("blocks_per_shard must be >= 1")
    returns_all = load_all_returns(verbose=False)

    benchmark = [
        {"commodity": commodity, "model": model}
        for commodity in COMMODITIES
        for model in BENCHMARK_MODELS
    ]
    sv_by_commodity: dict[str, list[dict]] = {}
    metadata = {}

    for commodity in COMMODITIES:
        n_forecasts = len(returns_all[commodity]) - window
        if n_forecasts <= 0:
            raise ValueError(
                f"{commodity}: {len(returns_all[commodity])} observations cannot support W={window}"
            )
        n_blocks = math.ceil(n_forecasts / refit_every)
        metadata[commodity] = {
            "n_observations": int(len(returns_all[commodity])),
            "n_forecasts": int(n_forecasts),
            "n_refit_blocks": int(n_blocks),
            "n_shards_per_variant": int(math.ceil(n_blocks / blocks_per_shard)),
        }
        shards = []
        for model in SV_VARIANTS:
            for start_block in range(0, n_blocks, blocks_per_shard):
                shards.append(
                    {
                        "commodity": commodity,
                        "model": model,
                        "start_block": int(start_block),
                        "n_blocks": int(min(blocks_per_shard, n_blocks - start_block)),
                    }
                )
        if len(shards) > 256:
            raise ValueError(
                f"{commodity}: matrix would contain {len(shards)} jobs, above GitHub's 256-job limit; "
                "increase blocks_per_shard"
            )
        sv_by_commodity[commodity] = shards

    return {
        "benchmark": benchmark,
        "sv": sv_by_commodity,
        "metadata": metadata,
        "config": {
            "window": window,
            "refit_every": refit_every,
            "blocks_per_shard": blocks_per_shard,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--refit-every", type=int, default=42)
    parser.add_argument("--blocks-per-shard", type=int, default=4)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    matrix = build(args.window, args.refit_every, args.blocks_per_shard)
    text = json.dumps(matrix, separators=(",", ":"))
    pretty = json.dumps(matrix, indent=2)
    print(pretty)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(pretty, encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write(f"benchmark={json.dumps(matrix['benchmark'], separators=(',', ':'))}\n")
            for commodity in COMMODITIES:
                fh.write(
                    f"sv_{commodity}="
                    f"{json.dumps(matrix['sv'][commodity], separators=(',', ':'))}\n"
                )
            fh.write(f"metadata={json.dumps(matrix['metadata'], separators=(',', ':'))}\n")


if __name__ == "__main__":
    main()
