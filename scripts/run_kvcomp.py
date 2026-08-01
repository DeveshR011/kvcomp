"""Command-line entry point for KV-cache compression sweeps.

Examples:
    Preview what a sweep would run, without loading the model::

        python scripts/run_kvcomp.py --config config/kvcomp/ruler_6gb.json --dry-run

    Run it, then build reports::

        python scripts/run_kvcomp.py --config config/kvcomp/ruler_6gb.json
        python scripts/run_kvcomp.py --report results/kvcomp/ruler_6gb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

# Cap how much of the card PyTorch will hand out. On a 6 GB GPU shared with a
# desktop, letting the allocator grow until the card is full makes Windows page
# GPU memory to host RAM rather than fail -- throughput then collapses by ~50x
# with nothing in the logs. Leaving a slice unclaimed keeps us clear of that
# cliff. Must be set before torch initialises CUDA.
#
# `expandable_segments` is deliberately not used: it makes `memory_reserved`
# report virtual reservations larger than the card, which breaks the
# feasibility arithmetic in engine.available_cache_bytes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from kvcomp.analysis import build_reports  # noqa: E402
from kvcomp.engine import EngineConfig  # noqa: E402
from kvcomp.policies import PolicyConfig  # noqa: E402
from kvcomp.runner import SweepConfig, run_sweep  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_sweep_config(path: Path) -> SweepConfig:
    """Build a :class:`SweepConfig` from JSON.

    Nested ``engine`` and ``policy_config`` objects are converted to their
    dataclasses, and unknown keys are rejected so a typo in a config surfaces
    immediately instead of being silently ignored for a multi-hour run.

    Raises:
        ValueError: If the file contains unknown keys.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    engine_payload = payload.pop("engine", {})
    if "dtype" in engine_payload:
        engine_payload["dtype"] = DTYPES[engine_payload["dtype"]]
    policy_payload = payload.pop("policy_config", {})

    def check(name: str, data: dict, target) -> None:
        allowed = {field.name for field in fields(target)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")

    check("engine", engine_payload, EngineConfig)
    check("policy_config", policy_payload, PolicyConfig)
    check("config", payload, SweepConfig)

    return SweepConfig(
        engine=EngineConfig(**engine_payload),
        policy_config=PolicyConfig(**policy_payload),
        **payload,
    )


# Cost model calibrated on an RTX 4050 (6 GB) with Qwen3-4B at NF4, chunked
# prefill, measured at 8,221 tokens. Chunking caps the cache, so a compressed
# policy's cost is close to linear in context length; the fixed term covers
# tokenisation and decode.
SECONDS_FIXED = 1.0
TOKENS_PER_SECOND = 1100.0

# H2O accumulates attention over every query, so its scoring is quadratic and
# dominates any sweep that includes it at long context.
H2O_SCORING_SECONDS_AT_8K = 15.0

# `full` keeps the whole cache, so decode slows as the context grows.
FULL_DECODE_PENALTY_AT_8K = 1.5


def estimate_seconds(policy: str, context_length: int) -> float:
    """Predict one run's wall time. Rough, but good enough to catch a sweep
    that would run for days before it is launched."""
    base = SECONDS_FIXED + context_length / TOKENS_PER_SECOND
    scale = (context_length / 8192.0) ** 2
    if policy == "h2o":
        return base + H2O_SCORING_SECONDS_AT_8K * scale
    if policy == "full":
        return base + FULL_DECODE_PENALTY_AT_8K * scale
    return base


def estimate_sweep_seconds(config: SweepConfig) -> tuple[float, dict[str, float]]:
    """Total predicted runtime, plus a per-policy breakdown.

    Cells the precheck will refuse are counted as free, since they are skipped
    without running. This mirrors what the runner actually does.
    """
    from kvcomp.runner import _policy_steps

    if config.benchmark == "ruler":
        per_length = len(config.tasks) * config.samples_per_cell
    else:
        per_length = len(config.depths) * config.samples_per_cell

    # Qwen3-4B geometry, matching kvcomp.engine.
    kv_bytes_per_token = 2 * 36 * 8 * 128 * 2
    spare_bytes = 2.2 * 2**30

    total = 0.0
    by_policy: dict[str, float] = {}

    for policy, budget in _policy_steps(config):
        for length in config.context_lengths:
            resident = length if policy == "full" else min(
                length, (budget if budget > 0 else length) + (config.engine.prefill_chunk or length)
            )
            if kv_bytes_per_token * resident > spare_bytes:
                continue  # precheck refuses this cell
            cost = estimate_seconds(policy, length) * per_length
            total += cost
            by_policy[policy] = by_policy.get(policy, 0.0) + cost

    return total, by_policy


def describe(config: SweepConfig) -> None:
    """Print the shape and cost of a sweep without loading anything.

    The VRAM estimate is the number that decides whether a cell is even
    attempted, so it is worth seeing before committing hours to a run.
    """
    from kvcomp.runner import _policy_steps

    steps = list(_policy_steps(config))
    if config.benchmark == "ruler":
        cells = len(config.tasks) * len(config.context_lengths)
    else:
        cells = len(config.context_lengths) * len(config.depths)
    samples = cells * config.samples_per_cell

    print(f"sweep         : {config.name}")
    print(f"benchmark     : {config.benchmark}")
    print(f"model         : {config.engine.model_id} ({config.engine.quantization})")
    print(f"policy steps  : {len(steps)}")
    print(f"samples       : {samples}")
    print(f"total runs    : {len(steps) * samples}")
    print()

    # Qwen3-4B geometry; recomputed exactly at runtime from the loaded config.
    per_token = 2 * 36 * 8 * 128 * 2
    print("estimated full-cache KV size (fp16):")
    for length in config.context_lengths:
        gib = per_token * length / 2**30
        flag = "  <-- exceeds a 6 GB card with weights resident" if gib > 3.0 else ""
        print(f"  {length:>7d} tokens : {gib:6.2f} GiB{flag}")
    print()
    print("compressed cache size by budget:")
    for budget in config.budgets:
        print(f"  budget {budget:>5d}    : {per_token * budget / 2**30:6.3f} GiB")
    print()

    total, by_policy = estimate_sweep_seconds(config)
    spent = f"{total / 60:.0f} min" if total < 3600 else f"{total / 3600:.1f} h"
    print(f"estimated runtime : {spent}  (+ ~2 min model load)")
    for policy, seconds in sorted(by_policy.items(), key=lambda item: -item[1]):
        share = seconds / total * 100 if total else 0
        print(f"  {policy:<14s} {seconds / 3600:5.2f} h  ({share:4.1f}%)")
    print()
    print("Estimates are calibrated at 8k tokens and extrapolated; treat them as")
    print("an order of magnitude, not a promise. Cells the precheck refuses are")
    print("counted as free because they are skipped rather than run.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="sweep config JSON")
    parser.add_argument("--report", type=Path, help="build reports for a results dir")
    parser.add_argument("--dry-run", action="store_true", help="describe without running")
    parser.add_argument("--quiet", action="store_true", help="suppress per-sample output")
    args = parser.parse_args()

    if args.report:
        results = args.report / "results.jsonl"
        if not results.exists():
            print(f"no results at {results}", file=sys.stderr)
            return 1
        for name, path in build_reports(results, args.report).items():
            print(f"{name:20s} {path}")
        return 0

    if not args.config:
        parser.error("one of --config or --report is required")

    config = load_sweep_config(args.config)

    if args.dry_run:
        describe(config)
        return 0

    describe(config)
    print()
    results_path = run_sweep(config, verbose=not args.quiet)
    print(f"\nresults: {results_path}")

    output_dir = results_path.parent
    for name, path in build_reports(results_path, output_dir).items():
        print(f"{name:20s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
