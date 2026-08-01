"""Sweep runner: evaluate policies across tasks, lengths, and budgets.

Results are appended to JSONL as each sample completes. On a 6 GB card a sweep
can take hours and a single oversized cell can hard-OOM the process, so
incremental writes plus resume-on-restart are correctness features here, not
conveniences.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .bench import Sample, generate_ruler, needle_haystack_sweep, score_sample
from .engine import EngineConfig, KVCompressionEngine
from .policies import PolicyConfig


@dataclass
class SweepConfig:
    """Definition of an experiment sweep.

    Attributes:
        name: Output directory name under ``results_root``.
        engine: Model and generation settings.
        benchmark: ``"ruler"`` or ``"niah"``.
        tasks: RULER task names. Ignored for NIAH.
        context_lengths: Prompt lengths to evaluate.
        depths: Needle depths, NIAH only.
        samples_per_cell: Samples per (task, length) pair.
        policies: Policy names to compare.
        budgets: KV budgets to sweep. ``full`` ignores these and runs once.
        policy_config: Base policy knobs; ``budget`` is overridden per sweep step.
        max_new_tokens: Decode cap.
        seed: Base seed for sample generation.
        results_root: Root output directory.
        haystack_path: Optional prose file used as filler.
        skip_if_oom_at_length: Once a (policy, budget) cell OOMs at some length,
            skip longer lengths for that cell. Failures are monotonic in length,
            so retrying wastes minutes per cell for no information.
        probe_infeasible: Actually run the first sample of a cell the precheck
            says cannot fit, then skip the rest. The measured cost of an
            over-budget configuration is a result worth having -- notably that
            full context at 32k does not fit on a 6 GB card -- but paying it
            once per cell is enough. Set ``False`` to trust the prediction and
            never run those cells.
    """

    name: str = "default"
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: str = "ruler"
    tasks: list[str] = field(
        default_factory=lambda: ["niah_single_1", "niah_multikey_1", "vt", "cwe"]
    )
    context_lengths: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    depths: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    samples_per_cell: int = 5
    policies: list[str] = field(
        default_factory=lambda: ["full", "streaming_llm", "snapkv", "pyramidkv", "h2o"]
    )
    budgets: list[int] = field(default_factory=lambda: [256, 512, 1024])
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    max_new_tokens: int = 48
    seed: int = 0
    results_root: str = "results/kvcomp"
    haystack_path: str | None = None
    skip_if_oom_at_length: bool = True
    probe_infeasible: bool = True


def _environment() -> dict[str, Any]:
    """Capture hardware and library versions for reproducibility."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu_name"] = properties.name
        info["gpu_total_bytes"] = properties.total_memory
        info["cuda"] = torch.version.cuda
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover - transformers is a hard dependency
        pass
    return info


def _serialize_config(config: SweepConfig) -> dict[str, Any]:
    """Convert a sweep config to JSON-safe form."""
    payload = asdict(config)
    payload["engine"]["dtype"] = str(config.engine.dtype)
    return payload


def _load_completed(path: Path) -> set[tuple]:
    """Read result keys already present, so a restart resumes rather than repeats."""
    if not path.exists():
        return set()
    done: set[tuple] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves one truncated line; the sample it
                # represents is simply re-run.
                continue
            done.add(
                (
                    row["policy"],
                    row["budget"],
                    row["task"],
                    row["context_length"],
                    row["index"],
                )
            )
    return done


def build_samples(config: SweepConfig, tokenizer) -> list[Sample]:
    """Generate the sample set described by ``config``.

    Raises:
        ValueError: If ``benchmark`` is not ``ruler`` or ``niah``.
    """
    haystack = None
    if config.haystack_path:
        path = Path(config.haystack_path)
        if path.exists():
            haystack = path.read_text(encoding="utf-8", errors="ignore")

    if config.benchmark == "ruler":
        return generate_ruler(
            tokenizer=tokenizer,
            tasks=config.tasks,
            context_lengths=config.context_lengths,
            samples_per_cell=config.samples_per_cell,
            seed=config.seed,
            essay_text=haystack,
        )
    if config.benchmark == "niah":
        return needle_haystack_sweep(
            tokenizer=tokenizer,
            context_lengths=config.context_lengths,
            depths=config.depths,
            samples_per_cell=config.samples_per_cell,
            seed=config.seed,
            haystack_text=haystack,
        )
    raise ValueError(f"unknown benchmark {config.benchmark!r}; expected ruler or niah")


def _policy_steps(config: SweepConfig) -> Iterator[tuple[str, int]]:
    """Yield ``(policy, budget)`` pairs.

    ``full`` is emitted once with budget ``-1``: it ignores the budget, and
    running it per budget would multiply the most expensive baseline by the
    length of the sweep for identical results.
    """
    for policy in config.policies:
        if policy == "full":
            yield policy, -1
        else:
            for budget in config.budgets:
                yield policy, budget


def run_sweep(config: SweepConfig, verbose: bool = True) -> Path:
    """Execute a sweep, writing results incrementally.

    Args:
        config: Sweep definition.
        verbose: Print per-sample progress.

    Returns:
        Path to the JSONL results file.
    """
    output_dir = Path(config.results_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    engine = KVCompressionEngine(config.engine)
    samples = build_samples(config, engine.tokenizer)

    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": _serialize_config(config),
                "environment": _environment(),
                "kv_bytes_per_token": engine.bytes_per_token(),
                "num_samples": len(samples),
                "started_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = _load_completed(results_path)
    if verbose and completed:
        print(f"resuming: {len(completed)} results already present")

    steps = list(_policy_steps(config))
    total = len(steps) * len(samples)
    done = 0

    with results_path.open("a", encoding="utf-8") as handle:
        for policy_name, budget in steps:
            oom_length: int | None = None
            probed_lengths: set[int] = set()

            for sample in samples:
                done += 1
                key = (
                    policy_name,
                    budget,
                    sample.task,
                    sample.context_length,
                    sample.index,
                )
                if key in completed:
                    continue

                if (
                    config.skip_if_oom_at_length
                    and oom_length is not None
                    and sample.context_length >= oom_length
                ):
                    row = {
                        "policy": policy_name,
                        "budget": budget,
                        "task": sample.task,
                        "context_length": sample.context_length,
                        "index": sample.index,
                        "depth": sample.metadata.get("depth"),
                        "score": 0.0,
                        "skipped_after_oom": True,
                        "oom": True,
                        "prediction": "",
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    continue

                policy_config = PolicyConfig(**{**asdict(config.policy_config)})
                policy_config.budget = None if budget < 0 else budget

                prompt = engine.build_prompt(sample.prompt)

                # Measure an over-budget configuration once per length so its
                # real cost is on record, then fall back to the prediction.
                force = (
                    config.probe_infeasible
                    and sample.context_length not in probed_lengths
                )
                if force:
                    probed_lengths.add(sample.context_length)

                record = engine.generate(
                    prompt=prompt,
                    policy_name=policy_name,
                    policy_config=policy_config,
                    max_new_tokens=config.max_new_tokens,
                    force=force,
                )

                if record.oom and config.skip_if_oom_at_length:
                    oom_length = sample.context_length

                score = 0.0 if record.error else score_sample(sample, record.text)

                row = {
                    "policy": policy_name,
                    "budget": budget,
                    "task": sample.task,
                    "context_length": sample.context_length,
                    "index": sample.index,
                    "depth": sample.metadata.get("depth"),
                    "score": score,
                    "prediction": record.text[:400],
                    "answers": sample.answers,
                    "prompt_tokens": record.prompt_tokens,
                    "generated_tokens": record.generated_tokens,
                    "prefill_seconds": record.prefill_seconds,
                    "compress_seconds": record.compress_seconds,
                    "decode_seconds": record.decode_seconds,
                    "total_seconds": record.total_seconds,
                    "decode_tokens_per_second": record.decode_tokens_per_second,
                    "cache_tokens_before": record.cache_tokens_before,
                    "cache_tokens_after": record.cache_tokens_after,
                    "compression_ratio": record.compression_ratio,
                    "cache_bytes": record.cache_bytes,
                    "peak_vram_bytes": record.peak_vram_bytes,
                    "oom": record.oom,
                    "error": record.error,
                    "forced_probe": force,
                    "precheck_blocked": bool(record.extra.get("precheck")),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()

                if verbose:
                    status = "OOM" if record.oom else f"{score:.2f}"
                    print(
                        f"[{done}/{total}] {policy_name:13s} b={budget:5d} "
                        f"{sample.task:16s} L={sample.context_length:6d} "
                        f"score={status:>4s} "
                        f"vram={record.peak_vram_bytes / 2**30:.2f}GiB "
                        f"t={record.total_seconds:.1f}s",
                        flush=True,
                    )

    return results_path
