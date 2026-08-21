"""Aggregation and reporting for sweep results.

Two things the original harness lacked and that any defensible comparison needs:

* **Uncertainty.** With a handful of samples per cell, a 3-point accuracy gap is
  usually noise. Every aggregate here carries a bootstrap confidence interval so
  differences can be read honestly.
* **Explicit OOM accounting.** A method that cannot run at a length has not
  scored zero in the same sense as a method that ran and answered wrongly.
  Both facts are reported separately rather than averaged together.

Depends only on the standard library, so reports can be produced anywhere.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class Aggregate:
    """Summary of one group of samples.

    Attributes:
        count: Samples in the group.
        mean: Mean score.
        ci_low: Lower bound of the 95% bootstrap interval.
        ci_high: Upper bound of the 95% bootstrap interval.
        oom_rate: Fraction that failed with out-of-memory.
        error_rate: Fraction that failed for any reason.
        mean_peak_vram_gib: Mean peak allocated VRAM.
        mean_cache_gib: Mean post-compression cache size.
        mean_compression: Mean fraction of cached positions evicted.
        mean_prefill_s: Mean prefill wall time.
        mean_compress_s: Mean compression wall time.
        mean_decode_tps: Mean decode throughput.
    """

    count: int = 0
    mean: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    oom_rate: float = 0.0
    error_rate: float = 0.0
    mean_peak_vram_gib: float = 0.0
    mean_cache_gib: float = 0.0
    mean_compression: float = 0.0
    mean_prefill_s: float = 0.0
    mean_compress_s: float = 0.0
    mean_decode_tps: float = 0.0


#: Fields that jointly identify one benchmark sample.
#:
#: ``depth`` is required even though only NIAH sets it. Every NIAH sample shares
#: ``task="niah"`` and distinguishes itself by needle depth, so a key without it
#: collapses all depths onto the same index: a 945-run sweep deduplicated to 135
#: rows, silently discarding 86% of the data from every published aggregate.
RESULT_KEY = ("policy", "budget", "task", "context_length", "index", "depth")


def load_results(path: str | Path, deduplicate: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL results file.

    Truncated lines are skipped: a run killed mid-write leaves one partial line,
    and that sample is simply re-run on resume.

    Args:
        path: JSONL file to read.
        deduplicate: Keep only the last row per ``RESULT_KEY``. Append-only logs
            can hold the same key twice -- most easily when two sweep processes
            share an output file -- and a duplicated sample would be
            double-weighted in every aggregate that follows.

    Returns:
        Result rows in file order.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not deduplicate:
        return rows

    latest: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        latest[tuple(row.get(key) for key in RESULT_KEY)] = row
    return list(latest.values())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Scores are bounded in ``[0, 1]`` and often bimodal (a needle is found or it
    is not), so a normal-approximation interval would misstate the uncertainty.
    Resampling makes no distributional assumption.

    Args:
        values: Observed scores.
        confidence: Interval width.
        iterations: Bootstrap resamples.
        seed: RNG seed, so reports are reproducible.

    Returns:
        ``(low, high)``. Degenerate inputs return the point value.
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low = means[max(0, math.floor(tail * iterations))]
    high = means[min(iterations - 1, math.ceil((1.0 - tail) * iterations) - 1)]
    return low, high


def aggregate(rows: Iterable[dict[str, Any]], seed: int = 0) -> Aggregate:
    """Summarise a group of result rows."""
    rows = list(rows)
    if not rows:
        return Aggregate()

    scores = [float(row.get("score", 0.0)) for row in rows]
    low, high = bootstrap_ci(scores, seed=seed)

    def average(key: str, scale: float = 1.0, successful_only: bool = True) -> float:
        # Timing and memory figures from a failed run describe the failure, not
        # the method, so they are excluded from performance averages.
        pool = [r for r in rows if not r.get("error")] if successful_only else rows
        values = [float(r.get(key) or 0.0) for r in pool if r.get(key) is not None]
        return _mean(values) / scale if values else 0.0

    return Aggregate(
        count=len(rows),
        mean=_mean(scores),
        ci_low=low,
        ci_high=high,
        oom_rate=_mean([1.0 if row.get("oom") else 0.0 for row in rows]),
        error_rate=_mean([1.0 if row.get("error") else 0.0 for row in rows]),
        mean_peak_vram_gib=average("peak_vram_bytes", 2**30),
        mean_cache_gib=average("cache_bytes", 2**30),
        mean_compression=average("compression_ratio"),
        mean_prefill_s=average("prefill_seconds"),
        mean_compress_s=average("compress_seconds"),
        mean_decode_tps=average("decode_tokens_per_second"),
    )


def group_by(
    rows: Iterable[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple, list[dict[str, Any]]]:
    """Bucket rows by the values of ``keys``."""
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    return dict(buckets)


def _csv_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write a CSV without external dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(_csv_escape(cell) for cell in row) + "\n")


AGGREGATE_HEADER = (
    "count",
    "score_mean",
    "ci_low",
    "ci_high",
    "oom_rate",
    "error_rate",
    "peak_vram_gib",
    "cache_gib",
    "compression",
    "prefill_s",
    "compress_s",
    "decode_tps",
)


def _aggregate_cells(summary: Aggregate) -> tuple:
    return (
        summary.count,
        f"{summary.mean:.4f}",
        f"{summary.ci_low:.4f}",
        f"{summary.ci_high:.4f}",
        f"{summary.oom_rate:.4f}",
        f"{summary.error_rate:.4f}",
        f"{summary.mean_peak_vram_gib:.3f}",
        f"{summary.mean_cache_gib:.4f}",
        f"{summary.mean_compression:.4f}",
        f"{summary.mean_prefill_s:.3f}",
        f"{summary.mean_compress_s:.3f}",
        f"{summary.mean_decode_tps:.2f}",
    )


def build_reports(results_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Produce aggregate CSVs and a markdown summary.

    Args:
        results_path: JSONL produced by the runner.
        output_dir: Directory for reports.

    Returns:
        Mapping of report name to path.
    """
    rows = load_results(results_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    groupings = {
        "by_method": ["policy", "budget"],
        "by_method_length": ["policy", "budget", "context_length"],
        "by_method_task": ["policy", "budget", "task"],
        "by_depth": ["policy", "budget", "context_length", "depth"],
    }

    for name, keys in groupings.items():
        buckets = group_by(rows, keys)
        table = [
            (*key, *_aggregate_cells(aggregate(bucket)))
            for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0]))
        ]
        path = output / f"aggregate_{name}.csv"
        write_csv(path, [*keys, *AGGREGATE_HEADER], table)
        written[name] = path

    written["summary"] = _write_markdown(rows, output / "summary.md")
    return written


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the headline comparison table."""
    lines: list[str] = [
        "# KV-Cache Compression Results",
        "",
        f"Total runs: {len(rows)}",
        "",
        "Scores are substring recall. Intervals are 95% percentile bootstrap.",
        "`oom` counts runs that could not execute at all, which is a different",
        "failure from answering incorrectly and is reported separately.",
        "",
        "## By method",
        "",
        "| policy | budget | n | score | 95% CI | oom | cache GiB | peak GiB | compress % | decode tok/s |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for key, bucket in sorted(
        group_by(rows, ["policy", "budget"]).items(), key=lambda item: str(item[0])
    ):
        policy, budget = key
        summary = aggregate(bucket)
        lines.append(
            f"| {policy} | {budget} | {summary.count} | {summary.mean:.3f} | "
            f"[{summary.ci_low:.3f}, {summary.ci_high:.3f}] | {summary.oom_rate:.0%} | "
            f"{summary.mean_cache_gib:.3f} | {summary.mean_peak_vram_gib:.2f} | "
            f"{summary.mean_compression:.0%} | {summary.mean_decode_tps:.1f} |"
        )

    lengths = sorted({row.get("context_length") for row in rows if row.get("context_length")})
    if lengths:
        lines += ["", "## Score by context length", "", "| policy | budget | " +
                  " | ".join(str(length) for length in lengths) + " |",
                  "|---|---:|" + "---:|" * len(lengths)]
        by_cell = group_by(rows, ["policy", "budget", "context_length"])
        for policy, budget in sorted(
            {(row["policy"], row["budget"]) for row in rows}, key=str
        ):
            cells = []
            for length in lengths:
                bucket = by_cell.get((policy, budget, length))
                if not bucket:
                    cells.append("-")
                    continue
                summary = aggregate(bucket)
                marker = " (OOM)" if summary.oom_rate > 0.5 else ""
                cells.append(f"{summary.mean:.2f}{marker}")
            lines.append(f"| {policy} | {budget} | " + " | ".join(cells) + " |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
