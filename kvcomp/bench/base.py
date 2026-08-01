"""Shared sample type, length control, and scoring.

Scoring follows RULER: a prediction is credited by how many of the expected
answer strings it contains, case-insensitively. Generative models pad answers
with commentary, so exact match would understate real accuracy; substring recall
is the convention these benchmarks report and keeps our numbers comparable to
published ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer surface needed for length control."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass
class Sample:
    """One benchmark item.

    Attributes:
        task: Task identifier, e.g. ``niah_single_1``.
        context_length: Target context length in tokens this sample was built for.
        index: Index within the task/length cell.
        prompt: User-turn content, before any chat template is applied.
        answers: Strings that a correct response must contain.
        metadata: Task-specific detail, such as needle depth.
    """

    task: str
    context_length: int
    index: int
    prompt: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def score_sample(sample: Sample, prediction: str) -> float:
    """Fraction of expected answers present in ``prediction``.

    Args:
        sample: The item that was asked.
        prediction: Raw model output.

    Returns:
        Recall in ``[0, 1]``. Tasks with a single answer therefore yield 0 or 1.
    """
    if not sample.answers:
        return 0.0
    text = prediction.lower()
    found = sum(1 for answer in sample.answers if answer.lower() in text)
    return found / len(sample.answers)


def token_length(tokenizer: Tokenizer, text: str) -> int:
    """Token count of ``text`` without special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def fit_to_length(
    tokenizer: Tokenizer,
    units: list[str],
    target_tokens: int,
    joiner: str = " ",
) -> str:
    """Concatenate ``units`` until close to ``target_tokens``.

    Growth is measured by re-encoding a doubling prefix rather than per unit,
    keeping the cost logarithmic instead of running the tokenizer thousands of
    times per sample.

    Args:
        tokenizer: Used to measure length.
        units: Text fragments to draw from, cycled if exhausted.
        target_tokens: Desired length.
        joiner: Separator placed between units.

    Returns:
        Text whose token count is at most ``target_tokens``.
    """
    if not units:
        return ""

    chosen: list[str] = []
    count = 0
    step = 64

    while count < target_tokens:
        batch = [units[(len(chosen) + i) % len(units)] for i in range(step)]
        candidate = chosen + batch
        measured = token_length(tokenizer, joiner.join(candidate))

        if measured > target_tokens:
            if step == 1:
                break
            step = max(1, step // 4)
            continue

        chosen = candidate
        count = measured
        step = min(step * 2, 4096)

    return joiner.join(chosen)


def insert_at_depth(haystack: str, needle: str, depth: float) -> str:
    """Insert ``needle`` at a relative ``depth`` in ``[0, 1]``.

    Insertion snaps to the nearest sentence boundary so the needle never lands
    mid-sentence, which would make it artificially easy to spot.

    Args:
        haystack: Filler text.
        needle: Sentence to place.
        depth: ``0.0`` for the start, ``1.0`` for the end.

    Returns:
        Haystack with the needle inserted.
    """
    sentences = re.split(r"(?<=[.!?])\s+", haystack)
    if len(sentences) <= 1:
        return f"{needle} {haystack}"
    position = max(0, min(len(sentences), round(depth * len(sentences))))
    sentences.insert(position, needle.strip())
    return " ".join(sentences)
