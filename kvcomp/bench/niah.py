"""Needle-in-a-Haystack depth sweep.

RULER reports accuracy per context length. NIAH adds the second axis that
matters for cache compression: *where* in the context the answer lives.

That axis is diagnostic here. A policy that keeps only sinks and a recent
window has a structural blind spot in the middle of the context, and averaging
over depths would hide it behind a mediocre-looking mean. Sweeping depth turns
that blind spot into something you can see directly in the depth x length grid.
"""

from __future__ import annotations

import random

from .base import Sample, Tokenizer, fit_to_length, insert_at_depth

HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

NEEDLE_TEMPLATE = (
    "The best thing to do in San Francisco is {activity} on a sunny day."
)

ACTIVITIES = [
    "eat a sandwich and sit in Dolores Park",
    "walk across the Golden Gate Bridge",
    "ride a cable car to Fisherman's Wharf",
    "watch the fog roll over Twin Peaks",
    "read a book at Ocean Beach",
]

QUESTION = (
    "What is the best thing to do in San Francisco? "
    "Answer using the exact phrase from the document."
)


def needle_haystack_sweep(
    tokenizer: Tokenizer,
    context_lengths: list[int],
    depths: list[float],
    samples_per_cell: int = 1,
    seed: int = 0,
    haystack_text: str | None = None,
) -> list[Sample]:
    """Build the depth x length NIAH grid.

    Args:
        tokenizer: Used for length control.
        context_lengths: Prompt lengths in tokens.
        depths: Relative needle positions in ``[0, 1]``.
        samples_per_cell: Samples per (length, depth) pair. Different activities
            are used across samples so repeats are not identical.
        seed: Base RNG seed; each cell derives its own stream.
        haystack_text: Optional prose filler. Defaults to repeated neutral text.

    Returns:
        Generated samples, each carrying ``depth`` in :attr:`Sample.metadata`.

    Raises:
        ValueError: If any depth falls outside ``[0, 1]``.
    """
    if any(not 0.0 <= depth <= 1.0 for depth in depths):
        raise ValueError("depths must lie in [0, 1]")

    if haystack_text:
        units = [
            sentence.strip() + ("" if sentence.strip().endswith(".") else ".")
            for sentence in haystack_text.replace("\n", " ").split(". ")
            if sentence.strip()
        ]
    else:
        units = [HAYSTACK_SENTENCE] * 512

    samples: list[Sample] = []
    for context_length in context_lengths:
        for depth in depths:
            for index in range(samples_per_cell):
                rng = random.Random(f"{seed}:niah:{context_length}:{depth}:{index}")
                activity = ACTIVITIES[index % len(ACTIVITIES)]
                needle = NEEDLE_TEMPLATE.format(activity=activity)

                haystack = fit_to_length(
                    tokenizer, units, max(128, context_length - 96)
                )
                haystack = insert_at_depth(haystack, needle, depth)

                prompt = (
                    "Read the following document carefully.\n\n"
                    f"{haystack}\n\n"
                    f"{QUESTION}"
                )
                samples.append(
                    Sample(
                        task="niah",
                        context_length=context_length,
                        index=index,
                        prompt=prompt,
                        answers=[activity],
                        metadata={"depth": depth, "seed": rng.random()},
                    )
                )

    return samples
