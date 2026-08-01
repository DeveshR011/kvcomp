"""RULER synthetic task generators.

RULER (Hsieh et al., 2024, arXiv:2404.06654) measures effective context length
with tasks whose difficulty is controllable and whose answers cannot be guessed
from parametric knowledge. Every task here is generated from a seeded RNG, so a
run is reproducible without downloading anything.

The suite deliberately spans three retrieval shapes, because compression methods
fail differently on each:

* **Needle tasks** reward keeping a few specific positions. Score-based policies
  do well as long as the needle falls inside the observation window's attention.
* **Aggregation tasks** (``cwe``, ``fwe``) need evidence spread across the whole
  context, so any policy that keeps a small fraction of positions should degrade
  sharply. These are where sink+recent heuristics are expected to break.
* **Multi-hop tracking** (``vt``) needs a chain of positions, punishing policies
  that keep isolated high-scoring tokens without their context.

A benchmark built only from single-needle tasks would flatter every method; the
aggregation tasks are what make the comparison informative.
"""

from __future__ import annotations

import random
import string
import uuid
from collections import Counter

from .base import Sample, Tokenizer, fit_to_length, insert_at_depth

#: Semantically empty filler. Carries no information that could substitute for
#: the needle, so retrieval cannot be short-circuited by topical guessing.
NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

WORD_POOL = [
    "apple", "silver", "mountain", "candle", "river", "purple", "engine", "garden",
    "window", "planet", "forest", "marble", "copper", "lantern", "harbor", "meadow",
    "cactus", "velvet", "anchor", "pepper", "shadow", "ribbon", "tunnel", "willow",
    "bronze", "cavern", "dolphin", "ember", "falcon", "glacier", "hazel", "ivory",
]

RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
)


def _magic_number(rng: random.Random) -> str:
    return str(rng.randint(1000000, 9999999))


def _magic_word(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(8))


def _needle(key: str, value: str) -> str:
    return f"One of the special magic numbers for {key} is: {value}."


def _noise_units(count: int = 512) -> list[str]:
    return [NOISE_SENTENCE] * count


def _essay_units(essay_text: str | None) -> list[str]:
    """Split real prose into sentences, falling back to noise if unavailable."""
    if not essay_text:
        return _noise_units()
    sentences = [s.strip() for s in essay_text.replace("\n", " ").split(". ") if s.strip()]
    return [s if s.endswith(".") else s + "." for s in sentences] or _noise_units()


def _build_niah(
    tokenizer: Tokenizer,
    rng: random.Random,
    context_length: int,
    units: list[str],
    num_needles: int,
    num_queried: int,
    values_per_key: int,
    value_kind: str,
    depth: float | None,
) -> tuple[str, list[str], dict]:
    """Assemble a needle-in-a-haystack prompt.

    Args:
        num_needles: Total needles inserted, including distractors.
        num_queried: How many keys the question asks about.
        values_per_key: Values attached to each queried key.
        value_kind: ``"number"``, ``"word"``, or ``"uuid"``.
        depth: Fixed relative depth for a single needle, or ``None`` to scatter.

    Returns:
        ``(prompt, answers, metadata)``.
    """
    def make_value() -> str:
        if value_kind == "uuid":
            return str(uuid.UUID(int=rng.getrandbits(128)))
        if value_kind == "word":
            return _magic_word(rng)
        return _magic_number(rng)

    keys = [f"{rng.choice(WORD_POOL)}-{rng.randint(100, 999)}" for _ in range(num_needles)]
    values = {key: [make_value() for _ in range(values_per_key)] for key in keys}

    # Reserve room for needles and the question so the final prompt lands near
    # the requested length rather than overshooting it.
    reserved = 64 + 24 * num_needles * values_per_key
    haystack = fit_to_length(tokenizer, units, max(128, context_length - reserved))

    depths = (
        [depth] * num_needles
        if depth is not None
        else [(i + 1) / (num_needles + 1) for i in range(num_needles)]
    )
    for key, needle_depth in zip(keys, depths):
        for value in values[key]:
            haystack = insert_at_depth(haystack, _needle(key, value), needle_depth)

    queried = keys[:num_queried]
    answers = [value for key in queried for value in values[key]]

    if num_queried == 1 and values_per_key == 1:
        question = f"What is the special magic number for {queried[0]}?"
    elif num_queried == 1:
        question = (
            f"What are all the special magic numbers for {queried[0]}? "
            f"There are {values_per_key} of them."
        )
    else:
        listed = ", ".join(queried)
        question = f"What are the special magic numbers for these: {listed}?"

    prompt = (
        "Some special magic numbers are hidden within the following text. "
        "Make sure to memorize them. I will quiz you about them afterwards.\n\n"
        f"{haystack}\n\n"
        f"{question}\n"
        "Answer with only the numbers, separated by commas."
    )
    return prompt, answers, {"depth": depth, "num_needles": num_needles}


def _build_vt(
    tokenizer: Tokenizer, rng: random.Random, context_length: int, chain_length: int = 4
) -> tuple[str, list[str], dict]:
    """Variable tracking: follow assignment chains to a target value.

    The answer requires every link in the chain, so dropping any one of the
    cached positions makes the item unanswerable. This is the multi-hop probe.
    """
    target = rng.randint(10000, 99999)
    names = rng.sample([w.upper() for w in WORD_POOL], k=min(len(WORD_POOL), 24))

    chain = names[:chain_length]
    statements = [f"VAR {chain[0]} = {target}"]
    statements += [f"VAR {chain[i]} = VAR {chain[i - 1]}" for i in range(1, chain_length)]

    # Distractor chains use different values, so a model cannot answer by
    # pattern-matching the assignment syntax alone.
    for name in names[chain_length:]:
        statements.append(f"VAR {name} = {rng.randint(10000, 99999)}")

    rng.shuffle(statements)
    filler = fit_to_length(tokenizer, _noise_units(), max(128, context_length - 512))
    body = filler + "\n" + "\n".join(statements)

    prompt = (
        "Memorize the variable assignments in the following text.\n\n"
        f"{body}\n\n"
        f"Question: Find all variables that are assigned the value {target}, "
        "directly or through a chain of assignments. "
        "Answer with the variable names only, separated by commas."
    )
    return prompt, chain, {"chain_length": chain_length, "value": target}


def _build_cwe(
    tokenizer: Tokenizer, rng: random.Random, context_length: int, num_common: int = 3
) -> tuple[str, list[str], dict]:
    """Common word extraction: report the words that appear most often.

    Evidence is distributed across the entire context by construction, so no
    small subset of positions is sufficient. This is the aggregation probe.
    """
    approximate_words = max(64, context_length // 2)
    common = [_magic_word(rng) for _ in range(num_common)]
    rare = [_magic_word(rng) for _ in range(max(8, approximate_words // 24))]

    # Common words appear ~6x as often as any rare word, a gap that is obvious
    # with the full context and invisible from a sampled fraction of it.
    words = [word for word in common for _ in range(30)]
    while len(words) < approximate_words:
        words.append(rng.choice(rare))
    rng.shuffle(words)

    text = fit_to_length(tokenizer, words, max(128, context_length - 128), joiner=" ")
    counts = Counter(text.split())
    answers = [word for word, _ in counts.most_common(num_common)]

    prompt = (
        "Below is a list of words. Some words appear much more often than others.\n\n"
        f"{text}\n\n"
        f"Question: What are the {num_common} most frequently appearing words? "
        "Answer with the words only, separated by commas."
    )
    return prompt, answers, {"num_common": num_common}


def _build_fwe(
    tokenizer: Tokenizer, rng: random.Random, context_length: int, alpha: float = 2.0
) -> tuple[str, list[str], dict]:
    """Frequent word extraction over a Zipf-like distribution.

    Softer than ``cwe``: frequencies decay smoothly rather than splitting into
    two groups, so it measures graceful degradation instead of a cliff.
    """
    vocabulary = [_magic_word(rng) for _ in range(64)]
    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(vocabulary))]

    approximate_words = max(64, context_length // 2)
    words = rng.choices(vocabulary, weights=weights, k=approximate_words)

    text = fit_to_length(tokenizer, words, max(128, context_length - 128), joiner=" ")
    counts = Counter(text.split())
    answers = [word for word, _ in counts.most_common(3)]

    prompt = (
        "Below is a list of words.\n\n"
        f"{text}\n\n"
        "Question: What are the 3 most frequently appearing words? "
        "Answer with the words only, separated by commas."
    )
    return prompt, answers, {"alpha": alpha}


def generate_ruler(
    tokenizer: Tokenizer,
    tasks: list[str],
    context_lengths: list[int],
    samples_per_cell: int = 10,
    seed: int = 0,
    essay_text: str | None = None,
) -> list[Sample]:
    """Generate a full RULER sample set.

    Args:
        tokenizer: Used for length control.
        tasks: Task names from :data:`RULER_TASKS`.
        context_lengths: Target prompt lengths in tokens.
        samples_per_cell: Samples per (task, length) pair.
        seed: Base RNG seed. Each cell derives its own stream, so adding a task
            or a length never perturbs the samples of the others.
        essay_text: Prose haystack for the ``_2``/``_3`` needle variants.

    Returns:
        Generated samples.

    Raises:
        ValueError: If a task name is unknown.
    """
    unknown = set(tasks) - set(RULER_TASKS)
    if unknown:
        raise ValueError(f"unknown RULER tasks: {sorted(unknown)}")

    noise = _noise_units()
    essay = _essay_units(essay_text)
    samples: list[Sample] = []

    for task in tasks:
        for context_length in context_lengths:
            for index in range(samples_per_cell):
                rng = random.Random(f"{seed}:{task}:{context_length}:{index}")

                if task == "niah_single_1":
                    built = _build_niah(
                        tokenizer, rng, context_length, noise, 1, 1, 1, "number", 0.5
                    )
                elif task == "niah_single_2":
                    built = _build_niah(
                        tokenizer, rng, context_length, essay, 1, 1, 1, "number", None
                    )
                elif task == "niah_single_3":
                    built = _build_niah(
                        tokenizer, rng, context_length, essay, 1, 1, 1, "uuid", None
                    )
                elif task == "niah_multikey_1":
                    built = _build_niah(
                        tokenizer, rng, context_length, essay, 4, 1, 1, "number", None
                    )
                elif task == "niah_multivalue":
                    built = _build_niah(
                        tokenizer, rng, context_length, essay, 1, 1, 4, "number", None
                    )
                elif task == "niah_multiquery":
                    built = _build_niah(
                        tokenizer, rng, context_length, essay, 4, 4, 1, "number", None
                    )
                elif task == "vt":
                    built = _build_vt(tokenizer, rng, context_length)
                elif task == "cwe":
                    built = _build_cwe(tokenizer, rng, context_length)
                else:
                    built = _build_fwe(tokenizer, rng, context_length)

                prompt, answers, metadata = built
                samples.append(
                    Sample(
                        task=task,
                        context_length=context_length,
                        index=index,
                        prompt=prompt,
                        answers=answers,
                        metadata=metadata,
                    )
                )

    return samples
