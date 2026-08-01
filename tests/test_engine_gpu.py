"""End-to-end GPU tests.

Opt-in via ``pytest -m gpu``; they need a CUDA device and downloaded weights.

These cover the failures that unit tests structurally cannot catch: the ones
that only appear when real attention kernels, a real cache, and real positional
encodings interact. Every bug found while building this engine was of that kind
-- each produced correct-looking output right up until it OOMed or silently
degraded quality.
"""

from __future__ import annotations

import pytest
import torch

from kvcomp.engine import EngineConfig, KVCompressionEngine
from kvcomp.policies import PolicyConfig

pytestmark = pytest.mark.gpu

LONG_CONTEXT_TOKENS = 8192


@pytest.fixture(scope="module")
def engine() -> KVCompressionEngine:
    """Load the model once; it is the dominant cost of this module."""
    return KVCompressionEngine(EngineConfig(max_new_tokens=16))


@pytest.fixture(scope="module")
def long_prompt(engine: KVCompressionEngine) -> str:
    """A long prompt with a retrievable fact in the middle."""
    filler = " ".join(
        f"Sentence {i}: the weather report for day {i} was unremarkable."
        for i in range(4000)
    )
    needle = "The vault access code is 84213."
    midpoint = len(filler) // 2
    document = f"{filler[:midpoint]} {needle} {filler[midpoint:]}"
    return engine.build_prompt(
        f"{document}\n\nQuestion: What is the vault access code? "
        "Answer with the number only."
    )


def test_model_geometry_matches_config(engine: KVCompressionEngine):
    assert engine.num_layers > 0
    assert engine.num_kv_heads > 0
    assert engine.bytes_per_token() > 0


def test_short_generation_works(engine: KVCompressionEngine):
    record = engine.generate(engine.build_prompt("Name the capital of France."), "full")
    assert record.error is None
    assert "paris" in record.text.lower()


def test_long_prefill_does_not_exhaust_memory(engine: KVCompressionEngine, long_prompt: str):
    """Regression test for the GQA/SDPA fallback.

    With ``enable_gqa=True`` the fused kernels reject the input and SDPA drops to
    the MATH backend, materialising a [heads, q, kv] attention matrix -- 8 GiB at
    8k tokens on a 6 GB card. The prompt below is large enough that the old path
    could not complete it at all.
    """
    record = engine.generate(long_prompt, "full", max_new_tokens=8)
    assert not record.oom, "long prefill OOMed; the SDPA fast path is not being taken"
    assert record.prompt_tokens > LONG_CONTEXT_TOKENS
    assert record.peak_vram_bytes < 6 * 2**30


@pytest.mark.parametrize(
    "policy", ["full", "streaming_llm", "snapkv", "pyramidkv", "tova", "h2o"]
)
def test_every_policy_runs(engine: KVCompressionEngine, long_prompt: str, policy: str):
    record = engine.generate(
        long_prompt, policy, PolicyConfig(budget=512), max_new_tokens=8
    )
    assert record.error is None, f"{policy} failed: {record.error}"
    assert record.generated_tokens > 0


@pytest.mark.parametrize("policy", ["streaming_llm", "snapkv", "pyramidkv", "h2o"])
def test_compression_actually_shrinks_the_cache(
    engine: KVCompressionEngine, long_prompt: str, policy: str
):
    record = engine.generate(
        long_prompt, policy, PolicyConfig(budget=512), max_new_tokens=8
    )
    assert record.cache_tokens_after < record.cache_tokens_before
    assert record.compression_ratio > 0.8


def test_full_policy_leaves_the_cache_intact(
    engine: KVCompressionEngine, long_prompt: str
):
    record = engine.generate(long_prompt, "full", max_new_tokens=8)
    assert record.compression_ratio == 0.0
    assert record.cache_tokens_after == record.cache_tokens_before


def test_smaller_budget_uses_less_cache_memory(
    engine: KVCompressionEngine, long_prompt: str
):
    small = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=128), max_new_tokens=8)
    large = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=1024), max_new_tokens=8)
    assert small.cache_bytes < large.cache_bytes


def test_snapkv_retrieves_a_mid_context_needle(
    engine: KVCompressionEngine, long_prompt: str
):
    """Score-based selection should find a needle that positional heuristics miss.

    This is the substantive claim of the whole method: attention scores identify
    which distant positions still matter.
    """
    record = engine.generate(
        long_prompt, "snapkv", PolicyConfig(budget=1024), max_new_tokens=16
    )
    assert "84213" in record.text


def test_compression_preserves_short_prompt_output_exactly(
    engine: KVCompressionEngine,
):
    """Below the budget nothing may be evicted, so output must match ``full``.

    Any divergence here means the cache is being mutated when it should not be,
    or that positions shifted -- both silent correctness bugs.
    """
    prompt = engine.build_prompt("List the first five prime numbers in order.")
    baseline = engine.generate(prompt, "full", max_new_tokens=16)
    compressed = engine.generate(
        prompt, "snapkv", PolicyConfig(budget=4096), max_new_tokens=16
    )
    assert compressed.text == baseline.text


def test_timings_are_recorded(engine: KVCompressionEngine, long_prompt: str):
    record = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=8)
    assert record.prefill_seconds > 0
    assert record.decode_seconds > 0
    assert record.total_seconds >= record.prefill_seconds


def test_h2o_prefill_is_slower_than_snapkv(
    engine: KVCompressionEngine, long_prompt: str
):
    """H2O accumulates attention over every query, SnapKV over a fixed window.

    Quantifying that gap is one of the benchmark's purposes, so the ordering is
    asserted rather than assumed.
    """
    snapkv = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=4)
    h2o = engine.generate(long_prompt, "h2o", PolicyConfig(budget=512), max_new_tokens=4)
    assert h2o.prefill_seconds > snapkv.prefill_seconds


class TestQueryAwareChunking:
    """Chunked prefill must not blind query-aware policies to the question.

    SnapKV, PyramidKV and TOVA rank cached positions by how much an observation
    window attends to them, and that window stands in for the question. When a
    prompt spans several chunks, compressing after each one means early
    evictions happen while the only queries seen are mid-document filler -- the
    answer is discarded before the question is ever read.

    The failure is silent and looks like the method being weak at long context.
    Measured on RULER `niah_single_1` at a 256-entry budget, accuracy tracked
    chunk count exactly: 1.00 at one chunk, 1.00 at two, 0.00 at four, with the
    model replying that the needle "is not provided in the text". Probing the
    cache with the prompt's tail before each eviction restored 1.00.
    """

    @pytest.fixture(scope="class")
    def multi_chunk_prompt(self, engine: KVCompressionEngine) -> str:
        """A prompt several chunks long whose answer sits far from the tail."""
        filler = " ".join(
            f"Sentence {i}: the day {i} report was unremarkable." for i in range(3000)
        )
        needle = "The special magic number for ivory-457 is: 3084821."
        cut = len(filler) // 3
        document = f"{filler[:cut]} {needle} {filler[cut:]}"
        return engine.build_prompt(
            f"{document}\n\nQuestion: What is the special magic number for "
            "ivory-457? Answer with the number only."
        )

    @pytest.mark.parametrize("policy", ["snapkv", "pyramidkv", "tova"])
    def test_needle_survives_multi_chunk_prefill(
        self, engine: KVCompressionEngine, multi_chunk_prompt: str, policy: str
    ):
        engine.config.prefill_chunk = 2048
        engine.config.query_aware_chunks = True
        record = engine.generate(
            multi_chunk_prompt, policy, PolicyConfig(budget=256), max_new_tokens=24
        )
        assert record.error is None
        assert "3084821" in record.text, (
            f"{policy} lost the needle across chunk boundaries; "
            "query-aware probing is not working"
        )

    def test_probe_leaves_no_residue_in_the_cache(
        self, engine: KVCompressionEngine, multi_chunk_prompt: str
    ):
        """The probe encodes the tail to score with, then must crop its own
        entries -- otherwise the tail is cached twice and the budget is wrong."""
        engine.config.prefill_chunk = 2048
        engine.config.query_aware_chunks = True
        record = engine.generate(
            multi_chunk_prompt, "snapkv", PolicyConfig(budget=256), max_new_tokens=8
        )
        assert record.cache_tokens_after <= 256 + 2048

    def test_probe_is_cropped_from_every_layer_under_uneven_budgets(
        self, engine: KVCompressionEngine, multi_chunk_prompt: str
    ):
        """Each layer must be trimmed to *its own* pre-probe length.

        `Cache.crop` trims all layers to one global length. Under PyramidKV's
        per-layer budgets the layers legitimately differ, so a global crop
        leaves the probe's keys behind in every layer shorter than the longest
        one. They sit at the end of the cache, where the recent-window rule
        protects them, so each chunk implants another copy of the question and
        evicts real content -- 16k output degraded to fluent nonsense while
        uniform-budget policies looked fine.
        """
        from kvcomp.engine import cache_seq_lengths  # local import: test-only
        from kvcomp.policies import PolicyConfig as PC
        from kvcomp.policies import build_policy

        engine.config.prefill_chunk = 2048
        engine.config.query_aware_chunks = True

        observed: list[list[int]] = []
        original = type(engine)._compress

        def record_lengths(self, cache, policy, capture, batch):
            observed.append(cache_seq_lengths(cache))
            return original(self, cache, policy, capture, batch)

        type(engine)._compress = record_lengths
        try:
            engine.generate(
                multi_chunk_prompt, "pyramidkv", PC(budget=1024), max_new_tokens=8
            )
        finally:
            type(engine)._compress = original

        policy = build_policy("pyramidkv", PC(budget=1024))
        budgets = [policy.layer_budget(i, engine.num_layers) for i in range(engine.num_layers)]

        # From the second chunk onward each layer should hold exactly its own
        # budget plus the new chunk -- no probe remainder.
        for lengths in observed[1:-1]:
            for index, (length, budget) in enumerate(zip(lengths, budgets)):
                assert length == budget + 2048, (
                    f"layer {index} holds {length}, expected {budget + 2048}; "
                    "probe entries were not cropped"
                )

    def test_single_chunk_prompts_skip_probing(
        self, engine: KVCompressionEngine
    ):
        """A prompt inside one chunk already ends with the question, so probing
        would be pure overhead."""
        engine.config.prefill_chunk = 4096
        engine.config.query_aware_chunks = True
        prompt = engine.build_prompt("What is 2 + 2? Answer with the number only.")
        record = engine.generate(prompt, "snapkv", PolicyConfig(budget=256), max_new_tokens=8)
        assert record.error is None


def test_oom_is_reported_not_raised(engine: KVCompressionEngine):
    """A sweep must survive an impossible cell rather than dying on it."""
    absurd = engine.build_prompt("word " * 400_000)
    record = engine.generate(absurd, "full", max_new_tokens=4)
    assert record.error is not None
    assert isinstance(record.peak_vram_bytes, int)


def test_greedy_decoding_is_deterministic(engine: KVCompressionEngine, long_prompt: str):
    first = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=12)
    second = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=12)
    assert first.text == second.text


def test_cache_memory_matches_the_analytic_estimate(
    engine: KVCompressionEngine, long_prompt: str
):
    """Guards the bytes-per-token figure used for pre-run feasibility checks."""
    budget = 512
    record = engine.generate(long_prompt, "snapkv", PolicyConfig(budget=budget), max_new_tokens=4)
    expected = engine.bytes_per_token() * budget
    assert record.cache_bytes == pytest.approx(expected, rel=0.05)


def test_no_memory_is_leaked_across_runs(engine: KVCompressionEngine, long_prompt: str):
    """Caches must be released between sweep cells, or later cells OOM for
    reasons that have nothing to do with their own configuration."""
    engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=4)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    for _ in range(3):
        engine.generate(long_prompt, "snapkv", PolicyConfig(budget=512), max_new_tokens=4)
    torch.cuda.empty_cache()

    assert torch.cuda.memory_allocated() - baseline < 256 * 2**20
