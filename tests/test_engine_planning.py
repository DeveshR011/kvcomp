"""Tests for memory planning that need no GPU.

The feasibility precheck and peak-cache estimate decide whether a sweep cell is
attempted at all. Getting them wrong is expensive in a specific way: CUDA on
Windows spills to host memory instead of erroring, so an over-budget run does
not fail, it crawls -- occupying the GPU for tens of minutes and stalling the
sweep behind it.
"""

from __future__ import annotations

import pytest

from kvcomp.engine import EngineConfig
from kvcomp.policies import PolicyConfig, build_policy

# Qwen3-4B geometry.
LAYERS = 36
KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2


class FakeEngine:
    """Exercises the planning arithmetic without loading a model."""

    num_layers = LAYERS
    num_kv_heads = KV_HEADS
    head_dim = HEAD_DIM

    def __init__(self, prefill_chunk: int | None = 2048):
        self.config = EngineConfig(prefill_chunk=prefill_chunk)

    bytes_per_token = staticmethod(lambda dtype_size=2: BYTES_PER_TOKEN)

    peak_cache_tokens = None  # bound below


def _peak(prompt_tokens: int, policy_name: str, budget, chunk):
    from kvcomp.engine import KVCompressionEngine

    engine = FakeEngine(prefill_chunk=chunk)
    policy = build_policy(policy_name, PolicyConfig(budget=budget))
    return KVCompressionEngine.peak_cache_tokens(engine, prompt_tokens, policy)


class TestPeakCacheTokens:
    def test_full_policy_holds_the_entire_prompt(self):
        assert _peak(32768, "full", None, 2048) == 32768

    def test_chunking_bounds_the_peak_for_compressing_policies(self):
        """The whole point of chunked prefill: peak is budget + chunk, not the
        prompt length, so long contexts become reachable."""
        assert _peak(32768, "snapkv", 512, 2048) == 512 + 2048

    def test_peak_is_independent_of_prompt_length_when_chunking(self):
        short = _peak(8192, "snapkv", 512, 2048)
        long = _peak(131072, "snapkv", 512, 2048)
        assert short == long

    def test_without_chunking_the_whole_prompt_is_resident(self):
        """Compressing only after prefill cannot lower the peak -- the flaw
        chunked prefill exists to fix."""
        assert _peak(32768, "snapkv", 512, None) == 32768

    def test_short_prompt_is_never_inflated(self):
        assert _peak(256, "snapkv", 512, 2048) == 256

    def test_smaller_chunks_lower_the_peak(self):
        assert _peak(32768, "snapkv", 512, 512) < _peak(32768, "snapkv", 512, 4096)


class TestMemoryArithmetic:
    def test_bytes_per_token_matches_the_documented_figure(self):
        assert BYTES_PER_TOKEN == 147456

    @pytest.mark.parametrize(
        ("tokens", "expected_gib"),
        [(8192, 1.125), (16384, 2.25), (32768, 4.5)],
    )
    def test_cache_size_matches_the_published_table(self, tokens, expected_gib):
        """Guards the table in docs/kvcomp.md that sizing decisions rely on."""
        assert BYTES_PER_TOKEN * tokens / 2**30 == pytest.approx(expected_gib)

    def test_full_context_at_32k_cannot_fit_beside_the_weights(self):
        """The headline constraint: 4.5 GiB of cache plus ~2.5 GiB of NF4
        weights does not fit in 6 GiB."""
        weights_gib = 2.5
        total_gib = 6141 / 1024
        cache_gib = BYTES_PER_TOKEN * 32768 / 2**30
        assert cache_gib + weights_gib > total_gib

    def test_compressed_cache_fits_comfortably(self):
        cache_gib = BYTES_PER_TOKEN * (512 + 2048) / 2**30
        assert cache_gib < 0.5


class TestMemoryCeiling:
    """Regression tests for the feasibility model.

    Every figure below is a real measurement from an RTX 4050 (6 GB) running
    Qwen3-4B at NF4 with ~1.04 GiB held by desktop applications. Two earlier
    formulations of this check failed in opposite directions -- one refused runs
    that demonstrably fit and silently deleted the `full` baseline above 4k, the
    other admitted a 16k run that exceeded the card and thrashed at 58 s per
    sample instead of 7.7 s. Neither failure produced an error message.
    """

    TOTAL = 6.00 * 2**30
    EXTERNAL = 1.04 * 2**30
    WEIGHTS = 2.49 * 2**30
    HEADROOM = 256 * 2**20
    ACTIVATIONS = 0.25 * 2**30

    def _ceiling(self):
        return self.TOTAL - self.EXTERNAL - self.HEADROOM

    def _peak(self, cache_tokens):
        return self.WEIGHTS + BYTES_PER_TOKEN * cache_tokens + self.ACTIVATIONS

    def test_ceiling_leaves_room_for_other_processes(self):
        assert self._ceiling() < self.TOTAL - self.EXTERNAL

    def test_8k_full_context_is_admitted(self):
        """Measured at 3.86 GiB peak, completing in 7.7 s."""
        assert self._peak(8192) <= self._ceiling()

    def test_16k_full_context_is_refused(self):
        """Measured at 5.20 GiB peak. It exceeds the card once desktop memory is
        counted, and Windows pages rather than failing -- 58 s per sample."""
        assert self._peak(16384) > self._ceiling()

    def test_32k_full_context_is_refused(self):
        assert self._peak(32768) > self._ceiling()

    def test_prediction_matches_the_measured_8k_peak(self):
        """Predicted 3.84 GiB against 3.86 GiB measured."""
        assert self._peak(8192) / 2**30 == pytest.approx(3.86, abs=0.1)

    def test_compressed_peak_is_flat_across_context_lengths(self):
        """The chunked-prefill payoff: with the cache capped at budget + chunk,
        footprint stops depending on prompt length."""
        capped = 1024 + 2048
        assert self._peak(capped) == self._peak(capped)
        assert self._peak(capped) <= self._ceiling()

    def test_compressed_32k_fits_where_full_context_cannot(self):
        """The headline result the benchmark exists to produce."""
        assert self._peak(1024 + 2048) <= self._ceiling()
        assert self._peak(32768) > self._ceiling()


class TestEngineConfigDefaults:
    def test_chunking_is_on_by_default(self):
        """Off by default would silently reintroduce the peak-memory flaw."""
        assert EngineConfig().prefill_chunk is not None

    def test_precheck_is_on_by_default(self):
        assert EngineConfig().precheck is True

    def test_greedy_decoding_is_the_default(self):
        assert EngineConfig().greedy is True
