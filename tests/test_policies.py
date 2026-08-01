"""Tests for KV compression policies.

Selection bugs are silent: a policy that quietly keeps the wrong positions still
produces fluent text, and the damage only shows up as a mysteriously low
benchmark score. These tests pin the invariants that make a selection valid.
"""

from __future__ import annotations

import pytest
import torch

from kvcomp.policies import (
    POLICIES,
    PolicyConfig,
    PyramidKVPolicy,
    _pool,
    _protected_mask,
    build_policy,
)

BATCH = 1
KV_HEADS = 4
SEQ_LEN = 512
BUDGET = 64

SCORED_POLICIES = ["snapkv", "h2o", "tova", "pyramidkv"]
ALL_COMPRESSING = ["streaming_llm", *SCORED_POLICIES]


@pytest.fixture
def scores() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(BATCH, KV_HEADS, SEQ_LEN, generator=generator)


def _select(name: str, scores: torch.Tensor | None, budget: int = BUDGET):
    policy = build_policy(name, PolicyConfig(budget=budget))
    return policy.select(
        scores=scores,
        seq_len=SEQ_LEN,
        budget=budget,
        num_kv_heads=KV_HEADS,
        batch=BATCH,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("name", ALL_COMPRESSING)
def test_select_returns_exact_budget(name, scores):
    indices = _select(name, scores)
    assert indices.shape[-1] == BUDGET, f"{name} must keep exactly `budget` positions"


@pytest.mark.parametrize("name", ALL_COMPRESSING)
def test_indices_are_in_range(name, scores):
    indices = _select(name, scores)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < SEQ_LEN


@pytest.mark.parametrize("name", ALL_COMPRESSING)
def test_indices_are_sorted_ascending(name, scores):
    """The cache stores positions in order; gathering with unsorted indices
    would permute the sequence and scramble relative RoPE geometry."""
    indices = _select(name, scores)
    assert torch.all(indices.diff(dim=-1) > 0), f"{name} produced unsorted/duplicate indices"


@pytest.mark.parametrize("name", ALL_COMPRESSING)
def test_indices_are_unique(name, scores):
    indices = _select(name, scores)
    for head in range(KV_HEADS):
        row = indices[0, head]
        assert row.unique().numel() == row.numel()


@pytest.mark.parametrize("name", SCORED_POLICIES)
def test_sink_positions_are_always_retained(name, scores):
    """Sinks absorb surplus attention mass. Evicting them destabilises the
    softmax and is the single most damaging edit to a cache, so every
    score-based policy must protect them regardless of their score."""
    config = PolicyConfig(budget=BUDGET, sink=8)
    policy = build_policy(name, config)
    # Drive sink scores to zero so only explicit protection can save them.
    weights = torch.rand(BATCH, KV_HEADS, SEQ_LEN) + 1.0
    weights[..., :8] = 0.0

    indices = policy.select(
        scores=weights,
        seq_len=SEQ_LEN,
        budget=BUDGET,
        num_kv_heads=KV_HEADS,
        batch=BATCH,
        device=torch.device("cpu"),
    )
    for head in range(KV_HEADS):
        kept = set(indices[0, head].tolist())
        assert set(range(8)).issubset(kept), f"{name} evicted sink positions"


def test_streaming_llm_keeps_only_sinks_and_recent():
    indices = _select("streaming_llm", None)
    kept = indices[0, 0].tolist()
    config = PolicyConfig(budget=BUDGET)
    expected = list(range(config.sink)) + list(
        range(SEQ_LEN - (BUDGET - config.sink), SEQ_LEN)
    )
    assert kept == expected


def test_streaming_llm_leaves_a_gap_in_the_middle():
    """The structural blind spot StreamingLLM trades away; NIAH depth sweeps
    exist precisely to expose it."""
    kept = set(_select("streaming_llm", None)[0, 0].tolist())
    assert SEQ_LEN // 2 not in kept


def test_snapkv_prefers_high_scoring_positions(scores):
    """A planted spike well outside the protected regions must survive."""
    weights = torch.full((BATCH, KV_HEADS, SEQ_LEN), 0.01)
    spike = 200
    weights[..., spike] = 100.0

    indices = _select("snapkv", weights)
    for head in range(KV_HEADS):
        assert spike in set(indices[0, head].tolist())


def test_h2o_keeps_recent_window():
    config = PolicyConfig(budget=BUDGET, recent=16)
    policy = build_policy("h2o", config)
    weights = torch.rand(BATCH, KV_HEADS, SEQ_LEN) + 1.0
    weights[..., -16:] = 0.0

    indices = policy.select(
        scores=weights,
        seq_len=SEQ_LEN,
        budget=BUDGET,
        num_kv_heads=KV_HEADS,
        batch=BATCH,
        device=torch.device("cpu"),
    )
    kept = set(indices[0, 0].tolist())
    assert set(range(SEQ_LEN - 16, SEQ_LEN)).issubset(kept)


def test_no_compression_when_sequence_fits_budget(scores):
    """Selecting more positions than exist is a hard error, so callers must
    skip compression instead; this documents that boundary."""
    with pytest.raises(RuntimeError):
        build_policy("snapkv", PolicyConfig(budget=1024)).select(
            scores=torch.rand(BATCH, KV_HEADS, 16),
            seq_len=16,
            budget=1024,
            num_kv_heads=KV_HEADS,
            batch=BATCH,
            device=torch.device("cpu"),
        )


def test_full_policy_signals_no_compression():
    policy = build_policy("full", PolicyConfig())
    assert policy.layer_budget(0, 36) < 0


def test_scored_policies_reject_missing_scores():
    for name in SCORED_POLICIES:
        with pytest.raises(ValueError, match="requires captured attention"):
            _select(name, None)


def test_unknown_policy_name_raises():
    with pytest.raises(KeyError, match="unknown policy"):
        build_policy("does_not_exist", PolicyConfig())


class TestPyramidAllocation:
    """PyramidKV must reallocate budget across layers without spending more."""

    def test_budget_decreases_with_depth(self):
        policy = PyramidKVPolicy(PolicyConfig(budget=512, pyramid_min=8))
        budgets = [policy.layer_budget(i, 36) for i in range(36)]
        assert all(a >= b for a, b in zip(budgets, budgets[1:]))
        assert budgets[0] > budgets[-1]

    def test_mean_budget_is_preserved(self):
        """The pyramid shape must not become a stealth budget increase, or
        comparisons against uniform-budget policies would be unfair."""
        policy = PyramidKVPolicy(PolicyConfig(budget=512, pyramid_min=1))
        budgets = [policy.layer_budget(i, 36) for i in range(36)]
        assert sum(budgets) / len(budgets) == pytest.approx(512, rel=0.05)

    def test_floor_is_respected(self):
        policy = PyramidKVPolicy(PolicyConfig(budget=512, pyramid_min=64))
        assert min(policy.layer_budget(i, 36) for i in range(36)) >= 64

    def test_single_layer_model_gets_full_budget(self):
        policy = PyramidKVPolicy(PolicyConfig(budget=512))
        assert policy.layer_budget(0, 1) == 512


class TestHelpers:
    def test_protected_mask_marks_both_ends(self):
        mask = _protected_mask(100, sink=4, recent=8, device=torch.device("cpu"))
        assert mask[:4].all() and mask[-8:].all()
        assert not mask[10:80].any()

    def test_protected_mask_handles_overlap(self):
        mask = _protected_mask(10, sink=8, recent=8, device=torch.device("cpu"))
        assert mask.all()

    def test_pool_preserves_length(self):
        values = torch.rand(1, 2, 100)
        assert _pool(values, 7).shape == values.shape

    def test_pool_spreads_a_spike_to_neighbours(self):
        """Pooling is what makes retained positions form coherent spans rather
        than isolated tokens stripped of their context."""
        values = torch.zeros(1, 1, 50)
        values[0, 0, 25] = 1.0
        pooled = _pool(values, 7)
        assert pooled[0, 0, 23] == pytest.approx(1.0)
        assert pooled[0, 0, 27] == pytest.approx(1.0)
        assert pooled[0, 0, 10] == pytest.approx(0.0)

    def test_pool_of_kernel_one_is_identity(self):
        values = torch.rand(1, 2, 32)
        assert torch.equal(_pool(values, 1), values)


def test_policy_registry_is_complete():
    assert set(POLICIES) == {
        "full",
        "streaming_llm",
        "snapkv",
        "h2o",
        "tova",
        "pyramidkv",
    }


@pytest.mark.parametrize("bad", [{"budget": 1, "window": 0}, {"budget": 1, "chunk": 0}])
def test_capture_request_validates_arguments(bad):
    from kvcomp.attention import CaptureRequest

    with pytest.raises(ValueError):
        CaptureRequest(**{k: v for k, v in bad.items() if k != "budget"})
