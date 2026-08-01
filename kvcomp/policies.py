"""KV-cache compression policies.

Every policy answers one question: given a filled KV cache, which positions do
we keep? Policies differ in what evidence they use and how they spend a fixed
budget across layers.

All policies here are *training-free* and operate after prefill, which is the
setting the published methods target.

Implemented:
    ``full``          No compression. Baseline and quality ceiling.
    ``streaming_llm`` Attention sinks + recent window. Uses no attention scores.
    ``snapkv``        Observation-window scores, pooled, top-k per head.
    ``h2o``           Accumulated attention over all queries + recent window.
    ``tova``          Ranks by the most recent query's attention only.
    ``pyramidkv``     SnapKV scoring with a layer-decreasing budget.

References:
    StreamingLLM  Xiao et al., 2023, arXiv:2309.17453
    H2O           Zhang et al., 2023, arXiv:2306.14048
    SnapKV        Li et al., 2024, arXiv:2404.14469
    TOVA          Oren et al., 2024, arXiv:2401.06104
    PyramidKV     Cai et al., 2024, arXiv:2406.02069
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from .attention import CaptureRequest


@dataclass
class PolicyConfig:
    """Knobs shared by the compression policies.

    Attributes:
        budget: KV positions retained per layer, per KV head. ``None`` disables
            compression. This is the primary quality/memory dial.
        sink: Leading positions always retained. Attention sinks absorb surplus
            attention mass; dropping them destabilises the softmax and is the
            single most damaging edit to a cache.
        recent: Trailing positions always retained, protecting local context.
        window: Observation-window size for score-based policies.
        pool_kernel: Width of the max-pool applied to scores before ranking.
            Neighbouring positions carry overlapping information, so pooling
            keeps a contiguous span rather than isolated spikes.
        chunk: Query-chunk size for policies that accumulate over all queries.
        pyramid_min: Floor on a layer's budget under pyramid allocation.
        pyramid_ratio: First-layer budget divided by last-layer budget under
            pyramid allocation. The mean is held at :attr:`budget` regardless,
            so this controls only how steeply budget is redistributed toward
            early layers, not how much is spent overall.

            A steep ratio starves the deepest layers: at ratio 36 with a
            256-entry budget the last layers fall to ~14 slots and hit
            :attr:`pyramid_min`, which measured 0.083 on RULER above 8k versus
            0.654 below it. Published results do not show that collapse, so the
            default is deliberately gentle.
    """

    budget: int | None = 1024
    sink: int = 16
    recent: int = 128
    window: int = 32
    pool_kernel: int = 7
    chunk: int = 128
    pyramid_min: int = 32
    pyramid_ratio: float = 4.0


class Policy:
    """Base class for compression policies.

    Subclasses declare how they need attention scored (:attr:`capture_mode`,
    ``None`` meaning "no scores required") and implement :meth:`select`.
    """

    name: ClassVar[str] = "base"
    capture_mode: ClassVar[str | None] = None

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def capture_request(self) -> CaptureRequest | None:
        """Build the capture request this policy needs, if any."""
        if self.capture_mode is None:
            return None
        return CaptureRequest(
            mode=self.capture_mode,
            window=self.config.window,
            chunk=self.config.chunk,
        )

    def layer_budget(self, layer_index: int, num_layers: int) -> int:
        """Budget for one layer. Uniform unless a subclass overrides."""
        del layer_index, num_layers
        assert self.config.budget is not None
        return self.config.budget

    def select(
        self,
        scores: torch.Tensor | None,
        seq_len: int,
        budget: int,
        num_kv_heads: int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Choose the positions to retain.

        Args:
            scores: ``[batch, kv_heads, seq_len]`` attention mass, or ``None``
                for score-free policies.
            seq_len: Current cached length.
            budget: Positions to keep for this layer.
            num_kv_heads: Number of KV heads.
            batch: Batch size.
            device: Device for the returned indices.

        Returns:
            Sorted ``int64`` indices of shape ``[batch, kv_heads, budget]``.
            Ascending order preserves the cache's positional layout.
        """
        raise NotImplementedError


def _protected_mask(
    seq_len: int, sink: int, recent: int, device: torch.device
) -> torch.Tensor:
    """Boolean ``[seq_len]`` marking positions that must never be evicted."""
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if sink > 0:
        mask[: min(sink, seq_len)] = True
    if recent > 0:
        mask[max(0, seq_len - recent) :] = True
    return mask


def _pool(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    """Max-pool scores along the position axis, preserving length.

    Isolated high-scoring keys are usually part of a locally informative span.
    Pooling before ranking spreads a position's importance to its neighbours so
    the retained set forms coherent spans instead of scattered tokens.
    """
    if kernel <= 1:
        return scores
    padding = kernel // 2
    pooled = F.max_pool1d(
        scores, kernel_size=kernel, stride=1, padding=padding
    )
    return pooled[..., : scores.shape[-1]]


def _topk_with_protected(
    scores: torch.Tensor,
    seq_len: int,
    budget: int,
    sink: int,
    recent: int,
) -> torch.Tensor:
    """Select ``budget`` positions by score, forcing protected ones in.

    Protected positions are given ``+inf`` score so they always survive, which
    keeps the returned count exactly ``budget`` without a separate merge step.
    """
    protected = _protected_mask(seq_len, sink, recent, scores.device)
    ranked = scores.masked_fill(protected, float("inf"))
    indices = ranked.topk(budget, dim=-1).indices
    return indices.sort(dim=-1).values


class FullPolicy(Policy):
    """No compression. Establishes the quality ceiling and the memory wall."""

    name = "full"
    capture_mode = None

    def layer_budget(self, layer_index: int, num_layers: int) -> int:
        del layer_index, num_layers
        return -1  # sentinel: never compress

    def select(self, scores, seq_len, budget, num_kv_heads, batch, device):
        del scores, budget
        indices = torch.arange(seq_len, device=device)
        return indices.view(1, 1, -1).expand(batch, num_kv_heads, seq_len)


class StreamingLLMPolicy(Policy):
    """Attention sinks plus a recent window, chosen without any scores.

    The cheapest possible policy: it never materialises attention, so prefill
    cost is unchanged. It is also position-only, making it the natural control
    for judging whether score-based selection earns its extra compute.
    """

    name = "streaming_llm"
    capture_mode = None

    def select(self, scores, seq_len, budget, num_kv_heads, batch, device):
        del scores
        sink = min(self.config.sink, budget)
        recent = budget - sink
        head = torch.arange(sink, device=device)
        tail = torch.arange(max(sink, seq_len - recent), seq_len, device=device)
        indices = torch.cat([head, tail])[:budget]
        return indices.view(1, 1, -1).expand(batch, num_kv_heads, indices.numel())


class SnapKVPolicy(Policy):
    """Rank positions by how much the prompt's tail attends to them.

    The observation window acts as a stand-in for future generation queries.
    Because scoring touches only the last ``window`` queries, the overhead is
    linear in sequence length rather than quadratic.
    """

    name = "snapkv"
    capture_mode = "window"

    def select(self, scores, seq_len, budget, num_kv_heads, batch, device):
        if scores is None:
            raise ValueError("snapkv requires captured attention scores")
        return _topk_with_protected(
            _pool(scores, self.config.pool_kernel),
            seq_len,
            budget,
            self.config.sink,
            self.config.window,
        )


class H2OPolicy(Policy):
    """Keep 'heavy hitters': positions with the largest accumulated attention.

    Scores are summed over every query in the prompt, so the statistic is
    quadratic in sequence length. That cost is the method's main practical
    drawback and one of the things this benchmark is built to measure.
    """

    name = "h2o"
    capture_mode = "full"

    def select(self, scores, seq_len, budget, num_kv_heads, batch, device):
        if scores is None:
            raise ValueError("h2o requires captured attention scores")
        return _topk_with_protected(
            scores, seq_len, budget, self.config.sink, self.config.recent
        )


class TOVAPolicy(Policy):
    """Rank by the final query's attention alone.

    TOVA is the ``window=1`` limit of observation-window scoring: a single
    query decides the ranking, with no pooling and no protected recent span
    beyond the sinks.
    """

    name = "tova"
    capture_mode = "window"

    def capture_request(self) -> CaptureRequest | None:
        return CaptureRequest(mode="window", window=1, chunk=self.config.chunk)

    def select(self, scores, seq_len, budget, num_kv_heads, batch, device):
        if scores is None:
            raise ValueError("tova requires captured attention scores")
        return _topk_with_protected(scores, seq_len, budget, self.config.sink, 1)


class PyramidKVPolicy(SnapKVPolicy):
    """SnapKV scoring with a budget that shrinks with depth.

    Lower layers attend broadly while upper layers concentrate on a few
    positions, so a uniform per-layer budget overspends at the top. Budgets form
    a decreasing arithmetic sequence whose mean equals the configured budget,
    leaving total memory unchanged while reallocating it toward early layers.
    """

    name = "pyramidkv"

    def layer_budget(self, layer_index: int, num_layers: int) -> int:
        assert self.config.budget is not None
        budget = self.config.budget
        if num_layers <= 1:
            return budget

        # Arithmetic sequence from `high` down to `low`, chosen so that
        # high / low == pyramid_ratio and (high + low) / 2 == budget. Holding
        # the mean fixed keeps total memory identical to a uniform allocation,
        # so pyramid and uniform policies stay directly comparable.
        ratio = max(1.0, self.config.pyramid_ratio)
        low = 2.0 * budget / (ratio + 1.0)
        high = low * ratio

        position = layer_index / (num_layers - 1)
        allocated = high - (high - low) * position

        floor = min(self.config.pyramid_min, budget)
        return max(floor, int(round(allocated)))


POLICIES: dict[str, type[Policy]] = {
    policy.name: policy
    for policy in (
        FullPolicy,
        StreamingLLMPolicy,
        SnapKVPolicy,
        H2OPolicy,
        TOVAPolicy,
        PyramidKVPolicy,
    )
}


def build_policy(name: str, config: PolicyConfig) -> Policy:
    """Instantiate a policy by name.

    Raises:
        KeyError: If ``name`` is not a registered policy.
    """
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(POLICIES)}")
    return POLICIES[name](config)
