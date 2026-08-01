"""A drop-in attention implementation that can capture attention scores.

Compression policies such as SnapKV and H2O need to know how much attention
each cached key position receives. Standard SDPA/flash kernels never
materialise the attention matrix, so those scores are unavailable.

Rather than monkeypatching the model's attention math (fragile across
Transformers releases), we register an additional attention implementation in
``ALL_ATTENTION_FUNCTIONS``. Transformers looks the implementation up by name
via ``config._attn_implementation``, so selecting it is a one-line change and
the surrounding model code stays untouched.

The registered function computes attention with SDPA exactly as the stock path
does. Its only extra behaviour is that, when a capture request is attached to
the attention module, it *additionally* computes the scores the policy asked
for and stores them on the module. Attention output is bit-identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

KVCOMP_ATTENTION = "kvcomp"

#: Attention-score accumulation strategies.
#: ``window`` scores only the final ``window`` query positions (SnapKV, PyramidKV).
#: ``full`` accumulates over every query position (H2O, TOVA).
CAPTURE_MODES = ("window", "full")


@dataclass
class CaptureRequest:
    """Instruction to collect attention statistics during a forward pass.

    A single request object is shared across all attention modules of a model.
    Each layer writes its own result into :attr:`scores` keyed by layer index.

    Attributes:
        mode: One of :data:`CAPTURE_MODES`.
        window: Number of trailing query positions to score when ``mode="window"``.
        chunk: Query-chunk size used when ``mode="full"``. Bounds peak memory of
            the materialised attention block to ``heads * chunk * keys`` elements.
        normalize: Divide accumulated mass by the number of queries that could
            attend to each position. Only meaningful for ``mode="full"``.

            Defaults to ``False``, matching H2O's published cumulative-sum
            formulation. The option exists because that formulation has an
            obvious positional bias -- causal masking lets ``N - j`` queries see
            position ``j``, so a raw sum rewards early positions for being early
            -- but normalising does not fix it, it inverts it. Late positions
            are seen by few queries, and those queries are adjacent, so their
            mean attention is high; measured retention shifted from the start of
            the document to its end and the score stayed at 0.00.

            Neither variant recovers a mid-context needle. The mismatch is
            structural rather than a tuning problem, so the faithful version is
            the default. See ``docs/kvcomp.md``.
        scores: Per-layer tensors of shape ``[batch, kv_heads, keys]`` holding the
            summed attention mass received by each cached position, already
            reduced across the query axis and averaged within each GQA group.
    """

    mode: str = "window"
    window: int = 32
    chunk: int = 128
    normalize: bool = False
    scores: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in CAPTURE_MODES:
            raise ValueError(f"mode must be one of {CAPTURE_MODES}, got {self.mode!r}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {self.chunk}")

    def reset(self) -> None:
        """Drop collected scores so the request can be reused for another pass."""
        self.scores.clear()


def _group_reduce(scores: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Average per-query-head scores within each GQA group.

    The KV cache stores one entry per *key/value* head, while attention scores
    are produced per *query* head. Under grouped-query attention several query
    heads share a single KV head, so a pruning decision must be made jointly for
    the whole group. We average the group's scores, which keeps a position alive
    if any meaningful fraction of its group still attends to it.

    Args:
        scores: ``[batch, query_heads, keys]``.
        num_key_value_groups: Query heads per KV head.

    Returns:
        ``[batch, kv_heads, keys]``.
    """
    batch, query_heads, keys = scores.shape
    kv_heads = query_heads // num_key_value_groups
    return scores.view(batch, kv_heads, num_key_value_groups, keys).mean(dim=2)


def _causal_window_mask(
    query_len: int, key_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Build the causal mask for the final ``window`` queries of a prefill.

    During prefill the trailing ``window`` queries sit at absolute positions
    ``key_len - window ... key_len - 1``. Query ``i`` of the window may attend to
    key ``j`` only when ``j <= key_len - window + i``.

    Returns:
        Boolean ``[window, key_len]`` tensor, ``True`` where attention is allowed.
    """
    del query_len
    offset = key_len - window
    keys = torch.arange(key_len, device=device).unsqueeze(0)
    queries = torch.arange(window, device=device).unsqueeze(1) + offset
    return keys <= queries


def _capture_window_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
    window: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Score cached positions using only the final ``window`` queries.

    This is the SnapKV observation-window formulation: the tail of the prompt is
    treated as a proxy for the queries the model will issue during generation,
    so positions that tail attends to are the ones worth keeping.

    Args:
        query: ``[batch, query_heads, q_len, head_dim]``, post-RoPE.
        key: ``[batch, kv_heads, k_len, head_dim]``, post-RoPE, full cache.
        scaling: Softmax temperature (``head_dim ** -0.5``).
        window: Number of trailing queries to use.
        num_key_value_groups: Query heads per KV head.

    Returns:
        ``[batch, kv_heads, k_len]`` summed attention mass.
    """
    key_len = key.shape[-2]
    window = min(window, query.shape[-2])

    observed = query[:, :, -window:, :]
    expanded_key = repeat_kv(key, num_key_value_groups)

    logits = torch.matmul(observed, expanded_key.transpose(-1, -2)) * scaling
    mask = _causal_window_mask(window, key_len, window, logits.device)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return _group_reduce(weights.sum(dim=-2), num_key_value_groups)


def _capture_full_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
    chunk: int,
    num_key_value_groups: int,
    normalize: bool = False,
) -> torch.Tensor:
    """Accumulate attention mass over *every* query position.

    This is the H2O heavy-hitter statistic. It is quadratic in sequence length,
    so queries are processed in chunks and only the running column sum is kept.
    Peak extra memory is ``batch * query_heads * chunk * key_len`` floats rather
    than the full attention matrix.

    Args and returns match :func:`_capture_window_scores`.
    """
    batch, query_heads, q_len, _ = query.shape
    key_len = key.shape[-2]
    offset = key_len - q_len

    expanded_key = repeat_kv(key, num_key_value_groups)
    totals = torch.zeros(batch, query_heads, key_len, device=query.device, dtype=torch.float32)
    key_index = torch.arange(key_len, device=query.device).unsqueeze(0)

    for start in range(0, q_len, chunk):
        stop = min(start + chunk, q_len)
        block = query[:, :, start:stop, :]

        logits = torch.matmul(block, expanded_key.transpose(-1, -2)) * scaling
        positions = torch.arange(start, stop, device=query.device).unsqueeze(1) + offset
        logits = logits.masked_fill(~(key_index <= positions), torch.finfo(logits.dtype).min)

        totals += torch.softmax(logits, dim=-1, dtype=torch.float32).sum(dim=-2)
        del logits

    if normalize:
        # Queries at positions `offset .. offset + q_len - 1` can see key j only
        # when their position is at least j, so the count falls off linearly
        # across the sequence. Dividing it out converts the running sum into
        # mean attention per attending query.
        keys = torch.arange(key_len, device=query.device, dtype=torch.float32)
        attending = (offset + q_len - torch.clamp(keys, min=float(offset))).clamp(min=1.0)
        totals = totals / attending

    return _group_reduce(totals, num_key_value_groups)


def kvcomp_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """SDPA attention that optionally records per-position attention mass.

    Registered under the name :data:`KVCOMP_ATTENTION`. The attention output is
    produced by the stock SDPA kernel and is unaffected by capture.

    Capture is driven by ``module._kvcomp_capture``; when that attribute is
    absent or ``None`` this is exactly ``sdpa_attention_forward``. Scores are
    only collected for multi-token (prefill) passes, since single-token decode
    steps carry no useful ranking signal.
    """
    request: CaptureRequest | None = getattr(module, "_kvcomp_capture", None)

    if request is not None and query.shape[-2] > 1:
        groups = getattr(module, "num_key_value_groups", 1)
        temperature = scaling if scaling is not None else query.shape[-1] ** -0.5

        with torch.no_grad():
            if request.mode == "window":
                scores = _capture_window_scores(
                    query, key, temperature, request.window, groups
                )
            else:
                scores = _capture_full_scores(
                    query, key, temperature, request.chunk, groups, request.normalize
                )
        request.scores[module.layer_idx] = scores

    # Expand KV heads explicitly instead of relying on SDPA's `enable_gqa`.
    #
    # This is not a micro-optimisation. With `enable_gqa=True` neither fused
    # kernel accepts the input ("both fused kernels require query, key and value
    # to have the same num_heads"), so SDPA silently falls back to the MATH
    # backend and materialises the full [batch, heads, q, kv] attention matrix.
    # Measured on an RTX 4050 at 8221 tokens: 8.06 GiB via enable_gqa versus
    # 64 MiB with an explicit repeat -- a 128x difference that is the sole
    # reason long contexts fit on a 6 GB card at all. The fallback is silent,
    # costs no accuracy, and simply presents as an OOM.
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]

    if is_causal is None:
        # A single decode query attends to the whole cache, so causal masking is
        # neither needed nor correct once positions have been evicted.
        is_causal = (
            query.shape[-2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def register_kvcomp_attention() -> None:
    """Make :data:`KVCOMP_ATTENTION` selectable as ``attn_implementation``.

    Idempotent, so it is safe to call at import time and again per model load.
    """
    if KVCOMP_ATTENTION not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(KVCOMP_ATTENTION, kvcomp_attention_forward)

    # Transformers picks a mask builder by implementation *name*. An unknown
    # name silently falls back to the eager path, which materialises a dense
    # [batch, heads, q, kv] mask -- 4 GiB at 8k tokens, and an OOM long before
    # the KV cache is the binding constraint. Since we delegate to SDPA, we must
    # claim SDPA's mask builder, which returns None and lets the kernel apply
    # causality internally.
    if KVCOMP_ATTENTION not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(KVCOMP_ATTENTION, sdpa_mask)


def attach_capture(model: torch.nn.Module, request: CaptureRequest | None) -> None:
    """Attach (or clear) a capture request on every attention module.

    Args:
        model: A causal LM whose attention modules expose ``layer_idx``.
        request: Request to install, or ``None`` to disable capture.
    """
    for module in model.modules():
        if hasattr(module, "num_key_value_groups") and hasattr(module, "layer_idx"):
            module._kvcomp_capture = request
