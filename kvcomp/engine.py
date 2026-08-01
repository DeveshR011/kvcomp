"""Model loading and a compression-aware generation loop.

Why this module does not use ``model.generate``
-----------------------------------------------
Compressing a KV cache breaks an assumption baked into the stock generation
path: that a cached position's index equals its position in the text. After
eviction the cache holds, say, 1024 entries drawn from a 32k-token prompt, so
``cache.get_seq_length()`` returns 1024 while the next token genuinely belongs
at position 32000. ``generate`` derives RoPE positions from the cache length and
would place that token at 1024 — silently corrupting every rotary phase and
degrading quality in a way that looks like the compression method failing.

So the loop below tracks the true text position independently of cache
occupancy and passes ``position_ids`` explicitly. Cached keys keep the rotary
phases they were built with, which is what the published methods specify.

Owning the loop also makes prefill and decode separately measurable — the split
matters here because policies trade prefill cost for decode memory.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

from .attention import KVCOMP_ATTENTION, attach_capture, register_kvcomp_attention
from .policies import Policy, PolicyConfig, build_policy


@dataclass
class EngineConfig:
    """Model-loading and generation settings.

    Attributes:
        model_id: HuggingFace repo id.
        quantization: ``"nf4"``, ``"int8"``, or ``"none"``. On a 6 GB card NF4 is
            effectively mandatory: it frees the headroom the KV cache needs.
        dtype: Compute dtype for non-quantized tensors.
        device: Target device.
        max_new_tokens: Decode cap.
        greedy: Use greedy decoding. Benchmarks need determinism, so this
            defaults to ``True``.
        trust_remote_code: Passed through to the loaders.
        prefill_chunk: Tokens per prefill step. Compression runs after each
            chunk, so the cache never exceeds ``budget + prefill_chunk``. This
            is what makes compression reduce *peak* memory rather than only
            steady-state memory -- see :meth:`KVCompressionEngine._prefill`.
            ``None`` prefills in one pass.
        vram_headroom_bytes: Fixed memory reserved for fragmentation and the
            transient KV-head expansion, on top of the per-chunk activation
            estimate. See :meth:`KVCompressionEngine.available_cache_bytes`.
        precheck: Refuse runs whose predicted cache cannot fit. Windows lets
            CUDA spill into shared host memory, so an over-budget run does not
            fail fast -- it crawls at a fraction of normal speed and can occupy
            the GPU for tens of minutes. Predicting the failure keeps a sweep
            moving.
    """

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization: str = "nf4"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    max_new_tokens: int = 64
    greedy: bool = True
    trust_remote_code: bool = False
    prefill_chunk: int | None = 2048
    # Calibrated against a measured run rather than guessed. Qwen3-4B at NF4
    # prefilling 8,168 tokens peaked at 3.86 GiB: 2.49 GiB weights, 1.12 GiB
    # cache, 0.25 GiB activations. Activations are estimated separately, so this
    # only needs to cover fragmentation and allocator slack. An earlier 1.2 GiB
    # value was roughly 5x too conservative and refused runs that demonstrably
    # fit, silently removing the `full` baseline above 4k.
    vram_headroom_bytes: int = 256 * 2**20
    precheck: bool = True
    query_aware_chunks: bool = True
    # Tokens of the prompt tail used to probe the cache before each eviction.
    # Deliberately independent of a policy's scoring window: the probe must be
    # long enough to contain the question, while the window controls how many of
    # its queries are scored. Tying them together starves TOVA, whose window is
    # a single token -- probing with one trailing token dropped it from 1.00 to
    # 0.00 on RULER `niah_single_1` at 4k.
    query_probe_tokens: int = 48


@dataclass
class GenerationRecord:
    """Measurements from a single prompt.

    Attributes:
        text: Decoded completion.
        prompt_tokens: Prompt length in tokens.
        generated_tokens: Number of tokens produced.
        prefill_seconds: Wall time for the prompt forward pass, including score
            capture but excluding compression.
        compress_seconds: Wall time spent selecting and applying eviction.
        decode_seconds: Wall time for the token-by-token loop.
        cache_tokens_before: Cached positions per layer prior to compression.
        cache_tokens_after: Cached positions per layer after compression,
            averaged across layers (pyramid budgets vary by layer).
        cache_bytes: Cache footprint after compression.
        peak_vram_bytes: Peak allocated CUDA memory across the whole call.
        oom: Whether the run aborted with a CUDA OOM.
        error: Error text when the run failed.
    """

    text: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    compress_seconds: float = 0.0
    decode_seconds: float = 0.0
    cache_tokens_before: int = 0
    cache_tokens_after: int = 0
    cache_bytes: int = 0
    peak_vram_bytes: int = 0
    oom: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.compress_seconds + self.decode_seconds

    @property
    def compression_ratio(self) -> float:
        """Cached positions removed, as a fraction. ``0.0`` means uncompressed."""
        if self.cache_tokens_before == 0:
            return 0.0
        return 1.0 - (self.cache_tokens_after / self.cache_tokens_before)

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.decode_seconds


def _quantization_config(mode: str) -> BitsAndBytesConfig | None:
    """Build a bitsandbytes config, or ``None`` for full precision."""
    if mode == "none":
        return None
    if mode == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Quantizing the quantization constants; small extra saving that
            # matters when the budget is this tight.
            bnb_4bit_use_double_quant=True,
        )
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unknown quantization {mode!r}; expected nf4, int8, or none")


def cache_num_bytes(cache: DynamicCache) -> int:
    """Total bytes held by a cache's key and value tensors."""
    total = 0
    for layer in cache.layers:
        if getattr(layer, "is_initialized", False):
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def cache_seq_lengths(cache: DynamicCache) -> list[int]:
    """Cached positions per layer."""
    return [
        layer.keys.shape[-2] if getattr(layer, "is_initialized", False) else 0
        for layer in cache.layers
    ]


@contextmanager
def _vram_scope(device: str) -> Iterator[None]:
    """Reset and later read CUDA peak-memory counters."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    yield
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class KVCompressionEngine:
    """Runs prompts under a chosen KV-cache compression policy."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        register_kvcomp_attention()

        # Memory held by everything outside this process, measured before the
        # model exists. A desktop GPU is not exclusively ours -- browsers and
        # background apps commonly hold a gigabyte -- and that slice must come
        # off the budget or every feasibility decision is optimistic.
        self.external_bytes = 0
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            self.total_bytes = total
            self.external_bytes = total - free
        else:
            self.total_bytes = 1 << 62

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=config.dtype,
            device_map={"": config.device},
            quantization_config=_quantization_config(config.quantization),
            attn_implementation=KVCOMP_ATTENTION,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()

        # Weight footprint, measured with the allocator pool released so it
        # reflects tensors rather than reservations.
        self.weight_bytes = 0
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            self.weight_bytes = torch.cuda.memory_allocated()

        model_config = self.model.config
        self.num_layers: int = model_config.num_hidden_layers
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.head_dim: int = getattr(
            model_config,
            "head_dim",
            model_config.hidden_size // model_config.num_attention_heads,
        )

    def bytes_per_token(self, dtype_size: int = 2) -> int:
        """KV bytes one token occupies across all layers.

        Useful for sizing a run before committing to it: multiply by the
        intended context length and compare against free VRAM.
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_size

    def build_prompt(self, user_content: str) -> str:
        """Apply the model's chat template to a single user turn."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compress(
        self, cache: DynamicCache, policy: Policy, capture, batch: int
    ) -> tuple[int, int]:
        """Apply the policy to every layer, mutating ``cache`` in place.

        Returns:
            ``(positions_before, mean_positions_after)``.
        """
        before = cache_seq_lengths(cache)
        retained: list[int] = []

        for layer_index, layer in enumerate(cache.layers):
            if not getattr(layer, "is_initialized", False):
                continue

            seq_len = layer.keys.shape[-2]
            budget = policy.layer_budget(layer_index, self.num_layers)

            if budget < 0 or seq_len <= budget:
                retained.append(seq_len)
                continue

            scores = capture.scores.get(layer_index) if capture is not None else None
            if scores is not None and scores.shape[-1] > seq_len:
                # A query probe appends its own keys while scoring; those slots
                # are cropped before compression, so trim the scores to match
                # the positions that actually remain.
                scores = scores[..., :seq_len]

            indices = policy.select(
                scores=scores,
                seq_len=seq_len,
                budget=budget,
                num_kv_heads=layer.keys.shape[1],
                batch=batch,
                device=layer.keys.device,
            )

            # `indices` is [B, kv_heads, budget]; gather needs it broadcast over
            # head_dim so each head keeps its own selection.
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
            layer.keys = layer.keys.gather(2, gather_index).contiguous()
            layer.values = layer.values.gather(2, gather_index).contiguous()
            retained.append(layer.keys.shape[-2])

        total_before = max(before) if before else 0
        total_after = int(sum(retained) / len(retained)) if retained else 0
        return total_before, total_after

    def activation_bytes(self, chunk_tokens: int) -> int:
        """Rough peak activation cost of a forward pass over ``chunk_tokens``.

        Prefill activations scale with the number of tokens processed at once,
        and on a 6 GB card they are not a rounding error: a single-pass 14.5k
        forward needed roughly 1.4 GiB beyond weights and cache. Ignoring this
        term makes the feasibility check optimistic, which is the worst kind of
        wrong here -- CUDA on Windows spills to host memory instead of failing,
        so an over-budget run crawls for tens of minutes rather than erroring.

        The MLP intermediate is the largest live tensor; the constant covers the
        several such buffers alive at once plus attention workspace.
        """
        config = self.model.config
        widest = max(config.intermediate_size, config.hidden_size)
        return 6 * chunk_tokens * widest * 2

    def memory_ceiling_bytes(self) -> int:
        """Largest total footprint this process may reach.

        Expressed as an absolute ceiling rather than "bytes still free", because
        free-memory readings swing with allocator pool state and proved
        impossible to threshold reliably. Total card capacity, memory held by
        other processes, and weight size are all stable, so a ceiling built from
        them behaves consistently between runs.
        """
        return max(
            0, self.total_bytes - self.external_bytes - self.config.vram_headroom_bytes
        )

    def predicted_peak_bytes(self, cache_tokens: int, chunk_tokens: int) -> int:
        """Predicted peak footprint for a run holding ``cache_tokens``.

        Validated against measurement on Qwen3-4B at NF4: an 8,168-token prefill
        was predicted at 2.49 + 1.12 + 0.25 = 3.86 GiB and measured at 3.86 GiB.
        The 16,384 case predicted 4.99 GiB and measured 5.20 GiB, and correctly
        sits above the ceiling on a 6 GB card sharing ~1 GiB with a desktop.
        """
        return (
            self.weight_bytes
            + self.bytes_per_token() * cache_tokens
            + self.activation_bytes(chunk_tokens)
        )

    def available_cache_bytes(self, chunk_tokens: int | None = None) -> int:
        """Bytes left for KV storage once weights and activations are counted."""
        if not torch.cuda.is_available():
            return 1 << 62
        overhead = self.weight_bytes + self.activation_bytes(chunk_tokens or 0)
        return max(0, self.memory_ceiling_bytes() - overhead)

    def peak_cache_tokens(self, prompt_tokens: int, policy: Policy) -> int:
        """Largest number of cached positions a run will hold at once.

        Without chunking the whole prompt is resident before anything can be
        evicted, so compression cannot lower the peak. With chunking the cache
        is trimmed every ``prefill_chunk`` tokens and the peak is bounded by
        ``budget + chunk`` regardless of prompt length.
        """
        budget = policy.layer_budget(0, self.num_layers)
        if budget < 0:
            return prompt_tokens

        chunk = self.config.prefill_chunk
        if chunk is None:
            return prompt_tokens
        return min(prompt_tokens, budget + chunk)

    def _probe_with_query(
        self,
        cache: DynamicCache,
        query_ids: torch.Tensor,
        text_position: int,
        batch: int,
        capture,
    ) -> None:
        """Score the cache against the prompt's tail, then discard the probe.

        Query-aware policies (SnapKV, PyramidKV, TOVA) rank cached positions by
        how much an observation window attends to them, and that window is
        meant to stand in for the question the model will be asked.

        Chunked prefill breaks that assumption. When a chunk is compressed, the
        only queries seen so far are the document text in that chunk, so early
        eviction happens with no knowledge of what is being looked for. Measured
        on RULER `niah_single_1` with a 256-entry budget, accuracy tracked chunk
        count exactly: 0.94 at one chunk, 0.69 at two, 0.00 at four, with the
        model reporting the needle "is not provided in the text".

        Running the prompt's tail through the model before each eviction
        restores the intended behaviour: scores reflect the actual question. The
        probe's own KV entries are cropped afterwards, since the tail is
        encoded for real when its chunk arrives.
        """
        # Record each layer's length individually. Under a non-uniform budget
        # the layers genuinely differ, and `Cache.crop` trims every layer to one
        # global length -- so any layer shorter than that keeps the probe's
        # entries. Those land at the end of the cache where the recent-window
        # rule protects them, so each chunk injects another copy of the question
        # into the deeper layers and evicts real content to make room. It
        # produced fluent nonsense at 16k while leaving uniform-budget policies
        # untouched, since their layers are all the same length.
        before = [
            layer.keys.shape[-2] if getattr(layer, "is_initialized", False) else 0
            for layer in cache.layers
        ]
        anchor = max(before) if before else 0
        length = query_ids.shape[1]

        rope = torch.arange(text_position, text_position + length, device=self.config.device)
        slots = torch.arange(anchor, anchor + length, device=self.config.device)

        attach_capture(self.model, capture)
        self.model(
            input_ids=query_ids,
            position_ids=rope.unsqueeze(0).expand(batch, -1),
            cache_position=slots,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        attach_capture(self.model, None)

        for layer, original in zip(cache.layers, before):
            if not getattr(layer, "is_initialized", False):
                continue
            if layer.keys.shape[-2] > original:
                layer.keys = layer.keys[:, :, :original, :].contiguous()
                layer.values = layer.values[:, :, :original, :].contiguous()

    def _prefill(
        self,
        cache: DynamicCache,
        policy: Policy,
        input_ids: torch.Tensor,
        batch: int,
        record: GenerationRecord,
    ):
        """Read the prompt, compressing the cache as it grows.

        Processing the prompt in chunks and evicting after each one is what lets
        a long context fit at all: peak cache is ``budget + prefill_chunk``
        rather than the full prompt. Compressing only once at the end would cap
        steady-state memory while leaving the peak untouched, so the longest
        contexts would remain out of reach for every policy alike.

        Two position spaces are kept distinct:

        * ``position_ids`` -- true text positions, driving RoPE. Cached keys
          keep the rotary phases they were built with.
        * ``cache_position`` -- indices within the cache, driving the causal
          mask. After eviction these no longer match text positions, and using
          text positions here would let a chunk's tokens see each other
          non-causally.

        Returns:
            Model output for the final chunk.
        """
        total = input_ids.shape[1]
        chunk_size = self.config.prefill_chunk or total
        capture = policy.capture_request()

        # The prompt's tail carries the question, so it is the observation
        # window these policies were designed around. Only needed when the
        # prompt spans several chunks; a single chunk already ends with it.
        query_ids = None
        if (
            capture is not None
            and capture.mode == "window"
            and total > chunk_size
            and self.config.query_aware_chunks
        ):
            probe = min(max(capture.window, self.config.query_probe_tokens), total)
            query_ids = input_ids[:, -probe:]

        outputs = None
        prefill_seconds = 0.0
        compress_seconds = 0.0
        observed_before = 0
        retained = 0

        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            piece = input_ids[:, start:stop]

            cached = cache.get_seq_length()
            rope_positions = torch.arange(start, stop, device=self.config.device)
            cache_positions = torch.arange(
                cached, cached + piece.shape[1], device=self.config.device
            )

            attach_capture(self.model, capture)
            tick = time.perf_counter()
            outputs = self.model(
                input_ids=piece,
                position_ids=rope_positions.unsqueeze(0).expand(batch, -1),
                cache_position=cache_positions,
                past_key_values=cache,
                use_cache=True,
                # Only the final position's logits drive generation. Otherwise
                # lm_head projects every prompt position into a 152k-entry
                # vocabulary: ~8 GB at 8k tokens, OOMing long before the KV
                # cache becomes the binding constraint.
                logits_to_keep=1,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_seconds += time.perf_counter() - tick

            # Capture is prefill-only; leaving it attached would make every
            # decode step pay for scoring it cannot use.
            attach_capture(self.model, None)

            # Re-score against the question before evicting. Skipped on the
            # final chunk, which already ends with the tail.
            if query_ids is not None and stop < total:
                if capture is not None:
                    capture.reset()
                self._probe_with_query(cache, query_ids, total, batch, capture)

            tick = time.perf_counter()
            before, after = self._compress(cache, policy, capture, batch)
            if capture is not None:
                capture.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compress_seconds += time.perf_counter() - tick

            observed_before = max(observed_before, before)
            retained = after

        record.prefill_seconds = prefill_seconds
        record.compress_seconds = compress_seconds
        # Reported against the prompt, so the ratio describes how much of the
        # context was discarded rather than how much of one chunk was.
        record.cache_tokens_before = max(observed_before, total)
        record.cache_tokens_after = retained
        record.cache_bytes = cache_num_bytes(cache)
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        policy_name: str = "full",
        policy_config: PolicyConfig | None = None,
        max_new_tokens: int | None = None,
        stop_on_eos: bool = True,
        force: bool = False,
    ) -> GenerationRecord:
        """Prefill, compress, then decode.

        Args:
            prompt: Fully-formed prompt text (already chat-templated).
            policy_name: Registered policy name.
            policy_config: Policy knobs; defaults to :class:`PolicyConfig`.
            max_new_tokens: Overrides the engine default.
            stop_on_eos: Halt at the tokenizer's EOS token.
            force: Run even when the precheck predicts the cache will not fit.
                Used to measure what an over-budget configuration actually costs
                rather than only predicting that it fails. Expect this to take
                minutes: the run does not error, it degrades into host-memory
                spilling.

        Returns:
            A :class:`GenerationRecord`. Failures, including OOM, are captured on
            the record rather than raised, so a sweep is not lost to one bad cell.
        """
        policy = build_policy(policy_name, policy_config or PolicyConfig())
        budget = max_new_tokens or self.config.max_new_tokens
        record = GenerationRecord()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_ids = inputs["input_ids"]
        batch, prompt_len = input_ids.shape
        record.prompt_tokens = prompt_len

        if self.config.precheck and not force:
            chunk = min(self.config.prefill_chunk or prompt_len, prompt_len)
            needed = self.predicted_peak_bytes(
                self.peak_cache_tokens(prompt_len, policy), chunk
            )
            available = self.memory_ceiling_bytes()
            if needed > available:
                # Reported as OOM because that is the honest outcome: this
                # configuration cannot hold its cache. Predicting it costs
                # milliseconds, whereas letting CUDA spill to host memory can
                # occupy the GPU for tens of minutes at a crawl.
                record.oom = True
                record.error = (
                    f"predicted peak {needed / 2**30:.2f} GiB exceeds ceiling "
                    f"{available / 2**30:.2f} GiB "
                    f"(external {self.external_bytes / 2**30:.2f} GiB)"
                )
                record.extra["precheck"] = True
                record.cache_tokens_before = prompt_len
                return record

        try:
            with _vram_scope(self.config.device):
                cache = DynamicCache(config=self.model.config)
                outputs = self._prefill(cache, policy, input_ids, batch, record)

                eos_ids = {self.tokenizer.eos_token_id}
                eos_ids.discard(None)

                generated: list[int] = []
                next_token = self._pick(outputs.logits[:, -1, :])
                # True text position, which keeps advancing regardless of how
                # many entries the cache actually holds.
                position = prompt_len

                start = time.perf_counter()
                for _ in range(budget):
                    token_id = int(next_token.item())
                    if stop_on_eos and token_id in eos_ids:
                        break
                    generated.append(token_id)

                    # RoPE uses the true text position; masking uses the cache's
                    # own index space. Conflating the two is what corrupts
                    # rotary phases once entries have been evicted.
                    rope_step = torch.tensor([position], device=self.config.device)
                    cache_step = torch.tensor(
                        [cache.get_seq_length()], device=self.config.device
                    )
                    outputs = self.model(
                        input_ids=next_token.view(batch, 1),
                        position_ids=rope_step.view(1, 1).expand(batch, -1),
                        cache_position=cache_step,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    next_token = self._pick(outputs.logits[:, -1, :])
                    position += 1

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                record.decode_seconds = time.perf_counter() - start
                record.generated_tokens = len(generated)
                record.text = self.tokenizer.decode(generated, skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError as exc:
            record.oom = True
            record.error = str(exc)[:500]
            self.reset()
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill a sweep
            record.error = f"{type(exc).__name__}: {exc}"[:500]
            self.reset()

        if torch.cuda.is_available():
            record.peak_vram_bytes = torch.cuda.max_memory_allocated()
            # Hand the cache's memory back to the driver rather than holding it
            # in PyTorch's pool. On a GPU shared with a desktop, a retained pool
            # keeps total usage near the card's limit, and Windows responds by
            # paging GPU memory to host RAM instead of erroring -- throughput
            # drops by roughly 50x with nothing in the logs to explain it.
            torch.cuda.empty_cache()
        return record

    def _pick(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the next token from final-position logits."""
        if self.config.greedy:
            return logits.argmax(dim=-1)
        probabilities = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def reset(self) -> None:
        """Release cached allocations after a failure."""
        attach_capture(self.model, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
