"""Probe which SDPA backends this machine can actually use.

Run this first on any new machine. PyTorch's SDPA silently falls back to the
MATH backend when no fused kernel accepts the input, and the math backend
materialises the full [heads, q, kv] attention matrix. At long context that is
the difference between 64 MiB and 8 GiB, and it surfaces only as an OOM with no
indication that a fallback occurred.

The case that matters here is ``enable_gqa=True``: it is the natural way to run
grouped-query attention without expanding KV heads, but on some builds no fused
kernel supports it. ``kvcomp`` therefore expands KV heads explicitly.

Usage:
    python scripts/check_sdpa_backends.py [seq_len]
"""

from __future__ import annotations

import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Qwen3-4B attention geometry.
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128

BACKENDS = {
    "flash": [SDPBackend.FLASH_ATTENTION],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "math": [SDPBackend.MATH],
}


def probe(seq_len: int, use_gqa: bool, backend: str | None) -> str:
    """Run one attention call and report the extra memory it needed."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    query = torch.randn(1, QUERY_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, KV_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    if not use_gqa:
        key = key.repeat_interleave(QUERY_HEADS // KV_HEADS, dim=1).contiguous()
    value = torch.randn_like(key)

    baseline = torch.cuda.memory_allocated()
    try:
        if backend is None:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=use_gqa
            )
        else:
            with sdpa_kernel(BACKENDS[backend]):
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True, enable_gqa=use_gqa
                )
        extra = (torch.cuda.max_memory_allocated() - baseline) / 2**20
        del output
        return f"ok    {extra:9.1f} MiB"
    except torch.OutOfMemoryError:
        return "OOM"
    except RuntimeError as exc:
        return f"unavailable ({str(exc)[:40]})"
    finally:
        del query, key, value


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device available")
        return 1

    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 8192

    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}")
    print(f"seq_len     : {seq_len}")
    print()
    print(f"{'configuration':<34}{'result'}")
    print("-" * 62)

    for label, backend in [("auto", None), *((name, name) for name in BACKENDS)]:
        for use_gqa in (True, False):
            mode = "enable_gqa=True" if use_gqa else "repeat_kv"
            print(f"{label:<12}{mode:<22}{probe(seq_len, use_gqa, backend)}")

    print()
    print(
        "If 'auto + enable_gqa=True' costs far more than 'auto + repeat_kv',\n"
        "SDPA is falling back to the math backend and kvcomp's explicit\n"
        "KV-head expansion is what keeps long contexts inside the VRAM budget."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
