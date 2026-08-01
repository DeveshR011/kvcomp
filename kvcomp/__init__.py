"""KV-cache compression benchmarking for memory-constrained GPUs.

This package implements training-free KV-cache compression methods
(StreamingLLM, SnapKV, H2O, PyramidKV, TOVA) on top of HuggingFace
Transformers, and evaluates them on RULER and Needle-in-a-Haystack
under a hard VRAM budget.
"""

__version__ = "0.1.0"
