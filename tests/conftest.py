"""Shared pytest configuration.

Puts the repository root on ``sys.path`` so tests import ``kvcomp`` without an
editable install, and skips GPU-marked tests when no CUDA device is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip ``@pytest.mark.gpu`` tests unless CUDA is usable."""
    if "gpu" not in item.keywords:
        return
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")
        return
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
