"""Shared pytest fixtures for Qwen2.5-0.5B tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure modules under examples/qwen2.5-0.5B are importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN_DIR = _REPO_ROOT / "examples" / "qwen2.5-0.5B"
if str(_QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(_QWEN_DIR))


@pytest.fixture
def device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype() -> torch.dtype:
    """Prefer bfloat16 on CUDA, float32 on CPU for test stability."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@pytest.fixture
def tiny_qwen_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a tiny fake Qwen config + model for fast offline tests.

    Monkey-patches transformers helpers so no network is needed.
    Returns (config, tokenizer) as simple stand-in objects.
    """
    from types import SimpleNamespace

    config = SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype=torch.float32,
        use_cache=False,
    )

    tokenizer = SimpleNamespace(
        vocab_size=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    return config, tokenizer


@pytest.fixture
def fineweb_shard(tmp_path: Path) -> Path:
    """Create a tiny synthetic FineWeb shard (uint16 memmap, no header).

    Matches what load_fineweb_batch expects: raw uint16 tokens.
    """
    import numpy as np

    shard_path = tmp_path / "fineweb_00000.bin"
    rng = np.random.default_rng(1337)
    tokens = rng.integers(0, 256, size=50_000, dtype=np.uint16)
    tokens.tofile(shard_path)
    return shard_path
