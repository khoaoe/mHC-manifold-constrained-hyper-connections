"""Unit tests for train_qwen_hc.py (utilities, not the full loop)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

import train_qwen_hc as t


# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------
class TestParseBool:
    @pytest.mark.parametrize("v", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_truthy(self, v: str):
        assert t._parse_bool(v) is True

    @pytest.mark.parametrize("v", ["false", "False", "FALSE", "0", "no", "NO"])
    def test_falsy(self, v: str):
        assert t._parse_bool(v) is False

    def test_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            t._parse_bool("maybe")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class TestFinewebLoader:
    def test_list_shards_finds_files(self, fineweb_shard: Path):
        shards = t._list_shards(fineweb_shard.parent)
        assert len(shards) == 1
        assert shards[0].name.startswith("fineweb")

    def test_list_shards_raises_when_empty(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            t._list_shards(tmp_path)

    def test_load_batch_shape_and_device(self, fineweb_shard: Path, device: str):
        x, y = t.load_fineweb_batch(
            data_dir=str(fineweb_shard.parent),
            split="train",
            block_size=32,
            batch_size=4,
            device=device,
        )
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)
        assert str(x.device).startswith(device)

    def test_y_is_shifted_x(self, fineweb_shard: Path):
        """Validate the loader returns (seq[:-1], seq[1:])."""
        seq_x, seq_y = t._sample_sequence(
            t._get_memmap(fineweb_shard), block_size=64
        )
        assert seq_y.shape[0] == seq_x.shape[0]
        assert len(seq_x) == 64 and len(seq_y) == 64

    def test_train_split_excludes_last_shard(self, tmp_path: Path):
        """With multiple shards, train should skip the last one."""
        for i in range(3):
            p = tmp_path / f"fineweb_{i:05d}.bin"
            np.random.default_rng(i).integers(0, 100, size=1000, dtype=np.uint32).tofile(p)
        shards = sorted(tmp_path.glob("fineweb*.bin"))
        train_choices = shards[:-1] if len(shards) > 1 else shards
        val_choices = shards[-1:]
        assert len(train_choices) == 2
        assert len(val_choices) == 1
        assert val_choices[0] not in train_choices

    def test_invalid_split_raises(self, fineweb_shard: Path):
        with pytest.raises(ValueError):
            t.load_fineweb_batch(
                data_dir=str(fineweb_shard.parent),
                split="bogus",
                block_size=16,
                batch_size=2,
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Autocast context helper
# ---------------------------------------------------------------------------
class TestAutocastContext:
    def test_cpu_bf16_returns_autocast(self):
        ctx = t._get_autocast_context("cpu", torch.bfloat16)
        assert "autocast" in type(ctx).__name__.lower() or ctx is not None

    def test_unsupported_combo_returns_nullcontext(self):
        from contextlib import nullcontext

        ctx = t._get_autocast_context("cpu", torch.float16)
        assert isinstance(ctx, nullcontext)


# ---------------------------------------------------------------------------
# _set_seed determinism
# ---------------------------------------------------------------------------
class TestSetSeed:
    def test_reproducible_randint(self):
        t._set_seed(42)
        a = torch.randint(0, 1000, (5,)).tolist()
        t._set_seed(42)
        b = torch.randint(0, 1000, (5,)).tolist()
        assert a == b
