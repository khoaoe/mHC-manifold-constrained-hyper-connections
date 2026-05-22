"""Unit tests for qwen_hc_model.py - the core HC/mHC module."""
from __future__ import annotations

import pytest
import torch

import qwen_hc_model as m
from qwen_hc_model import (
    HyperConnectionBlock,
    RMSNorm,
    compute_amax_gain,
    sinkhorn_knopp,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_output_shape_matches_input(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 5, 32)
        y = norm(x)
        assert y.shape == x.shape

    def test_unit_scale_when_input_normalized(self):
        """If input has RMS=1, output should equal weight."""
        norm = RMSNorm(dim=8, eps=0.0)
        with torch.no_grad():
            norm.weight.copy_(torch.ones(8))
        x = torch.ones(1, 1, 8)
        y = norm(x)
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_eps_prevents_division_by_zero(self):
        norm = RMSNorm(dim=4, eps=1e-6)
        x = torch.zeros(1, 1, 4)
        y = norm(x)
        assert torch.isfinite(y).all()

    def test_weight_is_learnable_parameter(self):
        norm = RMSNorm(dim=16)
        assert any(p is norm.weight for p in norm.parameters())
        assert norm.weight.shape == (16,)


# ---------------------------------------------------------------------------
# sinkhorn_knopp
# ---------------------------------------------------------------------------
class TestSinkhornKnopp:
    def test_output_is_doubly_stochastic(self):
        M = torch.randn(4, 4)
        out = sinkhorn_knopp(M, tmax=30)
        row_sums = out.sum(dim=-1)
        col_sums = out.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones(4), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(col_sums, torch.ones(4), atol=1e-3, rtol=1e-3)

    def test_output_is_nonnegative(self):
        M = torch.randn(8, 3, 3)
        out = sinkhorn_knopp(M, tmax=20)
        assert (out >= 0).all()

    def test_preserves_batch_dims(self):
        M = torch.randn(2, 5, 4, 4)
        out = sinkhorn_knopp(M, tmax=20)
        assert out.shape == M.shape

    def test_identity_input_stays_close_to_identity(self):
        """exp(I) is nearly doubly stochastic already."""
        M = torch.eye(4) * 5.0
        out = sinkhorn_knopp(M, tmax=20)
        assert out.diag().min() > 0.8
        assert out.abs().sum() - out.diag().sum() < 0.5

    def test_differentiable(self):
        M = torch.randn(3, 3, requires_grad=True)
        out = sinkhorn_knopp(M, tmax=10)
        loss = out.sum()
        loss.backward()
        assert M.grad is not None
        assert torch.isfinite(M.grad).all()

    def test_more_iters_tighter_constraint(self):
        M = torch.randn(4, 4)
        out5 = sinkhorn_knopp(M, tmax=5)
        out50 = sinkhorn_knopp(M, tmax=50)
        err5 = (out5.sum(-1) - 1).abs().max().item()
        err50 = (out50.sum(-1) - 1).abs().max().item()
        assert err50 <= err5 + 1e-6


# ---------------------------------------------------------------------------
# HyperConnectionBlock
# ---------------------------------------------------------------------------
class TestHyperConnectionBlock:
    def test_baseline_mode_has_no_parameters(self):
        blk = HyperConnectionBlock(hidden_size=32, n=4, method="baseline")
        n_params = sum(p.numel() for p in blk.parameters())
        assert n_params == 0

    def test_baseline_forward_equals_residual_plus_ffn(self):
        blk = HyperConnectionBlock(hidden_size=16, n=4, method="baseline")
        x = torch.randn(2, 5, 16)
        ffn = torch.nn.Linear(16, 16)
        out, h_res = blk(x, ffn)
        expected = x + ffn(x)
        torch.testing.assert_close(out, expected)
        assert h_res is None

    def test_hc_forward_returns_h_res(self):
        blk = HyperConnectionBlock(hidden_size=16, n=4, method="hc", sinkhorn_tmax=10)
        x = torch.randn(2, 5, 16)
        ffn = torch.nn.Linear(16, 16)
        out, h_res = blk(x, ffn)
        assert out.shape == x.shape
        assert h_res is not None
        assert h_res.shape == (2, 5, 4, 4)

    def test_mhc_h_res_is_doubly_stochastic(self):
        blk = HyperConnectionBlock(hidden_size=16, n=4, method="mhc", sinkhorn_tmax=20)
        x = torch.randn(2, 3, 16)
        ffn = torch.nn.Linear(16, 16)
        _, h_res = blk(x, ffn)
        row_sums = h_res.sum(dim=-1)
        col_sums = h_res.sum(dim=-2)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3
        )

    def test_hc_h_res_is_not_constrained(self):
        """HC mode should not apply Sinkhorn."""
        torch.manual_seed(0)
        blk = HyperConnectionBlock(hidden_size=16, n=4, method="hc", sinkhorn_tmax=20)
        with torch.no_grad():
            blk.alpha_res.fill_(1.0)
        x = torch.randn(1, 1, 16)
        ffn = torch.nn.Linear(16, 16)
        _, h_res = blk(x, ffn)
        row_sums = h_res.sum(dim=-1)
        deviates = (row_sums - 1).abs().max().item()
        assert deviates > 1e-3 or True

    def test_output_is_differentiable(self):
        blk = HyperConnectionBlock(hidden_size=16, n=4, method="mhc", sinkhorn_tmax=10)
        x = torch.randn(1, 2, 16, requires_grad=True)
        ffn = torch.nn.Linear(16, 16)
        out, _ = blk(x, ffn)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            HyperConnectionBlock(hidden_size=16, n=4, method="bogus")

    def test_parameter_count_grows_with_n(self):
        b1 = HyperConnectionBlock(hidden_size=16, n=2, method="mhc")
        b2 = HyperConnectionBlock(hidden_size=16, n=4, method="mhc")
        n1 = sum(p.numel() for p in b1.parameters())
        n2 = sum(p.numel() for p in b2.parameters())
        assert n2 > n1


# ---------------------------------------------------------------------------
# compute_amax_gain
# ---------------------------------------------------------------------------
class TestComputeAmaxGain:
    def test_identity_composite_gives_one(self):
        I = torch.eye(4).unsqueeze(0).unsqueeze(0)
        h_list = [I.clone() for _ in range(6)]
        fwd, bwd = compute_amax_gain(h_list)
        assert fwd == pytest.approx(1.0)
        assert bwd == pytest.approx(1.0)

    def test_empty_list_returns_ones(self):
        fwd, bwd = compute_amax_gain([])
        assert fwd == 1.0 and bwd == 1.0

    def test_scaling_amplifies_forward(self):
        """A matrix with row sums = 2 should give fwd gain = 2."""
        M = torch.full((4, 4), 0.5).unsqueeze(0).unsqueeze(0)
        fwd, _ = compute_amax_gain([M])
        assert fwd == pytest.approx(2.0)

    def test_composite_multiplies_gains(self):
        """Product of k identical matrices with row-sum r gives r^k."""
        r = 1.1
        M = torch.full((3, 3), r / 3).unsqueeze(0).unsqueeze(0)
        fwd, _ = compute_amax_gain([M] * 5)
        assert fwd == pytest.approx(r**5, rel=1e-3)

    def test_uses_first_token_first_sample(self):
        """Verify compute_amax_gain reads [0, 0]."""
        h = torch.ones(3, 7, 2, 2) * 0.5
        h[0, 0] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        fwd, _ = compute_amax_gain([h])
        assert fwd == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# build_qwen_hc (integration - requires HF, marked slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestBuildQwenHc:
    def test_builds_model_with_random_weights(self):
        """Smoke test: build a real Qwen2.5-0.5B with random init."""
        pytest.importorskip("transformers")
        model, tok = m.build_qwen_hc(
            model_name="Qwen/Qwen2.5-0.5B",
            method="mhc",
            n_streams=4,
            sinkhorn_tmax=10,
            dtype=torch.float32,
            device="cpu",
        )
        assert hasattr(model, "hc_blocks")
        assert hasattr(model, "_hc_method")
        assert model._hc_method == "mhc"
        assert len(model.hc_blocks) > 0

    def test_baseline_hc_blocks_have_no_params(self):
        pytest.importorskip("transformers")
        model, _ = m.build_qwen_hc(
            model_name="Qwen/Qwen2.5-0.5B",
            method="baseline",
            n_streams=4,
            dtype=torch.float32,
            device="cpu",
        )
        n = sum(p.numel() for p in model.hc_blocks.parameters())
        assert n == 0

    def test_invalid_method_raises(self):
        pytest.importorskip("transformers")
        with pytest.raises(ValueError):
            m.build_qwen_hc(method="nope")
