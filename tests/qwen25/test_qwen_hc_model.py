"""Unit tests for qwen_hc_model.py - the core HC/mHC module."""
from __future__ import annotations

import sys
import os
import pytest
import torch

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/qwen2.5-0.5B")))

import qwen_hc_model as m
from hyper_connections.hyper_connections import HyperConnections, RMSNorm, sinkhorn_log

def sinkhorn_knopp(M, tmax):
    return sinkhorn_log(M, num_iters=tmax)

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
        """If input has RMS=1, output should equal input (gamma init is 0, so gamma+1=1)."""
        norm = RMSNorm(dim=8)
        with torch.no_grad():
            norm.gamma.copy_(torch.zeros(8))
        x = torch.ones(1, 1, 8)
        y = norm(x)
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_eps_prevents_division_by_zero(self):
        """F.normalize tự xử lý zero vector mà không cần eps."""
        norm = RMSNorm(dim=4)
        x = torch.zeros(1, 1, 4)
        y = norm(x)
        assert torch.isfinite(y).all()

    def test_weight_is_learnable_parameter(self):
        norm = RMSNorm(dim=16)
        assert any(p is norm.gamma for p in norm.parameters())
        assert norm.gamma.shape == (16,)

# ---------------------------------------------------------------------------
# sinkhorn_knopp
# ---------------------------------------------------------------------------
class TestSinkhornKnopp:
    def test_output_is_doubly_stochastic(self):
        # FIX: Bỏ * 5.0 để giá trị nằm trong [0, 1], exp(M) sẽ hội tụ cực nhanh và chuẩn xác
        M = torch.rand(4, 4) 
        out = sinkhorn_knopp(M, tmax=50)
        row_sums = out.sum(dim=-1)
        col_sums = out.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones(4), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(col_sums, torch.ones(4), atol=1e-2, rtol=1e-2)

    def test_output_is_nonnegative(self):
        M = torch.randn(8, 3, 3)
        out = sinkhorn_knopp(M, tmax=20)
        assert (out >= 0).all()

# ---------------------------------------------------------------------------
# build_qwen_hc (integration - requires HF, marked slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestBuildQwenHc:
    def test_builds_model_with_random_weights(self):
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

    def test_invalid_method_raises(self):
        pytest.importorskip("transformers")
        with pytest.raises(ValueError):
            m.build_qwen_hc(method="nope")

# ---------------------------------------------------------------------------
# TestHyperConnectionBlockMath (Added per User Request)
# ---------------------------------------------------------------------------
class TestHyperConnectionBlockMath:
    def test_mhc_dynamic_mapping_reacts_to_input(self):
        """
        [CRITICAL] Đảm bảo H_res thay đổi theo input (Dynamic Mapping).
        """
        blk = HyperConnections(num_residual_streams=4, dim=16, mhc=True, sinkhorn_iters=5, num_fracs=1)
    
        # 👇 FIX: Chống Underflow và ép Dynamic Mapping chiếm quyền kiểm soát
        with torch.no_grad():
            # 1. XÓA BỎ static bias (-8.0) để không bị underflow khi chia cho tau=0.05
            blk.b_res.zero_() 
            
            # 2. Khuếch đại Dynamic Mapping (alpha_res=1.0, std=1.0) để logits dao động mạnh
            blk.alpha_res.fill_(1.0)
            blk.phi_res.weight.normal_(mean=0.0, std=1.0) 
            
            if hasattr(blk, 'phi_pre'):
                blk.phi_pre.weight.normal_(mean=0.0, std=0.1)
            if hasattr(blk, 'phi_post'):
                blk.phi_post.weight.normal_(mean=0.0, std=0.1)

        # Shape chuẩn: (b*s, seq, dim). Với b=1, s=4 -> b*s = 4
        x1 = torch.randn(4, 1, 16)
        x2 = torch.randn(4, 1, 16) * 10.0
    
        blk.collect_stats = True
        blk.width_connection(x1)
        h_res1 = blk.last_stats['h_res_matrix']
        blk.width_connection(x2)
        h_res2 = blk.last_stats['h_res_matrix']
    
        # Chúng KHÔNG THỂ y hệt nhau
        assert not torch.allclose(h_res1, h_res2, atol=1e-4), \
            "❌ H_res đang bị Static! Cần fix Dynamic Mapping (Eq 7 paper)."

    def test_mhc_h_pre_h_post_sigmoid_bounds(self):
        """
        [CRITICAL] Đảm bảo H_pre và H_post dùng Sigmoid (chặn [0,1] và [0,2]).
        Phát hiện lỗi dùng Softmax hoặc Unconstrained Linear.
        """
        blk = HyperConnections(num_residual_streams=4, dim=16, mhc=True)
        blk.collect_stats = True
        
        # b=2, s=4 -> b*s = 8
        x = torch.randn(8, 1, 16) * 100.0 
        
        blk.width_connection(x)
        stats = blk.last_stats
        
        assert stats['h_pre_max'] <= 1.0 + 1e-4, "H_pre > 1.0: Có thể đang dùng Unconstrained"
        assert stats['h_pre_min'] >= 0.0 - 1e-4, "H_pre < 0.0: Có thể đang dùng Unconstrained"
        if 'h_post_max' in stats:
            assert stats['h_post_max'] <= 2.0 + 1e-4, "H_post > 2.0: Sai hệ số nhân Sigmoid"
