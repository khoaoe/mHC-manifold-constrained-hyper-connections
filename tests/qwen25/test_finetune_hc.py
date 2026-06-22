# File: tests/qwen25/test_finetune_hc.py
"""Unit tests for the Classification Wrapper (QwenHCForSequenceClassification).

Tests:
  1. Forward pass produces correct output shape.
  2. Loss is valid (not None, not NaN).
  3. Backward pass: gradients flow to classification head AND mHC layers.
"""
from __future__ import annotations

import pytest
import torch

# conftest.py already adds examples/qwen2.5-0.5B to sys.path
from qwen_hc_model import build_qwen_hc
from qwen_classification import QwenHCForSequenceClassification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=["mhc", "hc", "baseline"])
def clf_model_and_tokenizer(request, dtype):
    """Build a classification model for each method (mhc, hc, baseline).

    Uses float32 on CPU for numerical stability in tests.
    """
    method = request.param
    causal_model, tokenizer = build_qwen_hc(
        model_name="Qwen/Qwen2.5-0.5B",
        method=method,
        n_streams=4,
        pretrained=False,
        dtype=dtype,
        device="cpu",
    )
    clf_model = QwenHCForSequenceClassification(
        causal_model, hidden_size=896, num_labels=2,
    )
    # Ensure classification head uses same dtype
    clf_model.score.to(dtype=dtype)
    return clf_model, tokenizer, method


# ---------------------------------------------------------------------------
# Test 1: Forward pass shape
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestClassifierForwardPass:
    def test_logits_shape(self, clf_model_and_tokenizer):
        clf_model, tokenizer, method = clf_model_and_tokenizer
        clf_model.eval()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask[0, -3:] = 0  # Simulate padding on first sequence

        with torch.no_grad():
            outputs = clf_model(input_ids=input_ids, attention_mask=attention_mask)

        assert "logits" in outputs, f"[{method}] Output missing 'logits' key"
        assert outputs["logits"].shape == (batch_size, 2), (
            f"[{method}] Expected logits shape ({batch_size}, 2), "
            f"got {outputs['logits'].shape}"
        )

    def test_logits_without_attention_mask(self, clf_model_and_tokenizer):
        clf_model, tokenizer, method = clf_model_and_tokenizer
        clf_model.eval()

        input_ids = torch.randint(0, 1000, (2, 16))

        with torch.no_grad():
            outputs = clf_model(input_ids=input_ids)

        assert outputs["logits"].shape == (2, 2), (
            f"[{method}] Forward without attention_mask failed"
        )


# ---------------------------------------------------------------------------
# Test 2: Loss is valid
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestClassifierLoss:
    def test_loss_is_valid(self, clf_model_and_tokenizer):
        clf_model, tokenizer, method = clf_model_and_tokenizer
        clf_model.train()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([1, 0])

        outputs = clf_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        )
        loss = outputs["loss"]

        assert loss is not None, f"[{method}] Loss is None!"
        assert not torch.isnan(loss), f"[{method}] Loss is NaN!"
        assert not torch.isinf(loss), f"[{method}] Loss is Inf!"
        assert loss.item() > 0, f"[{method}] Loss should be positive for random weights"


# ---------------------------------------------------------------------------
# Test 3: Backward pass – gradient flow
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestClassifierBackwardPass:
    def test_gradient_reaches_classification_head(self, clf_model_and_tokenizer):
        clf_model, tokenizer, method = clf_model_and_tokenizer
        clf_model.train()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([1, 0])

        outputs = clf_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        )
        outputs["loss"].backward()

        assert clf_model.score.weight.grad is not None, (
            f"[{method}] Gradient did NOT reach the classification head (score.weight)!"
        )
        assert clf_model.score.weight.grad.abs().sum() > 0, (
            f"[{method}] Classification head gradient is all zeros!"
        )

    def test_gradient_reaches_transformer_layers(self, clf_model_and_tokenizer):
        clf_model, tokenizer, method = clf_model_and_tokenizer
        clf_model.train()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([1, 0])

        outputs = clf_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        )
        outputs["loss"].backward()

        # Check gradient flows to the deepest transformer layer
        base_model = clf_model.causal_model.model
        last_layer = base_model.layers[-1]
        q_proj_grad = last_layer.self_attn.q_proj.weight.grad

        assert q_proj_grad is not None, (
            f"[{method}] Gradient did NOT reach the last transformer layer (q_proj)!"
        )

    def test_gradient_reaches_hc_blocks(self, clf_model_and_tokenizer):
        """For hc/mhc methods, verify gradient flows into HC blocks too."""
        clf_model, tokenizer, method = clf_model_and_tokenizer

        if method == "baseline":
            pytest.skip("Baseline has no hc_blocks to check")

        clf_model.train()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([1, 0])

        outputs = clf_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        )
        outputs["loss"].backward()

        # Check at least one HC block parameter received gradients
        has_grad = False
        for param in clf_model.causal_model.hc_blocks.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, (
            f"[{method}] Gradient did NOT reach any HC block parameter!"
        )
