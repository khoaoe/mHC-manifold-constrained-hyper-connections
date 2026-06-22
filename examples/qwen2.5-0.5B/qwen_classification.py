# File: examples/qwen2.5-0.5B/qwen_classification.py
"""Sequence Classification wrapper for Qwen2.5 HC/mHC models.

Strips the CausalLM head from a model built by `build_qwen_hc` and attaches
a linear classification head. Supports all three methods: baseline, hc, mhc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from qwen_hc_model import (
    _get_base_model,
    _get_layers,
    _build_causal_mask,
    _attn_forward,
)


class QwenHCForSequenceClassification(nn.Module):
    """Wrap a Qwen2.5 HC/mHC CausalLM model for sequence classification.

    For *baseline* models the standard HuggingFace forward is reused.
    For *hc/mhc* models the manual layer loop (expand → attn+mlp → reduce)
    mirrors `forward_with_hc` but outputs classification logits instead of
    language-model logits.
    """

    def __init__(self, causal_model: nn.Module, hidden_size: int = 896, num_labels: int = 2):
        super().__init__()
        # Keep the full causal_model so we can access hc_blocks, _hc_expand, etc.
        self.causal_model = causal_model
        self.num_labels = num_labels
        # Classification head (no bias, same convention as HF)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        hidden_states = self._backbone_forward(input_ids, attention_mask)

        # Pool: take the last non-padding token per sequence
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else:
            sequence_lengths = torch.full(
                (batch_size,), input_ids.shape[1] - 1,
                dtype=torch.long, device=input_ids.device,
            )

        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]

        logits = self.score(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # Backbone forward – dispatches to HF or manual HC loop
    # ------------------------------------------------------------------
    def _backbone_forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the final hidden states (batch, seq_len, hidden_size)."""
        model = self.causal_model

        # --- Baseline: use standard HuggingFace forward ---
        if getattr(model, "hc_blocks", None) is None:
            base_model = _get_base_model(model)
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return outputs.last_hidden_state

        # --- HC / mHC: manual forward with stream expand/reduce ---
        base_model = _get_base_model(model)
        layers = _get_layers(base_model)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        # Embedding
        if hasattr(base_model, "embed_tokens"):
            hidden_states = base_model.embed_tokens(input_ids)
        elif hasattr(base_model, "wte"):
            hidden_states = base_model.wte(input_ids)
        else:
            raise AttributeError("Could not locate token embedding layer.")

        # Expand to multi-stream
        hidden_states = model._hc_expand(hidden_states)

        causal_mask = _build_causal_mask(
            (batch_size, seq_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for i, layer in enumerate(layers):
            attn_norm = getattr(layer, "input_layernorm", None) or getattr(layer, "ln_1", None)
            ffn_norm = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)
            if attn_norm is None or ffn_norm is None:
                raise AttributeError("Could not locate layer norms for attention/MLP.")

            # -- Attention branch (hc_block index = 2*i) --
            branch_attn_in, add_attn_residual_fn = model.hc_blocks[2 * i](hidden_states)
            attn_in = attn_norm(branch_attn_in)
            attn_out = _attn_forward(layer, base_model, attn_in, causal_mask, position_ids)
            hidden_states = add_attn_residual_fn(attn_out)

            # -- MLP branch (hc_block index = 2*i+1) --
            branch_mlp_in, add_mlp_residual_fn = model.hc_blocks[2 * i + 1](hidden_states)
            ffn_in = ffn_norm(branch_mlp_in)
            mlp_out = layer.mlp(ffn_in)
            hidden_states = add_mlp_residual_fn(mlp_out)

        # Reduce multi-stream back to single stream
        hidden_states = model._hc_reduce(hidden_states)

        # Final layer-norm
        norm_layer = getattr(base_model, "norm", None) or getattr(base_model, "ln_f", None)
        if norm_layer is not None:
            hidden_states = norm_layer(hidden_states)

        return hidden_states
