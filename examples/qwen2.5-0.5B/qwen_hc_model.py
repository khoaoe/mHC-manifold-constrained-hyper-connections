"""Qwen2.5-0.5B HC/mHC model wrapper and utilities."""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from hyper_connections.hyper_connections import HyperConnections


def build_qwen_hc(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    method: str = "mhc",
    n_streams: int = 4,
    sinkhorn_tmax: int = 20,
    pretrained: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    num_fracs: int = 1,
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """Build a Qwen2.5 model with HC/mHC blocks.

    Args:
        model_name: Hugging Face model identifier.
        method: One of {"baseline", "hc", "mhc"}.
        n_streams: Number of hyper-connection streams.
        sinkhorn_tmax: Sinkhorn-Knopp iterations for mHC.
        pretrained: Whether to load pretrained weights.
        dtype: Torch dtype for model parameters.
        device: Target device.
        num_fracs: Number of fractions for Frac-Connections.

    Returns:
        (model, tokenizer)
    """
    method = method.lower()
    if method not in {"baseline", "hc", "mhc"}:
        raise ValueError(f"Unsupported method: {method}")

    print("📥 Loading Qwen2.5 config and tokenizer...")
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "torch_dtype"):
        config.torch_dtype = dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if pretrained:
        print(f"📥 Building model with pretrained weights from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    else:
        print("📥 Building model with random weights...")
        model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    base_model = _get_base_model(model)
    layers = _get_layers(base_model)
    hidden_size = _infer_hidden_size(config)
    num_layers = len(layers)

    if method == "baseline":
        model.hc_blocks = None
        model._hc_expand = None
        model._hc_reduce = None
        hc_params = 0
    else:
        # Get expand/reduce stream functions from the library
        expand_fn, reduce_fn = HyperConnections.get_expand_reduce_stream_functions(
            n_streams
        )
        model._hc_expand = expand_fn
        model._hc_reduce = reduce_fn

        model.hc_blocks = nn.ModuleList(
            [
                HyperConnections(
                    num_residual_streams=n_streams,
                    dim=hidden_size,
                    layer_index=j,  # j runs from 0 to 2*num_layers - 1
                    mhc=(method == "mhc"),
                    sinkhorn_iters=sinkhorn_tmax,
                    num_fracs=num_fracs,
                )
                for j in range(2 * num_layers)  # 2 blocks per layer: Attn + MLP
            ]
        )
        hc_params = sum(p.numel() for p in model.hc_blocks.parameters())

    model._hc_method = method
    model.to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        "✅ Qwen2.5 ready: "
        f"layers={num_layers} "
        f"total_params={total_params / 1e6:.2f}M "
        f"hc_params={hc_params / 1e6:.2f}M"
    )
    return model, tokenizer


def forward_with_hc(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a forward pass with HC/mHC replacing the MLP residual path.

    For baseline: uses native HuggingFace forward (correct RoPE/mask handling).
    For HC/mHC: manual layer loop with expand/reduce stream functions.
    """

    # --- Baseline: use native HF forward for correctness ---
    if model.hc_blocks is None:
        outputs = model(input_ids, labels=labels)
        return outputs.loss

    # --- HC / mHC: manual forward with stream expand/reduce ---
    base_model = _get_base_model(model)
    layers = _get_layers(base_model)

    batch_size, seq_len = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

    if hasattr(base_model, "embed_tokens"):
        hidden_states = base_model.embed_tokens(input_ids)
    elif hasattr(base_model, "wte"):
        hidden_states = base_model.wte(input_ids)
    else:
        raise AttributeError("Could not locate token embedding layer.")

    # Expand to multi-stream residual
    hidden_states = model._hc_expand(hidden_states)

    attention_mask = _build_causal_mask(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    n_streams = model.hc_blocks[0].num_residual_streams

    for i, layer in enumerate(layers):
        attn_norm = getattr(layer, "input_layernorm", None) or getattr(
            layer, "ln_1", None
        )
        ffn_norm = getattr(layer, "post_attention_layernorm", None) or getattr(
            layer, "ln_2", None
        )
        if attn_norm is None or ffn_norm is None:
            raise AttributeError("Could not locate layer norms for attention/MLP.")

        # ---------------------------------------------------------
        # BRANCH 1: ATTENTION with HC/mHC (layer_index = 2 * i)
        # ---------------------------------------------------------
        # 1. Extract single-stream from multi-stream highway (hidden_states stays un-normalized)
        branch_attn_in, add_attn_residual_fn = model.hc_blocks[2 * i](hidden_states)

        # 2. Apply Pre-Norm ON THE SINGLE-STREAM branch (mathematically correct)
        attn_in = attn_norm(branch_attn_in)

        # 3. Run Attention on base batch -> VRAM-friendly
        attn_out = _attn_forward(layer, base_model, attn_in, attention_mask, position_ids)

        # 4. Write back to multi-stream highway (mix via H_res)
        hidden_states = add_attn_residual_fn(attn_out)

        # ---------------------------------------------------------
        # BRANCH 2: FFN / MLP with HC/mHC (layer_index = 2 * i + 1)
        # ---------------------------------------------------------
        # 1. Extract single-stream for MLP
        branch_mlp_in, add_mlp_residual_fn = model.hc_blocks[2 * i + 1](hidden_states)

        # 2. Apply Pre-Norm ON THE SINGLE-STREAM branch
        ffn_in = ffn_norm(branch_mlp_in)

        # 3. Run MLP on base batch
        mlp_out = layer.mlp(ffn_in)

        # 4. Write back to multi-stream highway
        hidden_states = add_mlp_residual_fn(mlp_out)

        # Collect metrics
        if getattr(model, "collect_metrics", False):
            if not hasattr(model, "_residual_norms"):
                model._residual_norms = []
            model._residual_norms.append(hidden_states.norm(p=2, dim=-1).mean().item())

    # Reduce multi-stream back to single stream
    hidden_states = model._hc_reduce(hidden_states)

    norm_layer = getattr(base_model, "norm", None) or getattr(base_model, "ln_f", None)
    if norm_layer is not None:
        hidden_states = norm_layer(hidden_states)

    logits = model.lm_head(hidden_states)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


def _infer_hidden_size(config: AutoConfig) -> int:
    for name in ("hidden_size", "n_embd", "dim", "model_dim"):
        if hasattr(config, name):
            return int(getattr(config, name))
    raise ValueError("Unable to infer hidden size from config.")


def _get_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _get_layers(base_model: nn.Module) -> nn.ModuleList:
    if hasattr(base_model, "layers"):
        return base_model.layers
    if hasattr(base_model, "h"):
        return base_model.h
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "layers"):
        return base_model.decoder.layers
    raise AttributeError("Could not locate transformer layers.")


def _build_causal_mask(
    input_shape: Tuple[int, int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    batch_size, seq_len = input_shape
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.to(dtype=dtype)
    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


def _maybe_get_position_embeddings(
    layer: nn.Module, base_model: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Lấy position embeddings (RoPE) một cách an toàn."""
    # Trong transformers 4.51+, rotary_emb nằm ở base_model, không phải trong self_attn
    rotary = getattr(base_model, "rotary_emb", None)
    
    # Fallback cho bản cũ hơn
    if rotary is None:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            rotary = getattr(attn, "rotary_emb", None)

    if rotary is None:
        return None
    
    try:
        # Cách chuẩn cho transformers >= 4.36 (Qwen2, Llama 3, v.v.)
        return rotary(hidden_states, position_ids)
    except TypeError:
        try:
            # Fallback cho các phiên bản transformers cũ hơn
            return rotary(hidden_states, seq_len=hidden_states.shape[1])
        except Exception:
            return None


def _attn_forward(
    layer: nn.Module,
    base_model: nn.Module,
    attn_in: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Thực hiện forward pass cho attention layer của Qwen2."""
    bsz = attn_in.shape[0]
    if position_ids.shape[0] != bsz:
        n_repeat = bsz // position_ids.shape[0]
        position_ids = position_ids.repeat(n_repeat, 1)
    if attention_mask.shape[0] != bsz:
        n_repeat = bsz // attention_mask.shape[0]
        attention_mask = attention_mask.repeat(n_repeat, 1, 1, 1)

    position_embeddings = _maybe_get_position_embeddings(layer, base_model, attn_in, position_ids)
    
    kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "output_attentions": getattr(base_model.config, "output_attentions", False),
    }
    
    # Nếu lấy được position embeddings, thêm vào kwargs
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
        
    try:
        # Qwen2Attention trả về tuple: (hidden_states, attentions, past_key_value)
        # Chúng ta chỉ cần lấy phần tử đầu tiên (hidden_states)
        output = layer.self_attn(attn_in, **kwargs)
        if kwargs["output_attentions"] and len(output) > 1 and output[1] is not None:
            if not hasattr(base_model, "_attn_weights"):
                base_model._attn_weights = []
            base_model._attn_weights.append(output[1].detach())
        return output[0]
    except TypeError as e:
        # Nếu vẫn lỗi, in ra chi tiết để debug thay vì xóa tham số mù quáng
        print(f"❌ Lỗi khi gọi self_attn với các kwargs: {list(kwargs.keys())}")
        print(f"❌ Chi tiết lỗi: {e}")
        raise
