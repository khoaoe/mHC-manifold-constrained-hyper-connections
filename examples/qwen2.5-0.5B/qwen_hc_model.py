"""Qwen2.5-0.5B HC/mHC model wrapper and utilities."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


class RMSNorm(nn.Module):
    """Root mean square normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def sinkhorn_knopp(M: torch.Tensor, tmax: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Project matrices onto the Birkhoff polytope via Sinkhorn-Knopp."""
    M = torch.exp(M)
    for _ in range(tmax):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class HyperConnectionBlock(nn.Module):
    """Hyper-Connection block with optional manifold constraint."""

    def __init__(
        self,
        hidden_size: int,
        n: int = 4,
        method: str = "mhc",
        sinkhorn_tmax: int = 20,
    ) -> None:
        super().__init__()
        method = method.lower()
        if method not in {"baseline", "hc", "mhc"}:
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.n = n
        self.sinkhorn_tmax = sinkhorn_tmax

        if self.method == "baseline":
            return

        dim = hidden_size * n
        self.rms = RMSNorm(dim)
        self.phi_pre = nn.Linear(dim, n, bias=False)
        self.phi_post = nn.Linear(dim, n, bias=False)
        self.phi_res = nn.Linear(dim, n * n, bias=False)

        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.b_pre = nn.Parameter(torch.full((1, n), 1.0 / n))
        self.b_post = nn.Parameter(torch.ones(1, n))
        b_res_init = 0.99 * torch.eye(n) + (0.01 / n) * torch.ones(n, n)
        self.b_res = nn.Parameter(b_res_init)

    def forward(
        self, x: torch.Tensor, ffn_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.method == "baseline":
            return x + ffn_fn(x), None

        batch_size, seq_len, hidden_size = x.shape
        n = self.n

        x_exp = x.unsqueeze(2).expand(batch_size, seq_len, n, hidden_size)
        x_flat = x_exp.reshape(batch_size, seq_len, n * hidden_size)
        x_norm = self.rms(x_flat)

        h_pre_tilde = self.alpha_pre * torch.tanh(self.phi_pre(x_norm)) + self.b_pre
        h_post_tilde = self.alpha_post * torch.tanh(self.phi_post(x_norm)) + self.b_post
        h_res_tilde = (
            self.alpha_res
            * torch.tanh(self.phi_res(x_norm).view(batch_size, seq_len, n, n))
            + self.b_res
        )

        h_pre = torch.sigmoid(h_pre_tilde)
        h_post = 2.0 * torch.sigmoid(h_post_tilde)
        if self.method == "mhc":
            h_res = sinkhorn_knopp(h_res_tilde, tmax=self.sinkhorn_tmax)
        else:
            h_res = h_res_tilde

        x_ffn = torch.einsum("bsn,bsnc->bsc", h_pre, x_exp)
        ffn_out = ffn_fn(x_ffn)
        ffn_exp = ffn_out.unsqueeze(2) * h_post.unsqueeze(-1)
        res_mixed = torch.einsum("bsij,bsjc->bsic", h_res, x_exp)
        out = (res_mixed + ffn_exp).mean(dim=2)
        return out, h_res


@torch.no_grad()
def compute_amax_gain(h_res_list: List[torch.Tensor]) -> Tuple[float, float]:
    """Compute forward/backward Amax gain for the composite residual mapping."""
    if not h_res_list:
        return 1.0, 1.0
    comp = h_res_list[0][0, 0].clone()
    for h_res in h_res_list[1:]:
        comp = h_res[0, 0] @ comp
    fwd = comp.abs().sum(dim=1).max().item()
    bwd = comp.abs().sum(dim=0).max().item()
    return fwd, bwd


def build_qwen_hc(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    method: str = "mhc",
    n_streams: int = 4,
    sinkhorn_tmax: int = 20,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """Build a Qwen2.5 model with HC/mHC blocks and random weights.

    Args:
        model_name: Hugging Face model identifier.
        method: One of {"baseline", "hc", "mhc"}.
        n_streams: Number of hyper-connection streams.
        sinkhorn_tmax: Sinkhorn-Knopp iterations for mHC.
        dtype: Torch dtype for model parameters.
        device: Target device.

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

    print("📥 Building model with random weights...")
    model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    base_model = _get_base_model(model)
    layers = _get_layers(base_model)
    hidden_size = _infer_hidden_size(config)

    model.hc_blocks = nn.ModuleList(
        [
            HyperConnectionBlock(
                hidden_size=hidden_size,
                n=n_streams,
                method=method,
                sinkhorn_tmax=sinkhorn_tmax,
            )
            for _ in range(len(layers))
        ]
    )
    model._hc_method = method
    model.to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in model.parameters())
    hc_params = sum(p.numel() for p in model.hc_blocks.parameters())
    print(
        "✅ Qwen2.5 ready: "
        f"layers={len(layers)} "
        f"total_params={total_params / 1e6:.2f}M "
        f"hc_params={hc_params / 1e6:.2f}M"
    )
    return model, tokenizer


def forward_with_hc(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run a forward pass with HC/mHC replacing the MLP residual path."""
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

    attention_mask = _build_causal_mask(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    h_res_buffer: List[torch.Tensor] = []
    for i, layer in enumerate(layers):
        attn_norm = getattr(layer, "input_layernorm", None) or getattr(
            layer, "ln_1", None
        )
        ffn_norm = getattr(layer, "post_attention_layernorm", None) or getattr(
            layer, "ln_2", None
        )
        if attn_norm is None or ffn_norm is None:
            raise AttributeError("Could not locate layer norms for attention/MLP.")

        attn_in = attn_norm(hidden_states)
        attn_out = _attn_forward(layer, attn_in, attention_mask, position_ids)
        hidden_states = hidden_states + attn_out

        ffn_in = ffn_norm(hidden_states)
        out, h_res = model.hc_blocks[i](ffn_in, layer.mlp)
        hidden_states = out
        if h_res is not None:
            h_res_buffer.append(h_res.detach())

    norm_layer = getattr(base_model, "norm", None) or getattr(base_model, "ln_f", None)
    if norm_layer is not None:
        hidden_states = norm_layer(hidden_states)

    logits = model.lm_head(hidden_states)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss, h_res_buffer


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
    layer: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids=position_ids)
    except TypeError:
        try:
            return rotary(hidden_states, seq_len=hidden_states.shape[1])
        except Exception:
            return None


def _attn_forward(
    layer: nn.Module,
    attn_in: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    position_embeddings = _maybe_get_position_embeddings(layer, attn_in, position_ids)
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    try:
        return layer.self_attn(attn_in, **kwargs)[0]
    except TypeError:
        if "position_embeddings" in kwargs or "attention_mask" in kwargs:
            kwargs.pop("position_embeddings", None)
            kwargs.pop("attention_mask", None)
            return layer.self_attn(attn_in, **kwargs)[0]
        raise
