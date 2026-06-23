"""Plot Routing Matrix Heatmaps for HC vs mHC comparison.

Extracts H_res matrices from trained checkpoints and visualizes them as heatmaps
to show the structural difference between unconstrained HC (chaotic) and
Sinkhorn-constrained mHC (balanced doubly-stochastic).

Usage:
    python plot_heatmaps.py                          # Default: both HC and mHC
    python plot_heatmaps.py --methods hc mhc         # Explicit
    python plot_heatmaps.py --layers -1 0 23         # Specific layers
    python plot_heatmaps.py --output-dir figures/    # Custom output dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from einops import rearrange

# Ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qwen_hc_model import build_qwen_hc, forward_with_hc


# ============================================================
# 1. MODEL LOADING
# ============================================================

def load_model_from_ckpt(
    ckpt_path: str,
    method: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a trained Qwen2.5-0.5B + HC/mHC model from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"📥 Building Qwen2.5-0.5B with {method.upper()} architecture...")
    model, tokenizer = build_qwen_hc(
        model_name="Qwen/Qwen2.5-0.5B",
        method=method,
        n_streams=4,
        pretrained=False,
        dtype=dtype,
        device=device,
    )

    print(f"📥 Loading trained weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


# ============================================================
# 2. HOOK-BASED H_RES CAPTURE
# ============================================================

def capture_all_h_res(
    model, tokenizer, method: str, device: str = "cpu"
) -> List[torch.Tensor]:
    """Run a dummy forward pass and capture H_res matrices from all HC blocks.

    For HC (unconstrained): uses forward pre-hooks to reconstruct H_res
        from alpha/beta parameters (outer product formulation).
    For mHC (Sinkhorn-constrained): enables collect_stats flag and reads
        the doubly-stochastic H_res from block.last_stats.
    """
    captured: List[torch.Tensor] = []

    # ----- HC: Hook-based reconstruction -----
    def _hc_capture_hook(module, inputs):
        residuals = inputs[0]
        streams = module.num_residual_streams

        with torch.no_grad():
            if module.channel_first:
                residuals = rearrange(residuals, "b d ... -> b ... d")
            residuals = module.split_fracs(residuals)
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

            normed = module.norm(residuals)

            # Alpha (Width Connection)
            wc_weight = module.act(normed @ module.dynamic_alpha_fn)
            dynamic_alpha = wc_weight * module.dynamic_alpha_scale
            static_alpha = rearrange(module.static_alpha, "(f s) d -> f s d", s=streams)
            alpha = dynamic_alpha + static_alpha
            alpha = module.split_fracs(alpha)

            # Beta (Depth Connection)
            beta = None
            if module.add_branch_out_to_residual:
                dc_weight = module.act(normed @ module.dynamic_beta_fn)
                if not module.has_fracs:
                    dc_weight = rearrange(dc_weight, "... -> ... 1")
                dynamic_beta = dc_weight * module.dynamic_beta_scale
                static_beta = rearrange(module.static_beta, "... (s f) -> ... s f", s=streams)
                beta = dynamic_beta + static_beta

            # Reconstruct H_res (N x N)
            alpha = rearrange(alpha, "b s 1 n 1 t -> b s n t")
            beta = rearrange(beta, "b s 1 n 1 -> b s n")

            A_branch = alpha[..., 0]   # [batch, seq, streams_in]
            A_res = alpha[..., 1:]     # [batch, seq, streams_in, streams_out]

            H_res = A_res.transpose(-1, -2) + torch.einsum(
                "bso,bsi->bsoi", beta, A_branch
            )

            H_res_mean = H_res.mean(dim=(0, 1)).to(torch.float32)
            captured.append(H_res_mean.cpu())

    hooks = []

    if method == "hc":
        print("🎣 Registering forward hooks on all HC blocks...")
        for block in model.hc_blocks:
            h = block.register_forward_pre_hook(_hc_capture_hook)
            hooks.append(h)
    elif method == "mhc":
        print("📈 Enabling stats collection on all mHC blocks...")
        for block in model.hc_blocks:
            block.collect_stats = True

    # Dummy forward pass
    seq_len = 512
    print(f"🚀 Running forward pass with dummy batch [1, {seq_len}]...")
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)

    with torch.no_grad():
        forward_with_hc(model, dummy_input, labels=dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    # For mHC: extract from last_stats
    if method == "mhc":
        for block in model.hc_blocks:
            if hasattr(block, "last_stats") and "h_res_matrix" in block.last_stats:
                captured.append(block.last_stats["h_res_matrix"].to(torch.float32).cpu())
            block.collect_stats = False

    print(f"✅ Captured {len(captured)} H_res matrices.")
    return captured


# ============================================================
# 3. PLOTTING FUNCTIONS
# ============================================================

import numpy as np

def format_scientific_latex(val: float) -> str:
    """Format large numbers as LaTeX $a \\times 10^b$ for aesthetics."""
    if np.isnan(val) or val == 0:
        return "0"
    if abs(val) < 100:
        return f"{val:.3f}"
    if abs(val) < 1000:
        return f"{val:.1f}"
    exp = int(np.floor(np.log10(abs(val))))
    coef = val / (10**exp)
    return f"${coef:.1f} \\times 10^{{{exp}}}$"

def format_sum_latex(val: float) -> str:
    """Format row/col sums."""
    if abs(val) < 100:
        return f"{val:.2f}"
    exp = int(np.floor(np.log10(abs(val))))
    coef = val / (10**exp)
    return f"${coef:.1f} \\times 10^{{{exp}}}$"

def get_annot_array(H_np: np.ndarray) -> np.ndarray:
    """Generate string annotation array for seaborn heatmap."""
    annot = np.empty_like(H_np, dtype=object)
    for i in range(H_np.shape[0]):
        for j in range(H_np.shape[1]):
            annot[i, j] = format_scientific_latex(H_np[i, j])
    return annot

def plot_single_heatmap(
    H_matrix: torch.Tensor,
    title: str,
    filename: Path,
    cmap: str = "coolwarm",
    center: float = None,
    annot: bool = True,
):
    """Plot a single H_res routing matrix as a heatmap."""
    H_np = H_matrix.detach().cpu().float().squeeze().numpy()
    assert H_np.ndim == 2, f"Expected 2D matrix, got shape {H_np.shape}"

    # Use custom latex annotations if annot is True
    annot_data = get_annot_array(H_np) if annot else False
    fmt = "" if annot else ".3f"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        H_np,
        cmap=cmap,
        center=center,
        annot=annot_data,
        fmt=fmt,
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Routing Weight", "shrink": 0.8, "pad": 0.18},
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Input Stream", fontsize=11, labelpad=20)
    ax.set_ylabel("Output Stream", fontsize=11, labelpad=10)

    # Add row/col sum annotations (Paper Section 3.1 & 5.4)
    # Row sum = Forward Signal Gain, Col sum = Backward Gradient Gain
    row_sums = H_np.sum(axis=1)
    col_sums = H_np.sum(axis=0)
    n = H_np.shape[0]

    # Row sums on the right (Forward Signal Gain)
    for i in range(n):
        ax.text(
            n + 0.3, i + 0.5, f"fwd={format_sum_latex(row_sums[i])}",
            ha="left", va="center", fontsize=8, color="#333333",
            fontstyle="italic",
        )
    # Col sums on the bottom (Backward Gradient Gain)
    for j in range(n):
        ax.text(
            j + 0.5, n + 0.3, f"bwd={format_sum_latex(col_sums[j])}",
            ha="center", va="top", fontsize=8, color="#333333",
            fontstyle="italic",
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {filename}")


def plot_side_by_side(
    H_hc: torch.Tensor,
    H_mhc: torch.Tensor,
    layer_label: str,
    filename: Path,
):
    """Plot HC vs mHC heatmaps side by side for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, H, method_name, cmap in [
        (axes[0], H_hc, "HC (Unconstrained)", "coolwarm"),
        (axes[1], H_mhc, "mHC (Sinkhorn-Constrained)", "YlGnBu"),
    ]:
        H_np = H.detach().cpu().float().squeeze().numpy()
        annot_data = get_annot_array(H_np)

        sns.heatmap(
            H_np,
            cmap=cmap,
            annot=annot_data,
            fmt="",
            linewidths=0.8,
            linecolor="white",
            square=True,
            cbar_kws={"label": "Routing Weight", "shrink": 0.8, "pad": 0.18},
            ax=ax,
        )
        ax.set_title(f"{method_name}\n{layer_label}", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Input Stream", fontsize=10, labelpad=20)
        ax.set_ylabel("Output Stream", fontsize=10, labelpad=10)

        # Row & col sums (Paper terminology)
        row_sums = H_np.sum(axis=1)
        col_sums = H_np.sum(axis=0)
        n = H_np.shape[0]
        for i in range(n):
            ax.text(
                n + 0.3, i + 0.5, f"fwd={format_sum_latex(row_sums[i])}",
                ha="left", va="center", fontsize=7, color="#333",
                fontstyle="italic",
            )
        for j in range(n):
            ax.text(
                j + 0.5, n + 0.3, f"bwd={format_sum_latex(col_sums[j])}",
                ha="center", va="top", fontsize=7, color="#333",
                fontstyle="italic",
            )

    fig.suptitle(
        "Routing Matrix H_res: HC vs mHC Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved comparison: {filename}")


def plot_all_layers_strip(
    matrices: List[torch.Tensor],
    method: str,
    filename: Path,
    max_cols: int = 8,
):
    """Plot a strip of all layer heatmaps (mini-thumbnails) for overview."""
    n_blocks = len(matrices)
    n_cols = min(max_cols, n_blocks)
    n_rows = (n_blocks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    cmap = "YlGnBu" if method == "mhc" else "coolwarm"

    for idx, (H, ax) in enumerate(zip(matrices, axes_flat)):
        H_np = H.detach().cpu().float().squeeze().numpy()
        sns.heatmap(
            H_np,
            cmap=cmap,
            annot=False,
            cbar=False,
            square=True,
            linewidths=0.3,
            ax=ax,
        )
        block_type = "Attn" if idx % 2 == 0 else "MLP"
        layer_idx = idx // 2
        ax.set_title(f"L{layer_idx} {block_type}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes_flat[n_blocks:]:
        ax.set_visible(False)

    fig.suptitle(
        f"All {n_blocks} H_res Matrices — {method.upper()}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved strip: {filename}")


# ============================================================
# 4. COMPOSITE MAPPING (Paper Figure 8 - right half)
# ============================================================

def compute_composite_matrices(
    matrices: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Compute cumulative matrix products: Π_{l=0}^{L} H_res^l.

    This reproduces the Composite Mapping from Figure 8 of the paper.
    For HC: the composite explodes (values >> 1) as layers accumulate.
    For mHC: thanks to Compositional Closure of doubly-stochastic matrices,
             the composite remains stable with row/col sums ≈ 1.0.
    """
    n = matrices[0].squeeze().shape[-1]
    composite = torch.eye(n, dtype=torch.float32)
    composites: List[torch.Tensor] = []

    for H in matrices:
        H_f32 = H.detach().cpu().float().squeeze()
        composite = H_f32 @ composite
        composites.append(composite.clone())

    return composites


def plot_figure8_quad(
    H_single_hc: torch.Tensor,
    H_single_mhc: torch.Tensor,
    H_composite_hc: torch.Tensor,
    H_composite_mhc: torch.Tensor,
    layer_label: str,
    filename: Path,
):
    """Reproduce Figure 8 from the paper: 2x2 grid.

    Top row: Single-Layer Mapping (H_res^l)
    Bottom row: Composite Mapping (Π H_res)
    Left column: HC (Unconstrained)
    Right column: mHC (Sinkhorn-Constrained)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    configs = [
        (axes[0, 0], H_single_hc,    "HC — Single-Layer Mapping",    "coolwarm"),
        (axes[0, 1], H_single_mhc,   "mHC — Single-Layer Mapping",   "YlGnBu"),
        (axes[1, 0], H_composite_hc, "HC — Composite Mapping (Π)",   "coolwarm"),
        (axes[1, 1], H_composite_mhc,"mHC — Composite Mapping (Π)",  "YlGnBu"),
    ]

    for ax, H, subtitle, cmap in configs:
        H_np = H.detach().cpu().float().squeeze().numpy()
        n = H_np.shape[0]

        annot_data = get_annot_array(H_np)

        sns.heatmap(
            H_np,
            cmap=cmap,
            annot=annot_data,
            fmt="",
            linewidths=0.6,
            linecolor="white",
            square=True,
            cbar_kws={"label": "Routing Weight", "shrink": 0.75, "pad": 0.15},
            ax=ax,
        )

        ax.set_title(f"{subtitle}\n{layer_label}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Input Stream", fontsize=9, labelpad=18)
        ax.set_ylabel("Output Stream", fontsize=9, labelpad=8)

        # Row/col sum annotations with paper terminology
        row_sums = H_np.sum(axis=1)
        col_sums = H_np.sum(axis=0)
        for i in range(n):
            ax.text(
                n + 0.3, i + 0.5, f"fwd={format_sum_latex(row_sums[i])}",
                ha="left", va="center", fontsize=7, color="#333",
                fontstyle="italic",
            )
        for j in range(n):
            ax.text(
                j + 0.5, n + 0.3, f"bwd={format_sum_latex(col_sums[j])}",
                ha="center", va="top", fontsize=7, color="#333",
                fontstyle="italic",
            )

    fig.suptitle(
        "Figure 8 Reproduction: Routing Matrix H_res\n"
        "Single-Layer (top) vs Composite Mapping (bottom)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved Figure 8 quad: {filename}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot H_res routing matrix heatmaps for HC vs mHC"
    )
    parser.add_argument(
        "--hc-ckpt", type=str, default="out-qwen-hc/ckpt.pt",
        help="Path to HC checkpoint",
    )
    parser.add_argument(
        "--mhc-ckpt", type=str, default="out-qwen-mhc/ckpt.pt",
        help="Path to mHC checkpoint",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["hc", "mhc"],
        choices=["hc", "mhc"],
        help="Which methods to plot",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[-1, 0],
        help="Block indices to plot individual heatmaps (negative = from end)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="reports/qwen-4090-full/figures",
        help="Output directory for heatmap images",
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Also generate the full strip overview of all layers",
    )
    args = parser.parse_args()

    sns.set_theme(style="white")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    dtype = torch.bfloat16

    all_matrices: Dict[str, List[torch.Tensor]] = {}

    # Load and capture for each method
    for method in args.methods:
        ckpt = args.hc_ckpt if method == "hc" else args.mhc_ckpt
        print(f"\n{'='*60}")
        print(f"  Processing {method.upper()} from {ckpt}")
        print(f"{'='*60}")

        model, tokenizer = load_model_from_ckpt(ckpt, method, device, dtype)
        matrices = capture_all_h_res(model, tokenizer, method, device)
        all_matrices[method] = matrices

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Compute Composite Matrices (Paper Figure 8 - right half) ----
    all_composites: Dict[str, List[torch.Tensor]] = {}
    for method, matrices in all_matrices.items():
        print(f"\n🧮 Computing Composite Mapping for {method.upper()}...")
        composites = compute_composite_matrices(matrices)
        all_composites[method] = composites
        # Log the final composite's max value to show explosion vs stability
        final = composites[-1].squeeze().numpy()
        print(f"   Final composite max absolute value: {abs(final).max():.4e}")
        print(f"   Final composite row sums: {final.sum(axis=1)}")

    # ---- Plot individual layer heatmaps (Single-Layer) ----
    for method, matrices in all_matrices.items():
        n_blocks = len(matrices)
        for layer_idx in args.layers:
            idx = layer_idx if layer_idx >= 0 else n_blocks + layer_idx
            if idx < 0 or idx >= n_blocks:
                print(f"⚠️ Skipping invalid block index {layer_idx} (total={n_blocks})")
                continue

            block_type = "Attn" if idx % 2 == 0 else "MLP"
            layer_num = idx // 2
            label = f"Layer {layer_num} {block_type} (Block {idx})"

            plot_single_heatmap(
                matrices[idx],
                title=f"{method.upper()} — Single-Layer — {label}",
                filename=output_dir / f"heatmap_{method}_block{idx}.png",
                cmap="YlGnBu" if method == "mhc" else "coolwarm",
                center=None if method == "mhc" else None,
            )

    # ---- Plot Composite heatmaps ----
    for method, composites in all_composites.items():
        n_blocks = len(composites)
        for layer_idx in args.layers:
            idx = layer_idx if layer_idx >= 0 else n_blocks + layer_idx
            if idx < 0 or idx >= n_blocks:
                continue

            block_type = "Attn" if idx % 2 == 0 else "MLP"
            layer_num = idx // 2
            label = f"Layer {layer_num} {block_type} (Block {idx})"

            plot_single_heatmap(
                composites[idx],
                title=f"{method.upper()} — Composite Mapping (Π) — {label}",
                filename=output_dir / f"heatmap_{method}_composite_block{idx}.png",
                cmap="YlGnBu" if method == "mhc" else "coolwarm",
                center=None if method == "mhc" else None,
            )

    # ---- Plot side-by-side comparison (Single-Layer) ----
    if "hc" in all_matrices and "mhc" in all_matrices:
        hc_mats = all_matrices["hc"]
        mhc_mats = all_matrices["mhc"]

        for layer_idx in args.layers:
            n_hc = len(hc_mats)
            n_mhc = len(mhc_mats)
            idx_hc = layer_idx if layer_idx >= 0 else n_hc + layer_idx
            idx_mhc = layer_idx if layer_idx >= 0 else n_mhc + layer_idx

            if idx_hc < 0 or idx_hc >= n_hc or idx_mhc < 0 or idx_mhc >= n_mhc:
                continue

            block_type = "Attn" if idx_hc % 2 == 0 else "MLP"
            layer_num = idx_hc // 2
            label = f"Layer {layer_num} {block_type} (Block {idx_hc})"

            plot_side_by_side(
                hc_mats[idx_hc],
                mhc_mats[idx_mhc],
                layer_label=label,
                filename=output_dir / f"heatmap_comparison_block{idx_hc}.png",
            )

    # ---- Plot Figure 8 Quad (Single-Layer + Composite, HC vs mHC) ----
    if "hc" in all_matrices and "mhc" in all_matrices:
        hc_mats = all_matrices["hc"]
        mhc_mats = all_matrices["mhc"]
        hc_comp = all_composites["hc"]
        mhc_comp = all_composites["mhc"]

        for layer_idx in args.layers:
            n_hc = len(hc_mats)
            n_mhc = len(mhc_mats)
            idx_hc = layer_idx if layer_idx >= 0 else n_hc + layer_idx
            idx_mhc = layer_idx if layer_idx >= 0 else n_mhc + layer_idx

            if idx_hc < 0 or idx_hc >= n_hc or idx_mhc < 0 or idx_mhc >= n_mhc:
                continue

            block_type = "Attn" if idx_hc % 2 == 0 else "MLP"
            layer_num = idx_hc // 2
            label = f"Layer {layer_num} {block_type} (Block {idx_hc})"

            plot_figure8_quad(
                H_single_hc=hc_mats[idx_hc],
                H_single_mhc=mhc_mats[idx_mhc],
                H_composite_hc=hc_comp[idx_hc],
                H_composite_mhc=mhc_comp[idx_mhc],
                layer_label=label,
                filename=output_dir / f"figure8_block{idx_hc}.png",
            )

    # ---- Plot all-layers strip ----
    if args.all_layers:
        for method, matrices in all_matrices.items():
            plot_all_layers_strip(
                matrices,
                method=method,
                filename=output_dir / f"heatmap_strip_{method}.png",
            )

    print(f"\n🎉 All heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
