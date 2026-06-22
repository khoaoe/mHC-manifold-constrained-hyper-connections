import os
import sys
import torch
import matplotlib.pyplot as plt
from einops import rearrange
from transformers import AutoConfig

# Ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qwen_hc_model import build_qwen_hc, forward_with_hc

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze amax gain curves for HC/mHC")
    parser.add_argument("--ckpt-path", type=str, default="out-qwen-hc/ckpt.pt", help="Path to checkpoint file")
    parser.add_argument("--method", type=str, default="hc", choices=["hc", "mhc"], help="Architecture method")
    parser.add_argument("--output-path", type=str, default="amax_fwd_curve_mhc.png", help="Output path for plot")
    args = parser.parse_args()

    device = "cpu"
    dtype = torch.bfloat16
    ckpt_path = args.ckpt_path
    method = args.method

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

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

    global_h_res_matrices = []

    def capture_routing_hook(module, inputs):
        """
        Forward Pre-Hook to capture dynamically computed routing variables
        from unconstrained HyperConnections and mathematically reconstruct H_res.
        """
        residuals = inputs[0]
        streams = module.num_residual_streams

        with torch.no_grad():
            # 1. Reproduce dynamic routing variables as per the HC paper formulation
            if module.channel_first:
                residuals = rearrange(residuals, "b d ... -> b ... d")
            residuals = module.split_fracs(residuals)
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

            normed = module.norm(residuals)

            # --- Alpha (Width Connection) ---
            wc_weight = module.act(normed @ module.dynamic_alpha_fn)
            dynamic_alpha = wc_weight * module.dynamic_alpha_scale
            static_alpha = rearrange(module.static_alpha, "(f s) d -> f s d", s=streams)
            alpha = dynamic_alpha + static_alpha
            alpha = module.split_fracs(alpha)

            # --- Beta (Depth Connection) ---
            beta = None
            if module.add_branch_out_to_residual:
                dc_weight = module.act(normed @ module.dynamic_beta_fn)
                if not module.has_fracs:
                    dc_weight = rearrange(dc_weight, "... -> ... 1")
                dynamic_beta = dc_weight * module.dynamic_beta_scale
                static_beta = rearrange(module.static_beta, "... (s f) -> ... s f", s=streams)
                beta = dynamic_beta + static_beta

            # 2. Reconstruct H_res N x N matrix mathematically
            # Simplify shape to [batch, seq, streams,            # Assume num_fracs = 1 and num_input_views = 1
            alpha = rearrange(alpha, "b s 1 n 1 t -> b s n t")
            beta = rearrange(beta, "b s 1 n 1 -> b s n")

            # Extract weights
            # alpha[..., 0]: weights mapping input stream -> branch
            A_branch = alpha[..., 0] # [batch, seq, streams_in]
            
            # alpha[..., 1:]: weights mapping input stream -> bypassed streams
            A_res = alpha[..., 1:]   # [batch, seq, streams_in, streams_out]

            # Reconstruct implicit H_res: H_res[out, in] = A_res[in, out] + beta[out] * A_branch[in]
            # Transpose A_res to be [..., out, in], then compute outer product beta ⊗ A_branch
            H_res = A_res.transpose(-1, -2) + torch.einsum('bso,bsi->bsoi', beta, A_branch)

            # Average over seq and batch dynamically to capture the layer's structural property
            H_res_mean = H_res.mean(dim=(0, 1))
            global_h_res_matrices.append(H_res_mean.cpu())

    if method == "hc":
        print("🎣 Registering Forward Hooks on all HC Blocks...")
        for block in model.hc_blocks:
            block.register_forward_pre_hook(capture_routing_hook)
    elif method == "mhc":
        print("📈 Enabling stats collection on all mHC Blocks...")
        for block in model.hc_blocks:
            block.collect_stats = True

    # Dummy input
    seq_len = 1024
    print(f"🚀 Running forward pass with dummy batch [1, {seq_len}]...")
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
    
    with torch.no_grad():
        forward_with_hc(model, dummy_input, labels=dummy_input)

    if method == "mhc":
        for block in model.hc_blocks:
            if hasattr(block, "last_stats") and "h_res_matrix" in block.last_stats:
                global_h_res_matrices.append(block.last_stats["h_res_matrix"].cpu())

    print(f"✅ Captured {len(global_h_res_matrices)} explicit routing matrices (H_res).")

    # Compute Amax Gain across all layers
    print("🧮 Computing Composite Amax Gain...")
    streams = model.hc_blocks[0].num_residual_streams
    
    # 1. Bắt buộc dùng float32 để chịu đựng sự bùng nổ của HC
    composite_fwd = torch.eye(streams, dtype=torch.float32)
    composite_bwd = torch.eye(streams, dtype=torch.float32)

    amax_fwd_list = []
    amax_bwd_list = []

    for H in global_h_res_matrices:
        # 2. Ép H lên float32 trước khi nhân
        H = H.to(torch.float32)
        
        composite_fwd = H @ composite_fwd
        composite_bwd = composite_bwd @ H
        
        amax_fwd = composite_fwd.abs().sum(dim=-1).max().item()
        amax_bwd = composite_bwd.abs().sum(dim=-2).max().item()
        
        amax_fwd_list.append(amax_fwd)
        amax_bwd_list.append(amax_bwd)

    print(f"🔥 Final Composite Amax Fwd: {amax_fwd_list[-1]:.4e}")
    print(f"🔥 Final Composite Amax Bwd: {amax_bwd_list[-1]:.4e}")

    # Plot results
    print("📈 Generating Amax Gain curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(amax_fwd_list) + 1), amax_fwd_list, label='Amax Fwd (Forward Signal Amplification)', color='red', linewidth=2)
    plt.plot(range(1, len(amax_bwd_list) + 1), amax_bwd_list, label='Amax Bwd (Backward Gradient Amplification)', color='blue', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Stable Baseline (1.0)', linewidth=2)
    plt.yscale('log')
    plt.xlabel('HC Block Index (2 per layer: Attention, MLP)', fontsize=12)
    plt.ylabel('Composite Amax Gain (Log Scale)', fontsize=12)
    plt.title(f'Explosion of {method.upper()} Signal Routing Over Layers', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output_path, dpi=300)
    print(f"✅ Plot successfully saved to {args.output_path}")

if __name__ == "__main__":
    main()
