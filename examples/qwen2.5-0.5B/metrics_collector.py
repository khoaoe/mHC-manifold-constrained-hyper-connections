import torch
import math

class PaperMetricsCollector:
    def __init__(self, model, n_streams):
        self.model = model
        self.n_streams = n_streams
        self.base_model = model.model if hasattr(model, 'model') else model

    def enable_collection(self):
        self.model.collect_metrics = True
        if hasattr(self.base_model, "config"):
            self.base_model.config.output_attentions = True
        self.model._residual_norms = []
        self.base_model._attn_weights = []
        
        if hasattr(self.model, "hc_blocks") and self.model.hc_blocks is not None:
            for block in self.model.hc_blocks:
                block.collect_stats = True

    def disable_collection(self):
        self.model.collect_metrics = False
        if hasattr(self.base_model, "config"):
            self.base_model.config.output_attentions = False
        if hasattr(self.model, "hc_blocks") and self.model.hc_blocks is not None:
            for block in self.model.hc_blocks:
                block.collect_stats = False
        
        self.model._residual_norms = []
        self.base_model._attn_weights = []

    def compute_metrics(self):
        metrics = {}
        
        # 1. Composite Amax Gain & Single-Layer Deviation
        h_res_matrices = []
        row_sums = []
        col_sums = []
        max_devs = []
        
        if hasattr(self.model, "hc_blocks") and self.model.hc_blocks is not None:
            for block in self.model.hc_blocks:
                if hasattr(block, "last_stats") and "h_res_matrix" in block.last_stats:
                    H = block.last_stats["h_res_matrix"]
                    h_res_matrices.append(H)
                    
                    row_sum = H.sum(dim=-1)
                    col_sum = H.sum(dim=-2)
                    
                    row_sums.append(row_sum.mean().item())
                    col_sums.append(col_sum.mean().item())
                    max_devs.append((row_sum - 1.0).abs().max().item())
        
        if h_res_matrices:
            composite_fwd = torch.eye(self.n_streams, device=h_res_matrices[0].device)
            composite_bwd = torch.eye(self.n_streams, device=h_res_matrices[0].device)
            
            max_fwd_gain = 1.0
            max_bwd_gain = 1.0
            
            for H in h_res_matrices:
                composite_fwd = H @ composite_fwd
                composite_bwd = composite_bwd @ H
                
                max_fwd_gain = max(max_fwd_gain, composite_fwd.abs().sum(dim=-1).max().item())
                max_bwd_gain = max(max_bwd_gain, composite_bwd.abs().sum(dim=-2).max().item())
                
            metrics["amax_fwd_max"] = max_fwd_gain
            metrics["amax_bwd_max"] = max_bwd_gain
            metrics["h_res_row_sum_mean"] = sum(row_sums) / len(row_sums)
            metrics["h_res_col_sum_mean"] = sum(col_sums) / len(col_sums)
            metrics["h_res_row_sum_max_dev"] = max(max_devs)
        else:
            metrics["amax_fwd_max"] = 1.0
            metrics["amax_bwd_max"] = 1.0
            metrics["h_res_row_sum_mean"] = 1.0
            metrics["h_res_col_sum_mean"] = 1.0
            metrics["h_res_row_sum_max_dev"] = 0.0

        # 2. Attention Entropy
        attn_entropies = []
        if hasattr(self.base_model, "_attn_weights"):
            for attn_w in self.base_model._attn_weights:
                # attn_w shape: [batch, heads, seq, seq]
                p = attn_w.float() + 1e-8
                entropy = -(p * torch.log2(p)).sum(dim=-1).mean().item()
                attn_entropies.append(entropy)
                
        if attn_entropies:
            metrics["avg_attn_entropy"] = sum(attn_entropies) / len(attn_entropies)
        else:
            metrics["avg_attn_entropy"] = 0.0

        # 3. Residual Stream L2 Norm
        if hasattr(self.model, "_residual_norms") and self.model._residual_norms:
            metrics["residual_norm_final"] = self.model._residual_norms[-1]
            metrics["residual_norm_layer_curve"] = self.model._residual_norms
        else:
            metrics["residual_norm_final"] = 0.0
            metrics["residual_norm_layer_curve"] = []
            
        return metrics
