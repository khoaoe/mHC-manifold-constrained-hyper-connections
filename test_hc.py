import torch
import sys
import os

from hyper_connections.hyper_connections import HyperConnections

hc = HyperConnections(num_residual_streams=4, dim=512, mhc=True)
hc.collect_stats = True
x = torch.randn(2, 16, 4, 512)
branch_input, residuals_out, residual_kwargs = hc.width_connection(x)

stats = hc.last_stats
print("H_pre range:", stats.get('h_pre_min', 'N/A'), "to", stats.get('h_pre_max', 'N/A'))
print("H_res row_sum:", stats.get('h_res_row_sum', 'N/A'), "col_sum:", stats.get('h_res_col_sum', 'N/A'))
print("Shape branch_input:", branch_input.shape)
print("Shape residuals_out:", residuals_out.shape)
