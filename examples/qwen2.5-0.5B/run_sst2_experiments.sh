#!/bin/bash
# Script tự động chạy Linear Probing cho cả 3 method và vẽ biểu đồ

set -e # Dừng script nếu có lỗi

echo "=========================================================="
echo "🚀 BẮT ĐẦU CHẠY SST-2 LINEAR PROBING EXPERIMENTS"
echo "=========================================================="

# 1. Chạy Baseline
echo -e "\n\n---> [1/3] Đang chạy Linear Probing cho BASELINE..."
python run_finetune_sst2.py \
    --method baseline \
    --ckpt out-qwen-baseline/ckpt.pt \
    --linear-probe

# 2. Chạy HC
echo -e "\n\n---> [2/3] Đang chạy Linear Probing cho HC (Static)..."
python run_finetune_sst2.py \
    --method hc \
    --ckpt out-qwen-hc/ckpt.pt \
    --linear-probe

# 3. Chạy mHC
echo -e "\n\n---> [3/3] Đang chạy Linear Probing cho mHC (Manifold-Constrained)..."
python run_finetune_sst2.py \
    --method mhc \
    --ckpt out-qwen-mhc/ckpt.pt \
    --linear-probe

# 4. Vẽ biểu đồ báo cáo
echo -e "\n\n---> [4/4] Đang tổng hợp kết quả và vẽ biểu đồ báo cáo..."
python plot_sst2_results.py \
    --runs baseline=out-qwen-baseline-sst2-probe \
           hc=out-qwen-hc-sst2-probe \
           mhc=out-qwen-mhc-sst2-probe \
    --output-dir reports/sst2-linear-probe

echo -e "\n\n🎉 XONG! Biểu đồ báo cáo đã được lưu tại: reports/sst2-linear-probe/"
