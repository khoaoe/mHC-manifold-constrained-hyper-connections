#!/bin/bash
# ==============================================================================
# Script tự động chạy Linear Probing cho cả 3 methods × 2 tasks rồi vẽ biểu đồ
#
# Chạy: bash run_glue_experiments.sh
# ==============================================================================

set -e  # Dừng ngay nếu có lỗi

echo "=========================================================="
echo "🚀 CHẠY GLUE LINEAR PROBING: SST-2 + MRPC"
echo "   3 methods: Baseline / HC / mHC"
echo "=========================================================="

# ------------------------------------
# 1. BASELINE: SST-2 + MRPC
# ------------------------------------
echo -e "\n\n━━━ [1/3] BASELINE ━━━"
python run_finetune_glue.py \
    --method baseline \
    --ckpt out-qwen-baseline/ckpt.pt \
    --tasks sst2 mrpc \
    --linear-probe \
    --lr 3e-3 \
    --epochs 10

# ------------------------------------
# 2. HC: SST-2 + MRPC
# ------------------------------------
echo -e "\n\n━━━ [2/3] HC ━━━"
python run_finetune_glue.py \
    --method hc \
    --ckpt out-qwen-hc/ckpt.pt \
    --tasks sst2 mrpc \
    --linear-probe \
    --lr 3e-3 \
    --epochs 10

# ------------------------------------
# 3. mHC: SST-2 + MRPC
# ------------------------------------
echo -e "\n\n━━━ [3/3] mHC ━━━"
python run_finetune_glue.py \
    --method mhc \
    --ckpt out-qwen-mhc/ckpt.pt \
    --tasks sst2 mrpc \
    --linear-probe \
    --lr 3e-3 \
    --epochs 10

# ------------------------------------
# 4. Vẽ biểu đồ tổng hợp
# ------------------------------------
echo -e "\n\n━━━ [4/4] VẼ BIỂU ĐỒ BÁO CÁO ━━━"
python plot_glue_results.py \
    --task both \
    --sst2-runs baseline=out-qwen-baseline-sst2-probe \
                hc=out-qwen-hc-sst2-probe \
                mhc=out-qwen-mhc-sst2-probe \
    --mrpc-runs baseline=out-qwen-baseline-mrpc-probe \
                hc=out-qwen-hc-mrpc-probe \
                mhc=out-qwen-mhc-mrpc-probe \
    --output-dir reports/glue-linear-probe

echo -e "\n\n🎉 HOÀN TẤT! Biểu đồ báo cáo: reports/glue-linear-probe/"
echo "   Gồm: sst2_combined_panel.png"
echo "         mrpc_combined_panel.png"
echo "         glue_cross_task_summary.png  <-- BIỂU ĐỒ CHÍNH CHO BÁO CÁO"
