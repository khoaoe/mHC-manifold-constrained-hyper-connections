#!/usr/bin/env bash
# ============================================================================
# Qwen2.5-0.5B + mHC/HC/Baseline -- Full experiment on RTX 4090
# Usage:
#   chmod +x run_qwen_4090.sh
#   ./run_qwen_4090.sh                 # run all (skip finished variants)
#   FORCE_RETRAIN=1 ./run_qwen_4090.sh # force retrain
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------- Config ----------------
DATA_DIR="$ROOT/data/fineweb10B"
REPORT_DIR="$ROOT/reports/qwen-4090-full"
LOG_DIR="$ROOT/logs"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-7}"   # 7 shards là dư xăng chạy (~655M tokens)

# --- FINAL CONFIG CHO RTX 4090 ---
MAX_ITERS="${MAX_ITERS:-1000}"       # Chạy 1000 bước cập nhật tạ
BATCH_SIZE="${BATCH_SIZE:-8}"        # Nhồi 8 chuỗi vào để vắt kiệt Tensor Core
GRAD_ACCUM="${GRAD_ACCUM:-64}"       # 8 * 64 = 512 (Effective Batch mượt như lụa)
BLOCK_SIZE="${BLOCK_SIZE:-1024}"     # Vừa đủ ngữ cảnh, VRAM thở oxy khỏe re (Sequence Length)
DTYPE="${DTYPE:-bfloat16}"
N_STREAMS="${N_STREAMS:-4}"          # BẮT BUỘC để n=4 giữ đúng chuẩn paper
SINKHORN_TMAX="${SINKHORN_TMAX:-20}"
AMAX_LOG_INTERVAL="${AMAX_LOG_INTERVAL:-100}"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
PRETRAINED="${PRETRAINED:-0}"

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$LOG_DIR"

echo "============================================"
echo "  Qwen2.5-0.5B Experiment on RTX 4090"
echo "============================================"
echo "ROOT:           $ROOT"
echo "DATA_DIR:       $DATA_DIR"
echo "REPORT_DIR:     $REPORT_DIR"
echo "MAX_ITERS:      $MAX_ITERS"
echo "BATCH x ACCUM:  ${BATCH_SIZE}x${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "N_STREAMS:      $N_STREAMS"
echo "DTYPE:          $DTYPE"
echo "FORCE_RETRAIN:  $FORCE_RETRAIN"
echo "PRETRAINED:     $PRETRAINED"
echo "============================================"

# ---------------- GPU check ----------------
echo ""
echo "[INFO] Checking GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'GPU: {name}')
print(f'VRAM: {vram:.1f} GB')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
if vram < 22:
    raise SystemExit(f'Need >=24GB VRAM, found {vram:.1f}GB')
"

# ---------------- Download data ----------------
echo ""
echo "[INFO] Checking FineWeb10B data..."
N_SHARDS=$(ls "$DATA_DIR"/fineweb_*.bin 2>/dev/null | wc -l || echo 0)
if [ "$N_SHARDS" -lt "$((NUM_TRAIN_SHARDS + 1))" ]; then
    echo "   Downloading $NUM_TRAIN_SHARDS train shards + 1 val shard..."
    if [ -f "$ROOT/download_fineweb.py" ]; then
        python "$ROOT/download_fineweb.py" --output-dir "$DATA_DIR"
    else
        echo "[ERROR] download_fineweb.py not found"
        exit 1
    fi
fi
echo "[INFO] Data shards available: $(ls "$DATA_DIR"/fineweb_*.bin | wc -l)"

# ---------------- Helper: check if run finished ----------------
run_finished() {
    local out_dir="$1"
    if [ "$FORCE_RETRAIN" = "1" ] || [ "$FORCE_RETRAIN" = "true" ]; then
        return 1
    fi
    if [ ! -f "$out_dir/ckpt.pt" ] || [ ! -f "$out_dir/summary.json" ]; then
        return 1
    fi
    python -c "
import json, sys
s = json.load(open('$out_dir/summary.json'))
if not s.get('ok'): sys.exit(1)
if int(s.get('iter_num', 0)) < $MAX_ITERS: sys.exit(1)
" 2>/dev/null
}

# ---------------- Train function ----------------
train_variant() {
    local name="$1"
    local method="$2"
    local out_dir="$ROOT/out-qwen-$name"
    local log_file="$LOG_DIR/qwen-4090-$name.log"

    echo ""
    echo "========================================"
    echo "[RUN] Variant: $name (method=$method)"
    echo "      Out:  $out_dir"
    echo "      Log:  $log_file"
    echo "========================================"

    if run_finished "$out_dir"; then
        echo "[SKIP] $name already complete. Set FORCE_RETRAIN=1 to rerun."
        return 0
    fi

    if [ "$FORCE_RETRAIN" = "1" ] && [ -d "$out_dir" ]; then
        echo "[INFO] Removing old run: $out_dir"
        rm -rf "$out_dir"
    fi

    mkdir -p "$out_dir"

    # Use stdbuf for realtime logging and tee to file
    stdbuf -oL -eL python -u train_qwen_hc.py \
        --method "$method" \
        --out-dir "$out_dir" \
        --data-dir "$DATA_DIR" \
        --n-streams "$N_STREAMS" \
        --sinkhorn-tmax "$SINKHORN_TMAX" \
        --max-iters "$MAX_ITERS" \
        --eval-interval 500 \
        --eval-iters 50 \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --block-size "$BLOCK_SIZE" \
        --dtype "$DTYPE" \
        --device cuda \
        --compile true \
        --lr 5e-4 \
        --weight-decay 0.1 \
        --amax-log-interval "$AMAX_LOG_INTERVAL" \
        --amax-eval true \
        --pretrained "$PRETRAINED" \
        2>&1 | tee "$log_file"

    if ! run_finished "$out_dir"; then
        echo "[ERROR] $name did not complete. See log: $log_file"
        exit 1
    fi

    echo "[OK] $name done. Summary: $out_dir/summary.json"
}

# ---------------- Train 3 variants ----------------
train_variant "baseline" "baseline"
train_variant "hc"       "hc"
train_variant "mhc"      "mhc"

# ---------------- Summarize ----------------
echo ""
echo "[INFO] Summarizing training runs..."
python summarize_qwen_runs.py \
    --runs \
        baseline="$ROOT/out-qwen-baseline" \
        hc="$ROOT/out-qwen-hc" \
        mhc="$ROOT/out-qwen-mhc" \
    --output-dir "$REPORT_DIR"

# ---------------- Plot ----------------
echo ""
echo "[INFO] Generating figures..."
FIG_DIR="$REPORT_DIR/figures"
mkdir -p "$FIG_DIR"
python plot_qwen_comparison.py \
    --runs \
        baseline="$ROOT/out-qwen-baseline" \
        hc="$ROOT/out-qwen-hc" \
        mhc="$ROOT/out-qwen-mhc" \
    --output-dir "$FIG_DIR"

# ---------------- Final summary ----------------
echo ""
echo "============================================"
echo "  [DONE] EXPERIMENT COMPLETE"
echo "============================================"
echo "Reports:  $REPORT_DIR"
echo "Figures:  $FIG_DIR"
echo "Logs:     $LOG_DIR"
echo ""
echo "Key files:"
echo "  - training_summary.md"
echo "  - training_summary.csv"
echo "  - figures/amax_fwd_curve.png  (Fig 3 style)"
echo "  - figures/amax_bwd_curve.png  (Fig 7 style)"
echo "  - figures/val_loss_curve.png"
echo ""

# Print summary
if [ -f "$REPORT_DIR/training_summary.md" ]; then
    echo "=== Training Summary ==="
    cat "$REPORT_DIR/training_summary.md"
fi
