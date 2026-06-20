#!/usr/bin/env bash
# ============================================================================
# Qwen2.5-0.5B + mHC/HC/Baseline -- Quick Test (50 iterations)
# Usage:
#   chmod +x test_qwen_50_iters.sh
#   ./test_qwen_50_iters.sh
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$ROOT/../examples/qwen2.5-0.5B"
cd "$QWEN_DIR"

# ---------------- Config ----------------
DATA_DIR="$QWEN_DIR/data/tinystories"
REPORT_DIR="$ROOT/reports/test-50-iters"
LOG_DIR="$ROOT/logs"

# --- TEST CONFIG ---
MAX_ITERS=50
WARMUP_ITERS=5
EVAL_INTERVAL=25
EVAL_ITERS=10

BATCH_SIZE="${BATCH_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DTYPE="${DTYPE:-bfloat16}"
N_STREAMS="${N_STREAMS:-4}"
SINKHORN_TMAX="${SINKHORN_TMAX:-20}"
LR="${LR:-6e-4}"

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$LOG_DIR"

echo "============================================"
echo "  Test Qwen2.5-0.5B - 50 Iters"
echo "============================================"
echo "Running in:     $QWEN_DIR"
echo "DATA_DIR:       $DATA_DIR"
echo "LOG_DIR:        $LOG_DIR"
echo "MAX_ITERS:      $MAX_ITERS"
echo "WARMUP_ITERS:   $WARMUP_ITERS"
echo "EVAL_INTERVAL:  $EVAL_INTERVAL"
echo "============================================"

# ---------------- Helper: check if run finished ----------------
run_finished() {
    local out_dir="$1"
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
    local out_dir="$ROOT/out-test-$name"
    local log_file="$LOG_DIR/test-$name.log"

    echo ""
    echo "========================================"
    echo "[RUN] Variant: $name (method=$method)"
    echo "      Out:  $out_dir"
    echo "      Log:  $log_file"
    echo "========================================"

    # Force retrain for test
    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    local current_batch_size=$BATCH_SIZE
    local current_grad_accum=$GRAD_ACCUM
    if [ "$method" = "baseline" ]; then
        current_batch_size=8
        current_grad_accum=8
    else
        current_batch_size=4
        current_grad_accum=16
    fi

    # Activate venv before running
    set +e
    source "$ROOT/../.venv/bin/activate" 2>/dev/null
    set -e

    # Use stdbuf for realtime logging and tee to file
    stdbuf -oL -eL python -u train_qwen_hc.py \
        --method "$method" \
        --out-dir "$out_dir" \
        --data-dir "$DATA_DIR" \
        --n-streams "$N_STREAMS" \
        --sinkhorn-tmax "$SINKHORN_TMAX" \
        --max-iters "$MAX_ITERS" \
        --warmup-iters "$WARMUP_ITERS" \
        --eval-interval "$EVAL_INTERVAL" \
        --eval-iters "$EVAL_ITERS" \
        --batch-size "$current_batch_size" \
        --grad-accum "$current_grad_accum" \
        --block-size "$BLOCK_SIZE" \
        --dtype "$DTYPE" \
        --device cuda \
        --compile true \
        --lr "$LR" \
        --weight-decay 0.1 \
        --pretrained 0 \
        --num-fracs 1 \
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
        baseline="$ROOT/out-test-baseline" \
        hc="$ROOT/out-test-hc" \
        mhc="$ROOT/out-test-mhc" \
    --output-dir "$REPORT_DIR"

# ---------------- Plot ----------------
echo ""
echo "[INFO] Generating figures..."
FIG_DIR="$REPORT_DIR/figures"
mkdir -p "$FIG_DIR"
python plot_qwen_comparison.py \
    --runs \
        baseline="$ROOT/out-test-baseline" \
        hc="$ROOT/out-test-hc" \
        mhc="$ROOT/out-test-mhc" \
    --output-dir "$FIG_DIR"

echo ""
echo "============================================"
echo "  [DONE] TEST COMPLETE"
echo "============================================"
echo "Reports:  $REPORT_DIR"
echo "Figures:  $FIG_DIR"
echo "Logs:     $LOG_DIR"
