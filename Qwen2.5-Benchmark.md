# 🔄 Workflow đầy đủ cho Qwen2.5-0.5B Experiment

Đây là workflow tổng thể từ lúc chưa có gì đến lúc có báo cáo cuối cùng, chia thành **5 phases**:

---

## 📋 High-Level Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP (5 phút, chạy 1 lần)                            │
│  ──────────────────────────────────────                          │
│  • Clone repo vào máy có 4090                                    │
│  • Cài dependencies                                              │
│  • Download FineWeb10B (10 shards ≈ 10GB, ~15 phút)              │
│  • Verify GPU + VRAM                                             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: TRAINING (2-3 giờ/variant × 3 variants ≈ 6-9 giờ)      │
│  ─────────────────────────────────────                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │  baseline    │ → │     hc       │ → │     mhc      │         │
│  │  (skip HC    │   │ (no Sinkhorn)│   │ (with Sinkh.)│         │
│  │   params)    │   │              │   │              │         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│    out-qwen-*/        out-qwen-*/       out-qwen-*/             │
│    ├── ckpt.pt        ├── ckpt.pt       ├── ckpt.pt             │
│    ├── summary.json   ├── summary.json  ├── summary.json        │
│    └── history.csv    └── history.csv   └── history.csv         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: SUMMARIZE (30 giây)                                   │
│  ─────────────────────                                           │
│  • summarize_qwen_runs.py → training_summary.{csv,json,md}      │
│  • Kiểm tra: Amax Gain HC >> 1.0, Amax Gain mHC ≈ 1.0           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: PLOT (30 giây)                                        │
│  ─────────────                                                   │
│  • plot_qwen_comparison.py → 6 figures PNG                      │
│  • Key figures: amax_fwd_curve.png, amax_bwd_curve.png          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: REPORT & ARCHIVE                                      │
│  ─────────────────────────                                      │
│  • Zip reports/                                                 │
│  • Viết kết luận (xem template bên dưới)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Phase 1: Setup (chạy 1 lần)

```bash
# 1.1 SSH vào máy có RTX 4090
ssh user@your-4090-server

# 1.2 Clone repo + checkout branch đúng
git clone https://github.com/khoaoe/mHC-... mhc
cd mhc
git checkout qwen-4090-experiment  # branch chứa 4 file Qwen mới

# 1.3 Cài dependencies
cd examples/qwen2.5-0.5B
python -m venv .venv && source .venv/bin/activate
pip install -U pip torch transformers datasets accelerate tiktoken \
    einops pandas matplotlib seaborn huggingface_hub

# 1.4 Verify GPU
python -c "
import torch
print(torch.cuda.get_device_name(0))
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'bf16: {torch.cuda.is_bf16_supported()}')
"
# Expected: RTX 4090, 24GB, bf16=True

# 1.5 Download FineWeb10B shards (dùng script của nanoGPT)
cd ../nanogpt
python data/fineweb10B/download.py 9   # 9 train shards
# ~15 phút, ~10GB

# 1.6 Symlink data sang Qwen folder
cd ../qwen2.5-0.5B
mkdir -p data/fineweb10B
ln -s ../../nanogpt/data/fineweb10B/*.bin data/fineweb10B/
ls data/fineweb10B/   # Phải thấy 10 file .bin
```

**✅ Checkpoint Phase 1:** `ls data/fineweb10B/*.bin | wc -l` phải ra `10`

---

## 🚂 Phase 2: Training (chạy tự động, có resume)

### Cách 1: Dùng bash script (khuyến nghị)

```bash
# Chạy full 3 variants, tự skip nếu đã xong
chmod +x run_qwen_4090.sh
nohup ./run_qwen_4090.sh > experiment.log 2>&1 &
tail -f experiment.log
```

### Cách 2: Chạy thủ công từng variant (debug)

```bash
cd examples/qwen2.5-0.5B

# Variant 1: baseline (~2h trên 4090)
python train_qwen_hc.py \
    --method baseline \
    --out-dir out-qwen-baseline \
    --data-dir data/fineweb10B \
    --max-iters 5000 --batch-size 4 --grad-accum 16 \
    --dtype bfloat16 --compile true \
    2>&1 | tee logs/baseline.log

# Variant 2: hc (~2.5h)
python train_qwen_hc.py \
    --method hc \
    --out-dir out-qwen-hc \
    --data-dir data/fineweb10B \
    --max-iters 5000 --batch-size 4 --grad-accum 16 \
    --dtype bfloat16 --compile true \
    --amax-log-interval 100 \
    2>&1 | tee logs/hc.log

# Variant 3: mhc (~3h, chậm hơn do Sinkhorn 20 iters)
python train_qwen_hc.py \
    --method mhc \
    --out-dir out-qwen-mhc \
    --data-dir data/fineweb10B \
    --max-iters 5000 --batch-size 4 --grad-accum 16 \
    --dtype bfloat16 --compile true \
    --sinkhorn-tmax 20 --amax-log-interval 100 \
    2>&1 | tee logs/mhc.log
```

**✅ Checkpoint Phase 2:** Mỗi variant phải có:
```
out-qwen-{variant}/
├── ckpt.pt           # ~1-2 GB
├── summary.json      # có "ok": true
└── history.csv       # ~500 dòng (mỗi 10 iter 1 dòng)
```

**🔍 Sanity check nhanh giữa chừng:**
```bash
# Xem progress real-time
tail -f logs/mhc.log

# Kiểm tra variant đã xong chưa
python -c "
import json
for v in ['baseline', 'hc', 'mhc']:
    s = json.load(open(f'out-qwen-{v}/summary.json'))
    print(f'{v:8s} ok={s[\"ok\"]} iter={s[\"iter_num\"]} '
          f'ppl={s[\"final_val_ppl\"]:.1f} '
          f'amax_fwd={s[\"amax_fwd_max\"]:.2f}')
"
```

### 🚨 Xử lý khi crash

Nếu GPU bị OOM hoặc SSH rớt:
```bash
# Script tự skip variant đã xong (nhờ check summary.json)
./run_qwen_4090.sh  # chỉ chạy tiếp variant chưa xong

# Nếu muốn train lại 1 variant từ đầu:
rm -rf out-qwen-hc
python train_qwen_hc.py --method hc ...
```

---

## 📊 Phase 3: Summarize

```bash
cd examples/qwen2.5-0.5B

python summarize_qwen_runs.py \
    --runs \
        baseline=out-qwen-baseline \
        hc=out-qwen-hc \
        mhc=out-qwen-mhc \
    --output-dir reports/qwen-4090-full

cat reports/qwen-4090-full/training_summary.md
```

**✅ Checkpoint Phase 3:** Bảng summary phải show pattern:

| variant | final_val_ppl | amax_fwd_max | amax_bwd_max |
|---------|:---:|:---:|:---:|
| baseline | cao (~30-50) | 1.0 | 1.0 |
| **hc** | có thể NaN/spike | **>> 1.0** (5-100) | **>> 1.0** |
| **mhc** | thấp nhất | **≈ 1.0-1.6** | **≈ 1.0-1.6** |

**🚩 Red flags cần investigate:**
- HC không thấy Amax Gain tăng → kiểm tra `qwen_hc_model.py`, có thể method='hc' bị nhầm thành 'mhc'
- mHC Amax Gain > 2.0 → tăng `--sinkhorn-tmax` lên 50
- Cả 3 variants PPL giống hệt nhau → có thể HC block không được inject đúng

---

## 📈 Phase 4: Plot Figures

```bash
python plot_qwen_comparison.py \
    --runs \
        baseline=out-qwen-baseline \
        hc=out-qwen-hc \
        mhc=out-qwen-mhc \
    --output-dir reports/qwen-4090-full/figures

ls reports/qwen-4090-full/figures/
# val_loss_curve.png
# train_loss_curve.png
# best_val_loss_bar.png
# final_val_ppl_bar.png
# amax_fwd_curve.png   ← QUAN TRỌNG NHẤT
# amax_bwd_curve.png   ← QUAN TRỌNG THỨ 2
```

**✅ Checkpoint Phase 4:** Mở `amax_fwd_curve.png` và verify:
- Đường **baseline** nằm ngang ở y=1.0 (nét đứt)
- Đường **HC** tăng dần (có thể lên tới 10-100 ở cuối)
- Đường **mHC** nằm sát y=1.0 (có thể dao động ~1.0-1.6)

---

## 📝 Phase 5: Report

Tạo file `reports/qwen-4090-full/REPORT.md` với template:

```markdown
# mHC vs HC vs Baseline -- Qwen2.5-0.5B on RTX 4090

## Setup
- Model: Qwen2.5-0.5B (~494M params, random init)
- Dataset: FineWeb10B, 10 shards ≈ 1B tokens
- Hardware: RTX 4090 24GB, bfloat16
- Training: 5000 iters, batch=4, grad_accum=16 (effective=64)
- Expansion rate n=4, Sinkhorn tmax=20

## Key Finding
HC exhibits severe signal amplification (Amax Gain peaks at X),
confirming the theoretical analysis in mHC paper §3.1.
mHC maintains Amax Gain ≈ 1.0 throughout training via doubly
stochastic constraint (Sinkhorn-Knopp).

## Results
| Variant | Final Val PPL | Amax Fwd Max | Amax Bwd Max | Peak VRAM |
|---------|:---:|:---:|:---:|:---:|
| baseline | XX.X | 1.00 | 1.00 | XX GB |
| hc       | XX.X | XX.X | XX.X | XX GB |
| mhc      | XX.X | 1.XX | 1.XX | XX GB |

## Figures
![Amax Forward](figures/amax_fwd_curve.png)
![Amax Backward](figures/amax_bwd_curve.png)
![Val Loss](figures/val_loss_curve.png)
```

**Zip để archive/nộp:**
```bash
cd reports
zip -r qwen-4090-full.zip qwen-4090-full/
```

---

## ⏱️ Timeline ước lượng

| Phase | Thời gian | Có thể song song? |
|-------|-----------|:---:|
| 1. Setup | 20 phút | ❌ |
| 2. Train baseline | 2 giờ | ✅ (trên GPU khác) |
| 2. Train HC | 2.5 giờ | ✅ |
| 2. Train mHC | 3 giờ | ✅ |
| 3. Summarize | 30 giây | ❌ |
| 4. Plot | 30 giây | ❌ |
| 5. Report | 30 phút viết | ❌ |
| **Tổng (tuần tự)** | **~8-9 giờ** | — |

💡 **Tip:** Nếu có 3 GPU 4090, train 3 variants song song → tổng chỉ ~3.5 giờ.

---

## 🎯 Tóm tắt 1 dòng

> **Setup 1 lần → Bật bash script đi ngủ → Sáng dậy có sẵn 6 figures + report để nộp.**

Bạn muốn tôi:
1. ✍️ Viết chi tiết file `run_qwen_4090.sh` (tôi đã phác thảo ở message trước, có thể finalize)?
2. 🔧 Thêm cơ chế **resume từ checkpoint** (nếu train được 3000 iters rồi crash, load ckpt.pt chạy tiếp)?
3. 📊 Thêm **inference benchmark** sau khi train xong (đo tokens/sec, so sánh overhead)?
