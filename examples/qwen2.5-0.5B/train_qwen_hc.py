"""Training script for Qwen2.5-0.5B with HC/mHC variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from qwen_hc_model import build_qwen_hc, forward_with_hc
from metrics_collector import PaperMetricsCollector

DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_MEMMAP_CACHE: Dict[Path, np.memmap] = {}


def _parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_shards(data_dir: Path) -> List[Path]:
    shards = sorted(data_dir.glob("tinystories_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No tinystories_*.bin shards found in {data_dir}")
    return shards


def _get_memmap(path: Path) -> np.memmap:
    if path not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[path] = np.memmap(path, dtype=np.uint32, mode="r")
    return _MEMMAP_CACHE[path]


def _sample_sequence(memmap_arr: np.memmap, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    max_start = memmap_arr.shape[0] - block_size - 1
    if max_start <= 0:
        raise ValueError("Shard is too small for the requested block size.")
    start = random.randint(0, max_start)
    seq = np.array(memmap_arr[start : start + block_size + 1], dtype=np.int64)
    return seq[:-1], seq[1:]


def load_fineweb_batch(
    data_dir: str,
    split: str,
    block_size: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a random batch of FineWeb tokens from memmapped shards."""
    data_path = Path(data_dir)
    shards = _list_shards(data_path)

    if split == "train":
        shard_choices = shards[:-1] if len(shards) > 1 else shards
    elif split == "val":
        shard_choices = shards[-1:]
    else:
        raise ValueError(f"Unknown split: {split}")

    x_list = []
    y_list = []
    for _ in range(batch_size):
        shard = random.choice(shard_choices)
        memmap_arr = _get_memmap(shard)
        x_np, y_np = _sample_sequence(memmap_arr, block_size)
        x_list.append(torch.from_numpy(x_np))
        y_list.append(torch.from_numpy(y_np))

    x = torch.stack(x_list).to(device=device, non_blocking=True)
    y = torch.stack(y_list).to(device=device, non_blocking=True)
    return x, y


def _get_autocast_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device == "cpu" and dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=dtype)
    return nullcontext()


def _get_lr(iter_num: int, warmup_iters: int, max_iters: int, lr_max: float, lr_min: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if iter_num < warmup_iters:
        # Linear warmup
        return lr_max * (iter_num + 1) / warmup_iters
    if iter_num >= max_iters:
        return lr_min
    # Cosine decay
    progress = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def _evaluate(
    model: torch.nn.Module,
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []

    autocast_ctx = _get_autocast_context(device, dtype)
    
    n_streams = model.hc_blocks[0].num_residual_streams if hasattr(model, "hc_blocks") and model.hc_blocks else 4
    collector = PaperMetricsCollector(model, n_streams)
    metrics = {}
    
    with torch.no_grad():
        for i in range(eval_iters):
            x, y = load_fineweb_batch(
                data_dir=data_dir,
                split="val",
                block_size=block_size,
                batch_size=batch_size,
                device=device,
            )
            
            if i == 0:
                collector.enable_collection()
                
            with autocast_ctx:
                loss = forward_with_hc(model, x, y)
                
            if i == 0:
                metrics = collector.compute_metrics()
                collector.disable_collection()
                
            losses.append(loss.item())

    val_loss = float(np.mean(losses)) if losses else float("nan")
    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
    
    metrics["val_loss"] = val_loss
    metrics["val_ppl"] = val_ppl

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-0.5B HC/mHC")
    parser.add_argument("--method", required=True, choices=["baseline", "hc", "mhc"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--sinkhorn-tmax", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--warmup-iters", type=int, default=0,
                        help="Number of warmup steps for LR scheduler (0=no warmup)")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", type=_parse_bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=0.0,
                        help="Minimum LR at end of cosine decay (default: lr/10)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--pretrained", type=_parse_bool, default=False, help="Initialize with pretrained weights for finetuning")
    parser.add_argument("--num-fracs", type=int, default=1, help="Number of fractions for Frac-Connections")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU.")
        device = "cpu"

    dtype = DTYPE_MAP[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LR schedule config
    lr_max = args.lr
    lr_min = args.lr_min if args.lr_min > 0 else lr_max / 10.0
    warmup_iters = args.warmup_iters

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    total_tokens = tokens_per_step * args.max_iters

    print(f"📊 Training config:")
    print(f"   Method:          {args.method}")
    print(f"   Max iters:       {args.max_iters}")
    print(f"   Warmup iters:    {warmup_iters}")
    print(f"   Tokens/step:     {tokens_per_step:,}")
    print(f"   Total tokens:    {total_tokens:,} ({total_tokens/1e6:.0f}M)")
    print(f"   EBS:             {args.batch_size * args.grad_accum}")
    print(f"   LR:              {lr_max} -> {lr_min} (cosine)")
    if args.method != "baseline":
        print(f"   Num fracs:       {args.num_fracs}")

    _set_seed(1337)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    print("📥 Initializing model...")
    model, _ = build_qwen_hc(
        method=args.method,
        n_streams=args.n_streams,
        sinkhorn_tmax=args.sinkhorn_tmax,
        pretrained=args.pretrained,
        dtype=dtype,
        device=device,
        num_fracs=args.num_fracs,
    )

    if args.compile and hasattr(torch, "compile"):
        print("⚡ torch.compile enabled")
        model = torch.compile(model)
    elif args.compile:
        print("⚠️ torch.compile unavailable, continuing without compile")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr_max,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_scaler = device.startswith("cuda") and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_ctx = _get_autocast_context(device, dtype)

    best_val_loss = float("inf")
    last_val_metrics = {}
    history_rows: List[Dict[str, float]] = []

    start_time = time.time()
    print("🏁 Starting training...")

    for iter_num in range(1, args.max_iters + 1):
        # Update learning rate (cosine schedule with warmup)
        current_lr = _get_lr(iter_num - 1, warmup_iters, args.max_iters, lr_max, lr_min)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        for micro in range(args.grad_accum):
            x, y = load_fineweb_batch(
                data_dir=args.data_dir,
                split="train",
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
            )
            with autocast_ctx:
                loss = forward_with_hc(model, x, y)
            train_loss_sum += loss.item()

            loss = loss / args.grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
            
        hc_params = [p for n, p in model.named_parameters() if p.requires_grad and 'hc_blocks' in n]
        base_params = [p for n, p in model.named_parameters() if p.requires_grad and 'hc_blocks' not in n]
        
        pre_clip_hc = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in hc_params if p.grad is not None])) if hc_params else torch.tensor(0.0)
        pre_clip_base = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in base_params if p.grad is not None])) if base_params else torch.tensor(0.0)
        
        pre_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params if p.grad is not None]))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_loss = train_loss_sum / args.grad_accum

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            print("✅ Running eval...")
            last_val_metrics = _evaluate(
                model=model,
                data_dir=args.data_dir,
                block_size=args.block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                device=device,
                dtype=dtype,
            )

            if last_val_metrics["val_loss"] < best_val_loss:
                best_val_loss = last_val_metrics["val_loss"]
                ckpt = {
                    "model_state": model.state_dict(),
                    "iter_num": iter_num,
                    "val_loss": best_val_loss,
                }
                torch.save(ckpt, out_dir / "ckpt.pt")
                print("✅ Saved new best checkpoint")

        if iter_num % 10 == 0 or iter_num == 1:
            vram_gb = 0.0
            if device.startswith("cuda"):
                vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            msg = (
                f"⚡ [{iter_num}] loss={train_loss:.4f} "
                f"grad_pre={float(pre_clip_norm):.4f} grad_post={float(grad_norm):.4f} "
                f"lr={current_lr:.2e} vram={vram_gb:.2f}GB"
            )
            print(msg)

            history_rows.append(
                {
                    "iter": float(iter_num),
                    "train_loss": float(train_loss),
                    "val_loss": float(last_val_metrics.get("val_loss", float("nan"))),
                    "grad_norm_post_clip": float(grad_norm),
                    "grad_norm_pre_clip": float(pre_clip_norm),
                    "grad_norm_hc": float(pre_clip_hc),
                    "grad_norm_base": float(pre_clip_base),
                    "lr": float(current_lr),
                    "amax_fwd_max": float(last_val_metrics.get("amax_fwd_max", 1.0)),
                    "amax_bwd_max": float(last_val_metrics.get("amax_bwd_max", 1.0)),
                    "h_res_row_sum_max_dev": float(last_val_metrics.get("h_res_row_sum_max_dev", 0.0)),
                    "avg_attn_entropy": float(last_val_metrics.get("avg_attn_entropy", 0.0)),
                    "residual_norm_final": float(last_val_metrics.get("residual_norm_final", 0.0)),
                }
            )

    if "val_loss" not in last_val_metrics or math.isnan(last_val_metrics["val_loss"]):
        last_val_metrics = _evaluate(
            model=model,
            data_dir=args.data_dir,
            block_size=args.block_size,
            batch_size=args.batch_size,
            eval_iters=args.eval_iters,
            device=device,
            dtype=dtype,
        )

    peak_vram_gb = 0.0
    if device.startswith("cuda"):
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    elapsed_s = time.time() - start_time

    summary = {
        "ok": True,
        "method": args.method,
        "iter_num": args.max_iters,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_loss),
        "final_val_loss": float(last_val_metrics.get("val_loss", float("nan"))),
        "final_val_ppl": float(last_val_metrics.get("val_ppl", float("nan"))),
        "peak_vram_gb": float(peak_vram_gb),
        "elapsed_s": float(elapsed_s),
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "warmup_iters": warmup_iters,
        "lr_max": lr_max,
        "lr_min": lr_min,
    }
    summary.update({k: float(v) for k, v in last_val_metrics.items() if isinstance(v, (int, float))})

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("✅ Wrote summary.json")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "train_loss",
                "val_loss",
                "grad_norm_post_clip",
                "grad_norm_pre_clip",
                "grad_norm_hc",
                "grad_norm_base",
                "lr",
                "amax_fwd_max",
                "amax_bwd_max",
                "h_res_row_sum_max_dev",
                "avg_attn_entropy",
                "residual_norm_final",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)
    print("✅ Wrote history.csv")

    print("✅ Done")


if __name__ == "__main__":
    main()
