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

from qwen_hc_model import build_qwen_hc, compute_amax_gain, forward_with_hc

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
    shards = sorted(data_dir.glob("fineweb_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No fineweb_*.bin shards found in {data_dir}")
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


def _evaluate(
    model: torch.nn.Module,
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    dtype: torch.dtype,
    method: str,
    amax_eval: bool,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    losses: List[float] = []
    amax_fwd_vals: List[float] = []
    amax_bwd_vals: List[float] = []

    autocast_ctx = _get_autocast_context(device, dtype)
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = load_fineweb_batch(
                data_dir=data_dir,
                split="val",
                block_size=block_size,
                batch_size=batch_size,
                device=device,
            )
            with autocast_ctx:
                loss, h_res = forward_with_hc(model, x, y)
            losses.append(loss.item())
            if method != "baseline" and amax_eval:
                fwd, bwd = compute_amax_gain(h_res)
                amax_fwd_vals.append(fwd)
                amax_bwd_vals.append(bwd)

    val_loss = float(np.mean(losses)) if losses else float("nan")
    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

    if method == "baseline" or not amax_eval or not amax_fwd_vals:
        amax_stats = {
            "amax_fwd_final": 1.0,
            "amax_fwd_max": 1.0,
            "amax_fwd_mean": 1.0,
            "amax_bwd_final": 1.0,
            "amax_bwd_max": 1.0,
            "amax_bwd_mean": 1.0,
        }
        return val_loss, val_ppl, amax_stats

    amax_stats = {
        "amax_fwd_final": float(amax_fwd_vals[-1]),
        "amax_fwd_max": float(max(amax_fwd_vals)),
        "amax_fwd_mean": float(sum(amax_fwd_vals) / len(amax_fwd_vals)),
        "amax_bwd_final": float(amax_bwd_vals[-1]),
        "amax_bwd_max": float(max(amax_bwd_vals)),
        "amax_bwd_mean": float(sum(amax_bwd_vals) / len(amax_bwd_vals)),
    }
    return val_loss, val_ppl, amax_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-0.5B HC/mHC")
    parser.add_argument("--method", required=True, choices=["baseline", "hc", "mhc"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--sinkhorn-tmax", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=5000)
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
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--amax-log-interval", type=int, default=100)
    parser.add_argument("--amax-eval", type=_parse_bool, default=True)
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
        dtype=dtype,
        device=device,
    )

    if args.compile and hasattr(torch, "compile"):
        print("⚡ torch.compile enabled")
        model = torch.compile(model)
    elif args.compile:
        print("⚠️ torch.compile unavailable, continuing without compile")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_scaler = device.startswith("cuda") and dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_ctx = _get_autocast_context(device, dtype)

    best_val_loss = float("inf")
    last_val_loss = float("nan")
    last_val_ppl = float("nan")
    history_rows: List[Dict[str, float]] = []
    amax_fwd_vals: List[float] = []
    amax_bwd_vals: List[float] = []
    last_amax_fwd = 1.0
    last_amax_bwd = 1.0

    start_time = time.time()
    print("🏁 Starting training...")

    for iter_num in range(1, args.max_iters + 1):
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
                loss, h_res = forward_with_hc(model, x, y)
            train_loss_sum += loss.item()

            if (
                args.method != "baseline"
                and iter_num % args.amax_log_interval == 0
                and micro == 0
            ):
                last_amax_fwd, last_amax_bwd = compute_amax_gain(h_res)
                amax_fwd_vals.append(last_amax_fwd)
                amax_bwd_vals.append(last_amax_bwd)

            loss = loss / args.grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_loss = train_loss_sum / args.grad_accum

        val_loss = None
        val_ppl = None
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            print("✅ Running eval...")
            val_loss, val_ppl, eval_amax = _evaluate(
                model=model,
                data_dir=args.data_dir,
                block_size=args.block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                device=device,
                dtype=dtype,
                method=args.method,
                amax_eval=args.amax_eval,
            )
            last_val_loss = val_loss
            last_val_ppl = val_ppl

            if args.method != "baseline" and args.amax_eval:
                last_amax_fwd = eval_amax["amax_fwd_mean"]
                last_amax_bwd = eval_amax["amax_bwd_mean"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "model_state": model.state_dict(),
                    "iter_num": iter_num,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, out_dir / "ckpt.pt")
                print("✅ Saved new best checkpoint")

        if iter_num % 10 == 0 or iter_num == 1:
            vram_gb = 0.0
            if device.startswith("cuda"):
                vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            msg = (
                f"⚡ [{iter_num}] loss={train_loss:.4f} "
                f"grad={float(grad_norm):.4f} vram={vram_gb:.2f}GB"
            )
            if args.method != "baseline":
                msg += f" amax_fwd={last_amax_fwd:.3f} amax_bwd={last_amax_bwd:.3f}"
            print(msg)

            history_rows.append(
                {
                    "iter": float(iter_num),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss) if val_loss is not None else float("nan"),
                    "grad_norm": float(grad_norm),
                    "amax_fwd": float(last_amax_fwd),
                    "amax_bwd": float(last_amax_bwd),
                }
            )

    if math.isnan(last_val_loss):
        val_loss, val_ppl, _ = _evaluate(
            model=model,
            data_dir=args.data_dir,
            block_size=args.block_size,
            batch_size=args.batch_size,
            eval_iters=args.eval_iters,
            device=device,
            dtype=dtype,
            method=args.method,
            amax_eval=args.amax_eval,
        )
        last_val_loss = val_loss
        last_val_ppl = val_ppl

    peak_vram_gb = 0.0
    if device.startswith("cuda"):
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    if args.method == "baseline":
        amax_fwd_final = amax_fwd_max = amax_fwd_mean = 1.0
        amax_bwd_final = amax_bwd_max = amax_bwd_mean = 1.0
    else:
        if amax_fwd_vals:
            amax_fwd_final = float(amax_fwd_vals[-1])
            amax_fwd_max = float(max(amax_fwd_vals))
            amax_fwd_mean = float(sum(amax_fwd_vals) / len(amax_fwd_vals))
            amax_bwd_final = float(amax_bwd_vals[-1])
            amax_bwd_max = float(max(amax_bwd_vals))
            amax_bwd_mean = float(sum(amax_bwd_vals) / len(amax_bwd_vals))
        else:
            amax_fwd_final = amax_fwd_max = amax_fwd_mean = 1.0
            amax_bwd_final = amax_bwd_max = amax_bwd_mean = 1.0

    elapsed_s = time.time() - start_time

    summary = {
        "ok": True,
        "method": args.method,
        "iter_num": args.max_iters,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_loss),
        "final_val_loss": float(last_val_loss),
        "final_val_ppl": float(last_val_ppl),
        "amax_fwd_final": float(amax_fwd_final),
        "amax_fwd_max": float(amax_fwd_max),
        "amax_fwd_mean": float(amax_fwd_mean),
        "amax_bwd_final": float(amax_bwd_final),
        "amax_bwd_max": float(amax_bwd_max),
        "amax_bwd_mean": float(amax_bwd_mean),
        "peak_vram_gb": float(peak_vram_gb),
        "elapsed_s": float(elapsed_s),
    }

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
                "grad_norm",
                "amax_fwd",
                "amax_bwd",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)
    print("✅ Wrote history.csv")

    print("✅ Done")


if __name__ == "__main__":
    main()
