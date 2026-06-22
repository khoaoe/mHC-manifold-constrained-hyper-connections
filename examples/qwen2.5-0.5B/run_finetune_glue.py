# File: examples/qwen2.5-0.5B/run_finetune_glue.py
"""Fine-tune (or Linear Probe) a Qwen2.5 HC/mHC model on GLUE tasks.

Supports: SST-2 (sentiment) and MRPC (paraphrase detection).
Can run one task or both tasks in sequence with a single command.

Usage:
    # Run both tasks:
    python run_finetune_glue.py --method mhc --ckpt out-qwen-mhc/ckpt.pt --tasks sst2 mrpc --linear-probe

    # Run a single task:
    python run_finetune_glue.py --method mhc --ckpt out-qwen-mhc/ckpt.pt --tasks sst2 --linear-probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Ensure local modules are importable when run as script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qwen_hc_model import build_qwen_hc
from qwen_classification import QwenHCForSequenceClassification


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------
TASK_CONFIG = {
    "sst2": {
        "glue_name": "sst2",
        "num_labels": 2,
        "input_type": "single",       # single sentence
        "text_cols": ["sentence"],
        "metrics": ["accuracy"],
        "primary_metric": "accuracy",  # metric used for best model selection
        "description": "SST-2 (Sentiment Analysis)",
    },
    "mrpc": {
        "glue_name": "mrpc",
        "num_labels": 2,
        "input_type": "pair",          # sentence pair
        "text_cols": ["sentence1", "sentence2"],
        "metrics": ["accuracy", "f1"],
        "primary_metric": "f1",        # F1 is the standard metric for MRPC
        "description": "MRPC (Paraphrase Detection)",
    },
}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(
    task_name: str, tokenizer, max_length: int, batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Load and tokenize a GLUE task, return (train_dl, val_dl)."""
    cfg = TASK_CONFIG[task_name]
    print(f"\n📦 Preparing {cfg['description']} (max_length={max_length})...")
    dataset = load_dataset("glue", cfg["glue_name"])

    if cfg["input_type"] == "single":
        col = cfg["text_cols"][0]

        def preprocess_fn(examples):
            return tokenizer(
                examples[col],
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
    else:
        col1, col2 = cfg["text_cols"]

        def preprocess_fn(examples):
            return tokenizer(
                examples[col1],
                examples[col2],
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

    tokenized = dataset.map(preprocess_fn, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_dl = DataLoader(tokenized["train"], shuffle=True, batch_size=batch_size)
    val_dl = DataLoader(tokenized["validation"], shuffle=False, batch_size=batch_size)
    print(f"  Train samples: {len(tokenized['train'])}")
    print(f"  Val samples:   {len(tokenized['validation'])}")
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module, val_dl: DataLoader, device: str, task_name: str,
) -> Dict[str, float]:
    """Run evaluation and return a dict of metric_name -> value."""
    cfg = TASK_CONFIG[task_name]
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="[Eval]", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    results: Dict[str, float] = {}
    if "accuracy" in cfg["metrics"]:
        results["accuracy"] = accuracy_score(all_labels, all_preds)
        results["accuracy_pct"] = round(results["accuracy"] * 100, 2)
    if "f1" in cfg["metrics"]:
        results["f1"] = f1_score(all_labels, all_preds)
        results["f1_pct"] = round(results["f1"] * 100, 2)

    return results


# ---------------------------------------------------------------------------
# Single task training loop
# ---------------------------------------------------------------------------
def run_single_task(
    task_name: str,
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    device: str,
    total_params: int,
    trainable_params: int,
) -> Dict:
    """Train & evaluate on a single GLUE task. Returns the metrics history."""
    cfg = TASK_CONFIG[task_name]
    primary_metric = cfg["primary_metric"]

    print(f"\n{'='*60}")
    print(f"  📝 Task: {cfg['description']}")
    print(f"{'='*60}")

    # Prepare data
    train_dl, val_dl = prepare_dataset(
        task_name, tokenizer, args.max_length, args.batch_size,
    )

    # Re-initialize classification head for each task (fresh linear layer)
    hidden_size = args.hidden_size
    num_labels = cfg["num_labels"]
    model.score = nn.Linear(hidden_size, num_labels, bias=False).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    # Make sure the new head is trainable
    for param in model.score.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )

    probe_suffix = "-probe" if args.linear_probe else ""
    save_dir = args.save_dir or f"out-qwen-{args.method}-{task_name}{probe_suffix}"
    os.makedirs(save_dir, exist_ok=True)

    # Metrics history
    history = {
        "task": task_name,
        "task_description": cfg["description"],
        "method": args.method,
        "mode": "linear_probe" if args.linear_probe else "full_finetune",
        "model_name": args.model_name,
        "checkpoint": args.ckpt,
        "hidden_size": hidden_size,
        "num_labels": num_labels,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "primary_metric": primary_metric,
        "epochs": [],
    }

    best_score = 0.0

    print(f"\n🚀 Training for {args.epochs} epochs (lr={args.lr}, bs={args.batch_size})...\n")

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dl)

        # --- Eval ---
        eval_results = evaluate(model, val_dl, device, task_name)

        # Print results
        metrics_str = " | ".join(
            f"{k.upper()} = {v*100:.2f}%" for k, v in eval_results.items()
            if not k.endswith("_pct")
        )
        print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | {metrics_str}")

        # Record metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            **{k: round(v, 6) if not k.endswith("_pct") else v
               for k, v in eval_results.items()},
        }
        history["epochs"].append(epoch_metrics)

        # Save metrics after every epoch
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best model (by primary metric)
        current_score = eval_results[primary_metric]
        if current_score > best_score:
            best_score = current_score
            save_path = os.path.join(save_dir, "best_clf_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  💾 New best ({primary_metric})! Saved to {save_path}")

    # Final summary
    history[f"best_{primary_metric}"] = round(best_score, 6)
    history[f"best_{primary_metric}_pct"] = round(best_score * 100, 2)
    # Also store all best values for convenience
    for metric_name in cfg["metrics"]:
        best_val = max(e[metric_name] for e in history["epochs"])
        history[f"best_{metric_name}"] = round(best_val, 6)
        history[f"best_{metric_name}_pct"] = round(best_val * 100, 2)

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ {cfg['description']} complete!")
    print(f"  📊 Best {primary_metric.upper()}: {best_score*100:.2f}%")
    print(f"  📊 Metrics saved to: {metrics_path}")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune / Linear Probe Qwen HC/mHC on GLUE tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Linear Probe on both SST-2 and MRPC:
  python run_finetune_glue.py --method mhc --ckpt out-qwen-mhc/ckpt.pt --tasks sst2 mrpc --linear-probe

  # Full fine-tune on SST-2 only:
  python run_finetune_glue.py --method mhc --ckpt out-qwen-mhc/ckpt.pt --tasks sst2
        """,
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["sst2", "mrpc"],
        choices=list(TASK_CONFIG.keys()),
        help="GLUE tasks to run (default: sst2 mrpc)",
    )
    parser.add_argument(
        "--method", type=str, default="mhc", choices=["baseline", "hc", "mhc"],
        help="Which method to use (default: mhc)",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to pre-trained checkpoint (e.g. out-qwen-mhc/ckpt.pt)",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--linear-probe", action="store_true",
        help="Freeze backbone, only train classification head.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    mode_str = "Linear Probe" if args.linear_probe else "Full Fine-tune"
    tasks_str = " + ".join(t.upper() for t in args.tasks)
    print(f"\n{'='*60}")
    print(f"  {mode_str}: Qwen2.5 [{args.method.upper()}]")
    print(f"  Tasks: {tasks_str}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Build model + load checkpoint (once, shared across tasks)
    # ------------------------------------------------------------------
    print(f"\n📥 Building {args.method} model...")
    causal_model, tokenizer = build_qwen_hc(
        model_name=args.model_name,
        method=args.method,
        pretrained=False,
        dtype=dtype,
        device=device,
    )

    if args.ckpt and Path(args.ckpt).exists():
        print(f"📥 Loading checkpoint from {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location=device)

        # Extract model_state if it's wrapped
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        # Handle torch.compile prefix
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            cleaned_state_dict[new_key] = v

        missing, unexpected = causal_model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"  ✅ Loaded weights. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    elif args.ckpt:
        print(f"⚠️  Checkpoint not found: {args.ckpt}. Starting from random init.")
    else:
        print("📌 No checkpoint specified. Training from random init.")

    # Wrap with classification head (num_labels will be reset per task)
    model = QwenHCForSequenceClassification(
        causal_model, hidden_size=args.hidden_size, num_labels=2,
    )

    # Freeze backbone if linear probing
    if args.linear_probe:
        print("\n🧊 Linear Probing: Freezing entire backbone...")
        for param in model.causal_model.parameters():
            param.requires_grad = False
        print("  ✅ Backbone frozen.")

    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params/1e6:.2f}M")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M")

    # ------------------------------------------------------------------
    # 2. Run each task sequentially
    # ------------------------------------------------------------------
    all_results = {}
    for task_name in args.tasks:
        history = run_single_task(
            task_name=task_name,
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            total_params=total_params,
            trainable_params=trainable_params,
        )
        all_results[task_name] = history

    # ------------------------------------------------------------------
    # 3. Final summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print(f"  📊 FINAL RESULTS: {args.method.upper()} ({mode_str})")
    print(f"{'='*60}")
    print(f"  {'Task':<12} {'Accuracy':>12} {'F1':>12}")
    print(f"  {'-'*36}")
    for task_name, hist in all_results.items():
        acc = hist.get("best_accuracy_pct", "N/A")
        f1 = hist.get("best_f1_pct", "N/A")
        acc_str = f"{acc}%" if isinstance(acc, (int, float)) else acc
        f1_str = f"{f1}%" if isinstance(f1, (int, float)) else f1
        print(f"  {task_name.upper():<12} {acc_str:>12} {f1_str:>12}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
