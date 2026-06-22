# File: examples/qwen2.5-0.5B/run_finetune_sst2.py
"""Fine-tune a pre-trained Qwen2.5 HC/mHC model on SST-2 (sentiment classification).

Usage:
    python run_finetune_sst2.py --method mhc --ckpt out-qwen-mhc/ckpt.pt
    python run_finetune_sst2.py --method hc  --ckpt out-qwen-hc/ckpt.pt
    python run_finetune_sst2.py --method baseline --ckpt out-qwen-baseline/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Ensure local modules are importable when run as script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qwen_hc_model import build_qwen_hc
from qwen_classification import QwenHCForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen HC/mHC on SST-2")
    parser.add_argument(
        "--method", type=str, default="mhc", choices=["baseline", "hc", "mhc"],
        help="Which method to use (default: mhc)",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to pre-trained checkpoint (e.g. out-qwen-mhc/ckpt.pt). "
             "If not given, fine-tunes from random init.",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--linear-probe", action="store_true",
        help="Linear Probing mode: freeze the entire backbone and only train "
             "the classification head. Use this to evaluate representation quality.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ------------------------------------------------------------------
    # 1. Build model + optionally load pre-trained checkpoint
    # ------------------------------------------------------------------
    mode_str = "Linear Probe" if args.linear_probe else "Full Fine-tune"
    print(f"{'='*60}")
    print(f"  {mode_str}: Qwen2.5 [{args.method.upper()}] on SST-2")
    print(f"{'='*60}")

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
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
        missing, unexpected = causal_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠️  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print("  ✅ Checkpoint loaded.")
    elif args.ckpt:
        print(f"⚠️  Checkpoint not found: {args.ckpt}. Starting from random init.")
    else:
        print("📌 No checkpoint specified. Fine-tuning from random init.")

    # Wrap with classification head
    model = QwenHCForSequenceClassification(
        causal_model, hidden_size=args.hidden_size, num_labels=args.num_labels,
    )

    # ------------------------------------------------------------------
    # LINEAR PROBING: Freeze backbone, only train classification head
    # ------------------------------------------------------------------
    if args.linear_probe:
        print("\n🧊 Linear Probing mode: Freezing entire backbone...")
        for param in model.causal_model.parameters():
            param.requires_grad = False
        # Ensure classification head is trainable
        for param in model.score.parameters():
            param.requires_grad = True
        print("  ✅ Backbone frozen. Only classification head (score) will be trained.")

    model.to(device)

    # Fix tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params/1e6:.2f}M")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M")
    if args.linear_probe:
        print(f"  📌 Mode: Linear Probe (only {trainable_params} params trained)")

    # ------------------------------------------------------------------
    # 2. Prepare SST-2 dataset
    # ------------------------------------------------------------------
    print(f"\n📦 Preparing SST-2 dataset (max_length={args.max_length})...")
    dataset = load_dataset("glue", "sst2")

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )

    tokenized_datasets = dataset.map(preprocess_fn, batched=True)
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_dl = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    val_dl = DataLoader(tokenized_datasets["validation"], shuffle=False, batch_size=args.batch_size)
    print(f"  Train samples: {len(tokenized_datasets['train'])}")
    print(f"  Val samples:   {len(tokenized_datasets['validation'])}")

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )

    print(f"\n🚀 Starting fine-tuning for {args.epochs} epochs (lr={args.lr}, bs={args.batch_size})...\n")

    best_acc = 0.0
    probe_suffix = "-probe" if args.linear_probe else ""
    save_dir = args.save_dir or f"out-qwen-{args.method}-sst2{probe_suffix}"
    os.makedirs(save_dir, exist_ok=True)

    # ---- Metrics history for plotting ----
    history = {
        "method": args.method,
        "mode": "linear_probe" if args.linear_probe else "full_finetune",
        "model_name": args.model_name,
        "checkpoint": args.ckpt,
        "hidden_size": args.hidden_size,
        "num_labels": args.num_labels,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": [],
    }

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
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Eval ]", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["logits"], dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(
            f"  Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f} | "
            f"Val Accuracy = {val_acc*100:.2f}%"
        )

        # Record metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_accuracy_pct": round(val_acc * 100, 2),
        }
        history["epochs"].append(epoch_metrics)

        # Save metrics after every epoch (so we don't lose data on crash)
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, "best_clf_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  💾 New best! Saved to {save_path}")

    # Final summary
    history["best_val_accuracy"] = round(best_acc, 6)
    history["best_val_accuracy_pct"] = round(best_acc * 100, 2)
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Fine-tuning complete! Best Val Accuracy: {best_acc*100:.2f}%")
    print(f"  📊 Metrics saved to: {metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
