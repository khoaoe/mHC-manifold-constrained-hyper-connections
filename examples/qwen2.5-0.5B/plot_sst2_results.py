# File: examples/qwen2.5-0.5B/plot_sst2_results.py
"""Plot SST-2 fine-tuning comparison across Baseline / HC / mHC.

Reads metrics.json files produced by run_finetune_sst2.py and generates:
  1. Train Loss curve (per epoch, 3 methods overlaid)
  2. Validation Accuracy curve (per epoch, 3 methods overlaid)
  3. Best Accuracy bar chart (one bar per method)
  4. Combined 2-panel figure (Loss + Accuracy side by side)

Usage:
    python plot_sst2_results.py \
        --runs baseline=out-qwen-baseline-sst2 \
               hc=out-qwen-hc-sst2 \
               mhc=out-qwen-mhc-sst2 \
        --output-dir reports/sst2

    # Or with fewer runs (e.g. only mhc vs baseline):
    python plot_sst2_results.py \
        --runs baseline=out-qwen-baseline-sst2 mhc=out-qwen-mhc-sst2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works headless / SSH / CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Consistent styling (matches plot_qwen_comparison.py)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#4C78A8",
    "hc": "#F58518",
    "mhc": "#54A24B",
}

LABELS = {
    "baseline": "Baseline",
    "hc": "HC (Hyper-Connections)",
    "mhc": "mHC (Manifold-Constrained HC)",
}

MARKERS = {
    "baseline": "s",
    "hc": "^",
    "mhc": "o",
}


def _apply_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
    })


def _parse_runs(run_args: List[str]) -> Dict[str, Path]:
    """Parse CLI args like 'baseline=path/to/dir' into {name: Path}."""
    runs: Dict[str, Path] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Invalid run spec (expected name=path): {item}")
        name, path = item.split("=", 1)
        runs[name.strip()] = Path(path).expanduser()
    return runs


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics.json in {run_dir}. "
            f"Run `run_finetune_sst2.py` first to generate it."
        )
    with open(metrics_path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------
def plot_train_loss(
    all_metrics: Dict[str, Dict], output_dir: Path
) -> Path:
    """Line plot: Train Loss vs Epoch for each method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, metrics in all_metrics.items():
        epochs_data = metrics["epochs"]
        epochs = [e["epoch"] for e in epochs_data]
        losses = [e["train_loss"] for e in epochs_data]

        ax.plot(
            epochs, losses,
            label=LABELS.get(name, name),
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            markerfacecolor="white",
            markeredgewidth=2,
        )

    ax.set_title("SST-2 Fine-tuning: Training Loss", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right", framealpha=0.9)

    path = output_dir / "sst2_train_loss.png"
    _save_fig(fig, path)
    return path


def plot_val_accuracy(
    all_metrics: Dict[str, Dict], output_dir: Path
) -> Path:
    """Line plot: Validation Accuracy vs Epoch for each method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, metrics in all_metrics.items():
        epochs_data = metrics["epochs"]
        epochs = [e["epoch"] for e in epochs_data]
        accs = [e["val_accuracy_pct"] for e in epochs_data]

        ax.plot(
            epochs, accs,
            label=LABELS.get(name, name),
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            markerfacecolor="white",
            markeredgewidth=2,
        )

    ax.set_title("SST-2 Fine-tuning: Validation Accuracy", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="lower right", framealpha=0.9)

    path = output_dir / "sst2_val_accuracy.png"
    _save_fig(fig, path)
    return path


def plot_best_accuracy_bar(
    all_metrics: Dict[str, Dict], output_dir: Path
) -> Path:
    """Bar chart: Best Validation Accuracy per method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(all_metrics.keys())
    best_accs = [
        m.get("best_val_accuracy_pct", max(e["val_accuracy_pct"] for e in m["epochs"]))
        for m in all_metrics.values()
    ]
    colors = [COLORS.get(n, "#999999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(labels, best_accs, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    # Add value labels on top of bars
    for bar, acc in zip(bars, best_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    ax.set_title("SST-2: Best Validation Accuracy", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)")

    # Set y-axis range for better visual comparison
    min_acc = min(best_accs)
    max_acc = max(best_accs)
    margin = max(2.0, (max_acc - min_acc) * 0.3)
    ax.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin + 1))

    path = output_dir / "sst2_best_accuracy_bar.png"
    _save_fig(fig, path)
    return path


def plot_combined_panel(
    all_metrics: Dict[str, Dict], output_dir: Path
) -> Path:
    """2-panel figure: Train Loss (left) + Val Accuracy (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, metrics in all_metrics.items():
        epochs_data = metrics["epochs"]
        epochs = [e["epoch"] for e in epochs_data]
        losses = [e["train_loss"] for e in epochs_data]
        accs = [e["val_accuracy_pct"] for e in epochs_data]
        label = LABELS.get(name, name)
        color = COLORS.get(name, None)
        marker = MARKERS.get(name, "o")

        ax1.plot(epochs, losses, label=label, color=color, marker=marker,
                 markerfacecolor="white", markeredgewidth=2)
        ax2.plot(epochs, accs, label=label, color=color, marker=marker,
                 markerfacecolor="white", markeredgewidth=2)

    ax1.set_title("Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.legend(framealpha=0.9)

    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(framealpha=0.9)

    fig.suptitle("SST-2 Fine-tuning: Baseline vs HC vs mHC",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = output_dir / "sst2_combined_panel.png"
    _save_fig(fig, path)
    return path


def print_summary_table(all_metrics: Dict[str, Dict]) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'='*70}")
    print(f"  SST-2 Fine-tuning Results Summary")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Params (M)':>12} {'Best Acc (%)':>14} {'Final Loss':>12}")
    print(f"  {'-'*50}")

    for name, metrics in all_metrics.items():
        label = LABELS.get(name, name)
        params = metrics.get("total_params", 0) / 1e6
        best_acc = metrics.get(
            "best_val_accuracy_pct",
            max(e["val_accuracy_pct"] for e in metrics["epochs"]),
        )
        final_loss = metrics["epochs"][-1]["train_loss"]
        print(f"  {label:<12} {params:>12.2f} {best_acc:>14.2f} {final_loss:>12.4f}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SST-2 fine-tuning results (Baseline vs HC vs mHC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_sst2_results.py \\
      --runs baseline=out-qwen-baseline-sst2 \\
             hc=out-qwen-hc-sst2 \\
             mhc=out-qwen-mhc-sst2 \\
      --output-dir reports/sst2
        """,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run specs in 'name=dir' format (e.g. mhc=out-qwen-mhc-sst2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/sst2",
        help="Directory to save plots (default: reports/sst2)",
    )
    args = parser.parse_args()

    _apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    run_paths = _parse_runs(args.runs)
    all_metrics: Dict[str, Dict] = {}
    for name, path in run_paths.items():
        print(f"  📥 Loading metrics: {name} <- {path}")
        all_metrics[name] = _load_metrics(path)

    # Print summary table
    print_summary_table(all_metrics)

    # Generate plots
    print("  🎨 Generating plots...")
    outputs: List[Path] = []
    outputs.append(plot_train_loss(all_metrics, output_dir))
    outputs.append(plot_val_accuracy(all_metrics, output_dir))
    outputs.append(plot_best_accuracy_bar(all_metrics, output_dir))
    outputs.append(plot_combined_panel(all_metrics, output_dir))

    print(f"\n  ✅ Done! {len(outputs)} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
