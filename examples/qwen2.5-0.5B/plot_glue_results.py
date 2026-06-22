# File: examples/qwen2.5-0.5B/plot_glue_results.py
"""Plot GLUE fine-tuning comparison across Baseline / HC / mHC.

Reads metrics.json files and generates comparison plots for SST-2 and/or MRPC.

Usage:
    # Plot SST-2 results:
    python plot_glue_results.py --task sst2 \
        --runs baseline=out-qwen-baseline-sst2-probe \
               hc=out-qwen-hc-sst2-probe \
               mhc=out-qwen-mhc-sst2-probe

    # Plot MRPC results:
    python plot_glue_results.py --task mrpc \
        --runs baseline=out-qwen-baseline-mrpc-probe \
               hc=out-qwen-hc-mrpc-probe \
               mhc=out-qwen-mhc-mrpc-probe

    # Plot both tasks as a combined report:
    python plot_glue_results.py --task both \
        --sst2-runs baseline=out-qwen-baseline-sst2-probe \
                     hc=out-qwen-hc-sst2-probe \
                     mhc=out-qwen-mhc-sst2-probe \
        --mrpc-runs baseline=out-qwen-baseline-mrpc-probe \
                    hc=out-qwen-hc-mrpc-probe \
                    mhc=out-qwen-mhc-mrpc-probe \
        --output-dir reports/glue-probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#4C78A8",
    "hc": "#F58518",
    "mhc": "#54A24B",
}

LABELS = {
    "baseline": "Baseline",
    "hc": "HC",
    "mhc": "mHC",
}

MARKERS = {
    "baseline": "s",
    "hc": "^",
    "mhc": "o",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_runs(run_args: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Invalid run spec (expected name=path): {item}")
        name, path = item.split("=", 1)
        runs[name.strip()] = Path(path).expanduser()
    return runs


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    with open(metrics_path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Per-task plots
# ---------------------------------------------------------------------------
def plot_train_loss(
    all_metrics: Dict[str, Dict], task_name: str, output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in all_metrics.items():
        epochs = [e["epoch"] for e in metrics["epochs"]]
        losses = [e["train_loss"] for e in metrics["epochs"]]
        ax.plot(epochs, losses, label=LABELS.get(name, name),
                color=COLORS.get(name), marker=MARKERS.get(name, "o"),
                markerfacecolor="white", markeredgewidth=2)
    ax.set_title(f"{task_name.upper()}: Training Loss", fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.9)
    path = output_dir / f"{task_name}_train_loss.png"
    _save_fig(fig, path)
    return path


def plot_metric_curve(
    all_metrics: Dict[str, Dict], task_name: str, metric: str,
    output_dir: Path,
) -> Path:
    """Plot a metric curve (accuracy or f1) per epoch."""
    metric_pct = f"{metric}_pct"
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in all_metrics.items():
        epochs = [e["epoch"] for e in metrics["epochs"]]
        values = [e[metric_pct] for e in metrics["epochs"]]
        ax.plot(epochs, values, label=LABELS.get(name, name),
                color=COLORS.get(name), marker=MARKERS.get(name, "o"),
                markerfacecolor="white", markeredgewidth=2)
    label_map = {"accuracy": "Accuracy", "f1": "F1-Score"}
    ax.set_title(f"{task_name.upper()}: Validation {label_map.get(metric, metric)}",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{label_map.get(metric, metric)} (%)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.9)
    path = output_dir / f"{task_name}_val_{metric}.png"
    _save_fig(fig, path)
    return path


def plot_best_bar(
    all_metrics: Dict[str, Dict], task_name: str, metric: str,
    output_dir: Path,
) -> Path:
    """Bar chart of best metric per method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(all_metrics.keys())
    best_key = f"best_{metric}_pct"
    best_vals = [m.get(best_key, 0) for m in all_metrics.values()]
    colors = [COLORS.get(n, "#999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(labels, best_vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, best_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    label_map = {"accuracy": "Accuracy", "f1": "F1-Score"}
    ax.set_title(f"{task_name.upper()}: Best {label_map.get(metric, metric)}",
                 fontweight="bold", pad=12)
    ax.set_ylabel(f"{label_map.get(metric, metric)} (%)")
    min_v, max_v = min(best_vals), max(best_vals)
    margin = max(2.0, (max_v - min_v) * 0.3)
    ax.set_ylim(max(0, min_v - margin), min(100, max_v + margin + 1))

    path = output_dir / f"{task_name}_best_{metric}_bar.png"
    _save_fig(fig, path)
    return path


def plot_task_panels(
    all_metrics: Dict[str, Dict], task_name: str, output_dir: Path,
) -> Path:
    """2 or 3-panel figure for a task (Loss + Accuracy [+ F1])."""
    # Determine which metrics exist
    sample_epoch = list(all_metrics.values())[0]["epochs"][0]
    has_f1 = "f1_pct" in sample_epoch
    n_panels = 3 if has_f1 else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 2:
        ax_loss, ax_acc = axes
        ax_f1 = None
    else:
        ax_loss, ax_acc, ax_f1 = axes

    for name, metrics in all_metrics.items():
        epochs = [e["epoch"] for e in metrics["epochs"]]
        losses = [e["train_loss"] for e in metrics["epochs"]]
        accs = [e["accuracy_pct"] for e in metrics["epochs"]]
        lbl = LABELS.get(name, name)
        clr = COLORS.get(name)
        mkr = MARKERS.get(name, "o")
        kw = dict(label=lbl, color=clr, marker=mkr,
                  markerfacecolor="white", markeredgewidth=2)

        ax_loss.plot(epochs, losses, **kw)
        ax_acc.plot(epochs, accs, **kw)
        if has_f1 and ax_f1 is not None:
            f1s = [e["f1_pct"] for e in metrics["epochs"]]
            ax_f1.plot(epochs, f1s, **kw)

    ax_loss.set_title("Training Loss", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_loss.legend(framealpha=0.9)

    ax_acc.set_title("Validation Accuracy", fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_acc.legend(framealpha=0.9)

    if ax_f1 is not None:
        ax_f1.set_title("Validation F1-Score", fontweight="bold")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("F1 (%)")
        ax_f1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_f1.legend(framealpha=0.9)

    fig.suptitle(f"{task_name.upper()}: Baseline vs HC vs mHC",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = output_dir / f"{task_name}_combined_panel.png"
    _save_fig(fig, path)
    return path


# ---------------------------------------------------------------------------
# Cross-task summary bar chart
# ---------------------------------------------------------------------------
def plot_cross_task_summary(
    sst2_metrics: Dict[str, Dict],
    mrpc_metrics: Dict[str, Dict],
    output_dir: Path,
) -> Path:
    """Grouped bar chart: 3 methods × 2 tasks (SST-2 Accuracy, MRPC F1)."""
    methods = ["baseline", "hc", "mhc"]
    # Filter to available methods
    methods = [m for m in methods if m in sst2_metrics or m in mrpc_metrics]

    sst2_accs = [sst2_metrics.get(m, {}).get("best_accuracy_pct", 0) for m in methods]
    mrpc_f1s = [mrpc_metrics.get(m, {}).get("best_f1_pct", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, sst2_accs, width, label="SST-2 (Accuracy)",
                   color="#6BAED6", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, mrpc_f1s, width, label="MRPC (F1-Score)",
                   color="#74C476", edgecolor="white", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)

    ax.set_title("GLUE Benchmark: Linear Probe Results",
                 fontweight="bold", fontsize=16, pad=12)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods])
    ax.legend(framealpha=0.9, fontsize=12)

    all_vals = sst2_accs + mrpc_f1s
    min_v, max_v = min(all_vals), max(all_vals)
    margin = max(3.0, (max_v - min_v) * 0.3)
    ax.set_ylim(max(0, min_v - margin), min(100, max_v + margin + 1))

    path = output_dir / "glue_cross_task_summary.png"
    _save_fig(fig, path)
    return path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(
    all_metrics: Dict[str, Dict], task_name: str,
) -> None:
    sample = list(all_metrics.values())[0]
    has_f1 = "best_f1_pct" in sample
    header = f"  {'Method':<12} {'Accuracy':>12}"
    if has_f1:
        header += f" {'F1':>12}"
    print(f"\n  --- {task_name.upper()} ---")
    print(header)
    print(f"  {'-'*36}")
    for name, m in all_metrics.items():
        acc = m.get("best_accuracy_pct", "N/A")
        line = f"  {LABELS.get(name, name):<12} {acc:>11}%"
        if has_f1:
            f1 = m.get("best_f1_pct", "N/A")
            line += f" {f1:>11}%"
        print(line)


# ---------------------------------------------------------------------------
# Generate per-task plots
# ---------------------------------------------------------------------------
def generate_task_plots(
    all_metrics: Dict[str, Dict], task_name: str, output_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    outputs.append(plot_train_loss(all_metrics, task_name, output_dir))
    outputs.append(plot_metric_curve(all_metrics, task_name, "accuracy", output_dir))

    # F1 only for tasks that have it
    sample = list(all_metrics.values())[0]["epochs"][0]
    if "f1_pct" in sample:
        outputs.append(plot_metric_curve(all_metrics, task_name, "f1", output_dir))
        outputs.append(plot_best_bar(all_metrics, task_name, "f1", output_dir))

    outputs.append(plot_best_bar(all_metrics, task_name, "accuracy", output_dir))
    outputs.append(plot_task_panels(all_metrics, task_name, output_dir))
    return outputs


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GLUE fine-tuning results (SST-2 / MRPC / both)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, default="both", choices=["sst2", "mrpc", "both"],
        help="Which task(s) to plot (default: both)",
    )
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run specs for single-task mode: name=dir")
    parser.add_argument("--sst2-runs", nargs="+", default=None,
                        help="Run specs for SST-2 (used with --task both)")
    parser.add_argument("--mrpc-runs", nargs="+", default=None,
                        help="Run specs for MRPC (used with --task both)")
    parser.add_argument("--output-dir", type=str, default="reports/glue")
    args = parser.parse_args()

    _apply_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []

    if args.task in ("sst2", "mrpc"):
        # Single task mode
        if not args.runs:
            parser.error(f"--runs is required for --task {args.task}")
        run_paths = _parse_runs(args.runs)
        all_metrics = {n: _load_metrics(p) for n, p in run_paths.items()}
        print_summary_table(all_metrics, args.task)
        outputs.extend(generate_task_plots(all_metrics, args.task, output_dir))

    else:
        # Both tasks
        sst2_metrics = {}
        mrpc_metrics = {}

        if args.sst2_runs:
            for n, p in _parse_runs(args.sst2_runs).items():
                sst2_metrics[n] = _load_metrics(p)
        if args.mrpc_runs:
            for n, p in _parse_runs(args.mrpc_runs).items():
                mrpc_metrics[n] = _load_metrics(p)

        if sst2_metrics:
            print_summary_table(sst2_metrics, "sst2")
            outputs.extend(generate_task_plots(sst2_metrics, "sst2", output_dir))
        if mrpc_metrics:
            print_summary_table(mrpc_metrics, "mrpc")
            outputs.extend(generate_task_plots(mrpc_metrics, "mrpc", output_dir))
        if sst2_metrics and mrpc_metrics:
            outputs.append(
                plot_cross_task_summary(sst2_metrics, mrpc_metrics, output_dir)
            )

    print(f"\n  ✅ Done! {len(outputs)} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
