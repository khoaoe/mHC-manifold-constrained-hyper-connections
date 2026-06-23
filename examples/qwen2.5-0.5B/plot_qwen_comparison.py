"""Plot comparison figures for Qwen2.5-0.5B HC/mHC runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLORS = {
    "baseline": "#4C78A8",
    "hc": "#F58518",
    "mhc": "#54A24B",
}


def _parse_runs(run_args: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Invalid run spec: {item}")
        name, path = item.split("=", 1)
        runs[name.strip()] = Path(path).expanduser()
    return runs


def _ema(series: pd.Series, alpha: float) -> pd.Series:
    """Calculate Exponential Moving Average for smoothing."""
    return series.ewm(alpha=alpha, adjust=False).mean()


def _load_history(run_dir: Path) -> pd.DataFrame:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.csv: {history_path}")
    return pd.read_csv(history_path)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_val_loss(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        subset = df.dropna(subset=["val_loss"])
        if subset.empty:
            continue
        ax.plot(
            subset["iter"],
            subset["val_loss"],
            label=name,
            color=COLORS.get(name, None),
            linewidth=2,
            marker='o', # Thêm marker để dễ nhìn các điểm eval
            markersize=4
        )
    ax.set_title("Validation Loss vs Iteration", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    path = output_dir / "val_loss_curve.png"
    _save_fig(fig, path)
    return path


def _plot_train_loss(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        if "train_loss" not in df:
            continue
        # Đã SỬA: Dùng alpha=0.05 để làm mượt thực sự
        ema_series = _ema(df["train_loss"], alpha=0.05) 
        
        # Plot đường mờ (raw) phía sau, đường đậm (smoothed) phía trước
        ax.plot(df["iter"], df["train_loss"], color=COLORS.get(name, None), alpha=0.15)
        ax.plot(df["iter"], ema_series, label=name, color=COLORS.get(name, None), linewidth=2)
        
    ax.set_title("Training Loss vs Iteration (Smoothed)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    path = output_dir / "train_loss_curve.png"
    _save_fig(fig, path)
    return path


def _plot_train_loss_gap(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path | None:
    if "baseline" not in runs:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_df = runs["baseline"]
    for name, df in runs.items():
        if name == "baseline" or "train_loss" not in df:
            continue
        merged = pd.merge(baseline_df[["iter", "train_loss"]], df[["iter", "train_loss"]], on="iter", suffixes=("_base", "_method"))
        base_smoothed = _ema(merged["train_loss_base"], alpha=0.05)
        method_smoothed = _ema(merged["train_loss_method"], alpha=0.05)
        gap = method_smoothed - base_smoothed
        ax.plot(
            merged["iter"],
            gap,
            label=f"{name} vs baseline",
            color=COLORS.get(name, None),
            linewidth=2,
        )
        
    ax.axhline(0.0, color="red", linewidth=1.5, linestyle="--", label="Baseline Reference")
    ax.set_title("Absolute Training Loss Gap (Method - Baseline)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss Gap (Negative is better)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    path = output_dir / "train_loss_gap_curve.png"
    _save_fig(fig, path)
    return path


def _plot_grad_norm(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        # Đã SỬA: Check linh hoạt tên cột gradient
        col_name = "grad_pre" if "grad_pre" in df else "grad_norm" if "grad_norm" in df else None
        if not col_name:
            continue
            
        ema_series = _ema(df[col_name], alpha=0.05)
        ax.plot(df["iter"], df[col_name], color=COLORS.get(name, None), alpha=0.15)
        ax.plot(df["iter"], ema_series, label=name, color=COLORS.get(name, None), linewidth=2)
        
    ax.set_title("Gradient Norm vs Iteration (Smoothed)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Gradient L2 Norm", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    path = output_dir / "grad_norm_curve.png"
    _save_fig(fig, path)
    return path


def _plot_best_val_bar(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    variants, values = [], []
    for name, df in runs.items():
        subset = df.dropna(subset=["val_loss"])
        if subset.empty:
            continue
        variants.append(name)
        values.append(subset["val_loss"].min())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(variants, values, color=[COLORS.get(v, "#999999") for v in variants], width=0.5)
    
    # Đã SỬA: Thêm Text nổi trên đỉnh cột
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=12, fontweight='bold')
    
    ax.set_title("Best Validation Loss", fontsize=14, fontweight='bold')
    ax.set_ylabel("Loss (Lower is better)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15) # Tăng trần để không bị cắt chữ
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    path = output_dir / "best_val_loss_bar.png"
    _save_fig(fig, path)
    return path


def _plot_final_ppl_bar(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    variants, values = [], []
    for name, df in runs.items():
        subset = df.dropna(subset=["val_loss"])
        if subset.empty:
            continue
        last_val = subset["val_loss"].iloc[-1]
        variants.append(name)
        values.append(math.exp(last_val) if last_val < 20 else float("inf"))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(variants, values, color=[COLORS.get(v, "#999999") for v in variants], width=0.5)
    
    # Đã SỬA: Thêm Text nổi trên đỉnh cột
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=12, fontweight='bold')
    
    ax.set_title("Final Validation Perplexity", fontsize=14, fontweight='bold')
    ax.set_ylabel("Perplexity (Lower is better)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    path = output_dir / "final_val_ppl_bar.png"
    _save_fig(fig, path)
    return path


def _plot_amax_curve(runs: Dict[str, pd.DataFrame], output_dir: Path, col: str, filename: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(1.0, color=COLORS.get("baseline", "#4C78A8"), linestyle="--", linewidth=2, label="Baseline = 1.0 (Identity)")
    ax.axhline(1.6, color="#777777", linestyle=":", linewidth=2, label="mHC Safety Bound = 1.6")

    for name, df in runs.items():
        if name == "baseline":
            continue
        subset = df.dropna(subset=[col])
        if subset.empty:
            continue
        ax.plot(
            subset["iter"],
            subset[col],
            label=name,
            color=COLORS.get(name, None),
            linewidth=2,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(col.replace("_", " ").title() + " (Log Scale)", fontsize=12)
    
    # Đã SỬA: Đưa Amax về thang đo Logarit để thấy được vực thẳm của HC vs mHC
    ax.set_yscale('log')
    
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)

    path = output_dir / filename
    _save_fig(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Qwen2.5 comparison figures")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Dùng whitegrid cho nó khoa học
    sns.set_theme(style="whitegrid")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_paths = _parse_runs(args.runs)
    runs = {name: _load_history(path) for name, path in run_paths.items()}

    outputs: List[Path] = []
    outputs.append(_plot_val_loss(runs, output_dir))
    outputs.append(_plot_train_loss(runs, output_dir))
    
    gap_path = _plot_train_loss_gap(runs, output_dir)
    if gap_path:
        outputs.append(gap_path)
        
    outputs.append(_plot_grad_norm(runs, output_dir))
    outputs.append(_plot_best_val_bar(runs, output_dir))
    outputs.append(_plot_final_ppl_bar(runs, output_dir))
    
    outputs.append(
        _plot_amax_curve(
            runs, output_dir, col="amax_fwd_max", filename="amax_fwd_curve.png", title="Amax Forward Gain vs Iteration (Topology Stability)"
        )
    )
    outputs.append(
        _plot_amax_curve(
            runs, output_dir, col="amax_bwd_max", filename="amax_bwd_curve.png", title="Amax Backward Gain vs Iteration (Topology Stability)"
        )
    )

    print("✅ Generated figures successfully:")
    for path in outputs:
        print(f"✅ {path}")


if __name__ == "__main__":
    main()
