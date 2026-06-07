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
        )
    ax.set_title("Validation Loss vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "val_loss_curve.png"
    _save_fig(fig, path)
    return path


def _plot_train_loss(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        if "train_loss" not in df:
            continue
        ema_series = _ema(df["train_loss"], alpha=0.95)
        ax.plot(
            df["iter"],
            ema_series,
            label=name,
            color=COLORS.get(name, None),
            linewidth=2,
        )
    ax.set_title("Training Loss vs Iteration (EMA)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "train_loss_curve.png"
    _save_fig(fig, path)
    return path


def _plot_train_loss_gap(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path | None:
    if "baseline" not in runs:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_df = runs["baseline"]
    for name, df in runs.items():
        if "train_loss" not in df:
            continue
        merged = pd.merge(baseline_df[["iter", "train_loss"]], df[["iter", "train_loss"]], on="iter", suffixes=("_base", "_method"))
        base_smoothed = _ema(merged["train_loss_base"], alpha=0.05)
        method_smoothed = _ema(merged["train_loss_method"], alpha=0.05)
        gap = method_smoothed - base_smoothed
        ax.plot(
            merged["iter"],
            gap,
            label=name,
            color=COLORS.get(name, None),
            linewidth=2,
        )
    ax.axhline(0.0, color="gray", linewidth=1.5)
    ax.set_title("Absolute Training Loss Gap vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss Gap (Method - Baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "train_loss_gap_curve.png"
    _save_fig(fig, path)
    return path


def _plot_grad_norm(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        if "grad_norm" not in df:
            continue
        ema_series = _ema(df["grad_norm"], alpha=0.1)
        ax.plot(
            df["iter"],
            ema_series,
            label=name,
            color=COLORS.get(name, None),
            linewidth=2,
        )
    ax.set_title("Gradient Norm vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "grad_norm_curve.png"
    _save_fig(fig, path)
    return path


def _plot_best_val_bar(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    variants = []
    values = []
    for name, df in runs.items():
        subset = df.dropna(subset=["val_loss"])
        if subset.empty:
            continue
        variants.append(name)
        values.append(subset["val_loss"].min())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(variants, values, color=[COLORS.get(v, "#999999") for v in variants])
    ax.set_title("Best Validation Loss")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Best Val Loss")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "best_val_loss_bar.png"
    _save_fig(fig, path)
    return path


def _plot_final_ppl_bar(runs: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    variants = []
    values = []
    for name, df in runs.items():
        subset = df.dropna(subset=["val_loss"])
        if subset.empty:
            continue
        last_val = subset["val_loss"].iloc[-1]
        variants.append(name)
        values.append(math.exp(last_val) if last_val < 20 else float("inf"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(variants, values, color=[COLORS.get(v, "#999999") for v in variants])
    ax.set_title("Final Validation Perplexity")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Final Val PPL")
    ax.grid(True, axis="y", alpha=0.3)
    path = output_dir / "final_val_ppl_bar.png"
    _save_fig(fig, path)
    return path


def _plot_amax_curve(
    runs: Dict[str, pd.DataFrame],
    output_dir: Path,
    col: str,
    filename: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(
        1.0,
        color=COLORS.get("baseline", "#4C78A8"),
        linestyle="--",
        linewidth=2,
        label="baseline = 1.0 (identity)",
    )
    ax.axhline(1.6, color="#777777", linestyle=":", linewidth=1.5, label="mHC bound = 1.6")

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

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(col.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = output_dir / filename
    _save_fig(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Qwen2.5 comparison figures")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sns.set_theme(style="white")
    plt.style.use("seaborn-v0_8-white")

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
            runs,
            output_dir,
            col="amax_fwd",
            filename="amax_fwd_curve.png",
            title="Amax Forward Gain vs Iteration",
        )
    )
    outputs.append(
        _plot_amax_curve(
            runs,
            output_dir,
            col="amax_bwd",
            filename="amax_bwd_curve.png",
            title="Amax Backward Gain vs Iteration",
        )
    )

    print("✅ Generated figures:")
    for path in outputs:
        print(f"✅ {path}")


if __name__ == "__main__":
    main()
