"""Summarize Qwen2.5-0.5B HC/mHC runs into report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _parse_runs(run_args: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Invalid run spec: {item}")
        name, path = item.split("=", 1)
        runs[name.strip()] = Path(path).expanduser()
    return runs


def _format_val(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * max(widths[i], 3) for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Qwen2.5 runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = _parse_runs(args.runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant, run_dir in runs.items():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        rows.append(
            {
                "variant": variant,
                "best_val_loss": float(summary.get("best_val_loss", float("nan"))),
                "final_train_loss": float(summary.get("final_train_loss", float("nan"))),
                "final_val_loss": float(summary.get("final_val_loss", float("nan"))),
                "final_val_ppl": float(summary.get("final_val_ppl", float("nan"))),
                "amax_fwd_final": float(summary.get("amax_fwd_final", 1.0)),
                "amax_fwd_max": float(summary.get("amax_fwd_max", 1.0)),
                "amax_fwd_mean": float(summary.get("amax_fwd_mean", 1.0)),
                "amax_bwd_final": float(summary.get("amax_bwd_final", 1.0)),
                "amax_bwd_max": float(summary.get("amax_bwd_max", 1.0)),
                "amax_bwd_mean": float(summary.get("amax_bwd_mean", 1.0)),
                "iter_num": int(summary.get("iter_num", 0)),
                "elapsed_s": float(summary.get("elapsed_s", float("nan"))),
                "peak_vram_gb": float(summary.get("peak_vram_gb", float("nan"))),
                "avg_attn_entropy": float(summary.get("avg_attn_entropy", float("nan"))),
                "residual_norm_final": float(summary.get("residual_norm_final", float("nan"))),
                "hc_grad_norm_pre_clip": float(summary.get("hc_grad_norm_pre_clip", float("nan"))),
            }
        )

    columns = [
        "variant",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "final_val_ppl",
        "amax_fwd_final",
        "amax_fwd_max",
        "amax_fwd_mean",
        "amax_bwd_final",
        "amax_bwd_max",
        "amax_bwd_mean",
        "iter_num",
        "elapsed_s",
        "peak_vram_gb",
        "avg_attn_entropy",
        "residual_norm_final",
        "hc_grad_norm_pre_clip",
    ]

    csv_path = output_dir / "training_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "training_summary.json"
    json_payload = {row["variant"]: row for row in rows}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    md_path = output_dir / "training_summary.md"
    md_headers = columns
    md_rows = [[_format_val(row[col]) for col in md_headers] for row in rows]
    md_table = _markdown_table(md_headers, md_rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5 Training Summary\n\n")
        f.write(md_table)
        f.write("\n")

    print("✅ Wrote training_summary.csv")
    print("✅ Wrote training_summary.json")
    print("✅ Wrote training_summary.md")


if __name__ == "__main__":
    main()
