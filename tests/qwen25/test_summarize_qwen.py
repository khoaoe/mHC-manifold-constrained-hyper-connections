"""Unit tests for summarize_qwen_runs.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarize_qwen_runs as s


@pytest.fixture
def fake_run(tmp_path: Path) -> Path:
    """Create a fake run dir with a valid summary.json."""
    run_dir = tmp_path / "out-qwen-baseline"
    run_dir.mkdir()
    summary = {
        "ok": True,
        "method": "baseline",
        "iter_num": 5000,
        "best_val_loss": 3.14,
        "final_train_loss": 3.0,
        "final_val_loss": 3.14,
        "final_val_ppl": 23.1,
        "amax_fwd_final": 1.0,
        "amax_fwd_max": 1.0,
        "amax_fwd_mean": 1.0,
        "amax_bwd_final": 1.0,
        "amax_bwd_max": 1.0,
        "amax_bwd_mean": 1.0,
        "peak_vram_gb": 12.5,
        "elapsed_s": 3600.0,
        "avg_attn_entropy": 1.25,
        "residual_norm_final": 14.2,
        "hc_grad_norm_pre_clip": 2.45,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
class TestParseRuns:
    def test_happy_path(self):
        runs = s._parse_runs(["baseline=/a", "hc=/b", "mhc=/c"])
        assert runs == {"baseline": Path("/a"), "hc": Path("/b"), "mhc": Path("/c")}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError):
            s._parse_runs(["baseline"])

    def test_preserves_paths_with_equals(self):
        runs = s._parse_runs(["x=/path/with=equals"])
        assert runs["x"] == Path("/path/with=equals")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
class TestFormatVal:
    def test_int(self):
        assert s._format_val(5) == "5"

    def test_float(self):
        assert s._format_val(3.14159) == "3.1416"

    def test_nan(self):
        assert s._format_val(float("nan")) == "nan"

    def test_none(self):
        assert s._format_val(None) == "nan"


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------
class TestMarkdownTable:
    def test_basic_table(self):
        md = s._markdown_table(
            headers=["a", "b"],
            rows=[["1", "2"], ["33", "4"]],
        )
        assert "| a" in md
        assert "---" in md
        assert "33" in md

    def test_empty_rows(self):
        md = s._markdown_table(headers=["x"], rows=[])
        assert "x" in md


# ---------------------------------------------------------------------------
# End-to-end summary generation
# ---------------------------------------------------------------------------
class TestMain:
    def test_produces_all_artifacts(self, fake_run: Path, tmp_path: Path, monkeypatch):
        out_dir = tmp_path / "report"
        monkeypatch.setattr(
            "sys.argv",
            [
                "summarize_qwen_runs.py",
                "--runs",
                f"baseline={fake_run}",
                "--output-dir",
                str(out_dir),
            ],
        )
        s.main()
        assert (out_dir / "training_summary.csv").exists()
        assert (out_dir / "training_summary.json").exists()
        assert (out_dir / "training_summary.md").exists()

        csv_text = (out_dir / "training_summary.csv").read_text()
        assert "baseline" in csv_text
        assert "3.14" in csv_text or "3.1400" in csv_text

    def test_missing_summary_raises(self, tmp_path: Path, monkeypatch):
        empty_run = tmp_path / "empty"
        empty_run.mkdir()
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            [
                "summarize_qwen_runs.py",
                "--runs",
                f"x={empty_run}",
                "--output-dir",
                str(out_dir),
            ],
        )
        with pytest.raises(FileNotFoundError):
            s.main()

class TestNewMetricsInCSV:
    def test_csv_contains_new_metrics(self, fake_run: Path, tmp_path: Path, monkeypatch):
        out_dir = tmp_path / "report"
        monkeypatch.setattr("sys.argv", [
            "summarize_qwen_runs.py", "--runs", f"baseline={fake_run}",
            "--output-dir", str(out_dir),
        ])
        s.main()
        csv_text = (out_dir / "training_summary.csv").read_text()
        assert "avg_attn_entropy" in csv_text
        assert "1.25" in csv_text or "1.2500" in csv_text
