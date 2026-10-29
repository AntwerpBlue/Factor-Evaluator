from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from factor_evaluator.cli import COMMANDS
from factor_evaluator._commands import run_end_to_end_backtest as pipeline
from test_config_and_reports import full_synthetic


ENTRY = Path(__file__).resolve().parents[1] / "run_backtest.py"


def run_entry(tmp_path, *args):
    result = subprocess.run(
        [sys.executable, str(ENTRY), *map(str, args)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("command", [None, *COMMANDS])
def test_single_entry_help_outside_repository(tmp_path, command):
    args = ["--help"] if command is None else [command, "--help"]
    assert "usage:" in run_entry(tmp_path, *args).stdout


@pytest.mark.parametrize("suffix", ["csv", "feather"])
def test_factor_cli_formats_dates_limits_and_explicit_exports(tmp_path, suffix):
    factors, returns, _, _, _ = full_synthetic(n_dates=5, n_assets=6)
    factors = factors.with_columns(pl.col("code").str.slice(0, 6)).rename({"code": "header"})
    factors = factors.with_columns((-pl.col("alpha")).alias("beta"))
    returns = returns.with_columns(pl.col("code").str.slice(0, 6))
    labels = returns.select("date", "code").with_columns(pl.lit(1).alias("limit_label"))
    paths = [tmp_path / f"{name}.{suffix}" for name in ("factor", "returns", "limits")]
    for frame, path in zip((factors, returns, labels), paths):
        if suffix == "csv":
            frame.write_csv(path)
        else:
            frame.write_ipc(path)
    run_entry(tmp_path, "factor", "--factor", paths[0], "--forward-returns", paths[1],
              "--limit-labels", paths[2], "--min-assets", "3", "--buckets", "2",
              "--json", tmp_path / "evaluation.json", "--plot", tmp_path / "chart.svg")
    for factor in ("alpha", "beta"):
        payload = json.loads((tmp_path / f"evaluation_{factor}.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["limit_labels_supplied"] is True
        assert payload["intrinsic"][factor]["n_periods"] == 5
        assert (tmp_path / f"chart_{factor}.svg").stat().st_size > 1000
    assert not (tmp_path / "evaluation.json").exists()


def test_signal_lag_is_explicit_and_uses_calendar(tmp_path):
    from factor_evaluator._commands.factor import shift_factor_to_next_trade_date

    calendar = tmp_path / "days.txt"
    calendar.write_text("20240105\n20240108\n20240108\n20240109\n")
    frame = pl.DataFrame({"date": ["20240105", "20240109"], "code": ["000001", "000001"], "alpha": [1., 2.]})
    shifted = shift_factor_to_next_trade_date(frame, str(calendar))
    assert shifted["date"].to_list() == ["20240108"]
    result = subprocess.run([sys.executable, str(ENTRY), "factor", "--factor", "unused.csv",
                             "--forward-returns", "unused.csv", "--signal-lag", "1"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--signal-lag 1 requires --trading-days" in result.stderr


@pytest.mark.parametrize("task", ["both", "portfolio"])
def test_config_entry_exports_requested_tasks(tmp_path, task):
    factors, returns, benchmark, styles, weights = full_synthetic(n_dates=6, n_assets=6)
    for name, frame in (("factor", factors), ("returns", returns), ("benchmark", benchmark),
                        ("styles", styles), ("weights", weights)):
        frame.write_parquet(tmp_path / f"{name}.parquet")
    factor_line = 'factor_values = "factor.parquet"' if task == "both" else ""
    config = tmp_path / "settings.toml"
    config.write_text(f'''[paths]
{factor_line}
asset_forward_returns = "returns.parquet"
style_exposures = "styles.parquet"
target_weights = "weights.parquet"
[benchmarks.custom]
returns = "benchmark.parquet"
[factor]
buckets = 2
min_assets = 3
max_lag = 2
[output]
root = "report"
''', encoding="utf-8")
    run_entry(tmp_path, "--config", config, "--task", task, "--benchmark", "custom")
    assert (tmp_path / "report/portfolio/DONE.ok").exists()
    assert (tmp_path / "report/single_factor/factor_metrics_summary.xlsx").exists() == (task == "both")


def test_pipeline_passes_input_paths_and_uses_only_public_entry(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: captured.append(command))
    args = pipeline.parse_args(["--factor-values", "factor.parquet", "--start", "20240101",
                                "--end", "20240103", "--data-dir", str(tmp_path / "data"),
                                "--risk-dir", str(tmp_path / "risk")])
    pipeline.ensure_local_cache(args, tmp_path / "cache", tmp_path / "data", "20240101", "20240103")
    command = captured[0]
    assert Path(command[1]) == ENTRY
    assert command[2] == "prepare-data"
    assert command[command.index("--wind-baseinfo") + 1] == str(tmp_path / "data/wind_baseinfo.parquet")
    assert command[command.index("--risk-dir") + 1] == str((tmp_path / "risk").resolve())
