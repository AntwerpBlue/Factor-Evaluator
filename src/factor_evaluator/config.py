from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any

import polars as pl

from .data import FactorDataset, PortfolioDataset


PLACEHOLDER_TOKEN = "REPLACE_ME"


@dataclass(frozen=True)
class BacktestFileConfig:
    path: Path
    paths: dict[str, str]
    benchmarks: dict[str, dict[str, Any]]
    factor: dict[str, Any]
    portfolio: dict[str, Any]
    output: dict[str, Any]

    def resolve(self, value: str, *, required: bool = True) -> Path | None:
        value = os.path.expandvars(os.path.expanduser(str(value).strip()))
        if not value:
            if required:
                raise ValueError("A required data path is empty")
            return None
        if PLACEHOLDER_TOKEN in value:
            if required:
                raise ValueError(f"Data path is still a placeholder: {value}")
            return None
        path = Path(value)
        return path if path.is_absolute() else (self.path.parent / path).resolve()


def load_config(path: str | Path) -> BacktestFileConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return BacktestFileConfig(
        path=config_path,
        paths=dict(raw.get("paths", {})),
        benchmarks=dict(raw.get("benchmarks", {})),
        factor=dict(raw.get("factor", {})),
        portfolio=dict(raw.get("portfolio", {})),
        output=dict(raw.get("output", {})),
    )


def load_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, schema_overrides={
            name: pl.String for name in ("date", "Date", "asset", "code", "Code", "header")
        })
    if suffix in {".ipc", ".feather", ".arrow"}:
        return pl.read_ipc(path)
    raise ValueError(f"Unsupported data file format: {path}")


def _optional_frame(config: BacktestFileConfig, key: str) -> pl.DataFrame | None:
    path = config.resolve(config.paths.get(key, ""), required=False)
    return load_frame(path) if path is not None else None


def _benchmark_frame(
    config: BacktestFileConfig, benchmark: str, key: str, *, required: bool = True
) -> pl.DataFrame | None:
    if benchmark not in config.benchmarks:
        raise KeyError(f"Unknown benchmark '{benchmark}'")
    path = config.resolve(
        str(config.benchmarks[benchmark].get(key, "")), required=required
    )
    return load_frame(path) if path is not None else None


def load_factor_dataset(config: BacktestFileConfig, benchmark: str) -> FactorDataset:
    return FactorDataset(
        factors=load_frame(config.resolve(config.paths["factor_values"])),
        returns=load_frame(config.resolve(config.paths["asset_forward_returns"])),
        benchmark=_benchmark_frame(config, benchmark, "returns"),
        universe=_optional_frame(config, "universe"),
        styles=_optional_frame(config, "style_exposures"),
        limit_labels=_optional_frame(config, "tradability"),
    )


def load_portfolio_dataset(
    config: BacktestFileConfig, benchmark: str
) -> PortfolioDataset:
    return PortfolioDataset(
        returns=load_frame(config.resolve(config.paths["asset_forward_returns"])),
        benchmark=_benchmark_frame(config, benchmark, "returns"),
        benchmark_weights=_benchmark_frame(config, benchmark, "weights", required=False),
        styles=_optional_frame(config, "style_exposures"),
        industry_exposures=_optional_frame(config, "industry_exposures"),
        style_returns=_optional_frame(config, "style_factor_returns"),
        industry_returns=_optional_frame(config, "industry_factor_returns"),
    )


def load_target_weights(config: BacktestFileConfig) -> pl.DataFrame:
    return load_frame(config.resolve(config.paths["target_weights"]))


