"""Stable programmatic entry point for factor research evaluation.

This module deliberately orchestrates the existing :class:`FactorBacktester`
instead of duplicating its backtest statistics.  It adds the small amount of
input handling and structured output needed by research-control-plane clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import polars as pl

from .config import load_frame
from .data import FactorDataset
from .factor import FactorBacktestConfig, FactorBacktestResult, FactorBacktester
from .metrics import correlation, rank_correlation, safe_mean, safe_std


FrameInput: TypeAlias = pl.DataFrame | str | Path


@dataclass(frozen=True)
class FactorEvaluationConfig:
    """Configuration for :func:`evaluate_factors`.

    Returns are point-in-time forward returns keyed by the factor observation
    date.  The evaluator never shifts them implicitly.
    """

    start_date: str | None = None
    end_date: str | None = None
    min_assets: int = 5
    max_stability_lag: int = 10
    annualization: int = 252
    buckets: int = 5
    cost_bps: float = 0.0
    topk: float = 1.0
    power: float = 2.0

    def __post_init__(self) -> None:
        if self.min_assets < 3:
            raise ValueError("min_assets must be at least 3 for correlation metrics")
        if self.max_stability_lag < 1:
            raise ValueError("max_stability_lag must be positive")
        if self.annualization < 1:
            raise ValueError("annualization must be positive")
        if self.buckets < 2:
            raise ValueError("buckets must be at least 2")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if not 0 < self.topk <= 1:
            raise ValueError("topk must be in (0, 1]")
        if self.power <= 0:
            raise ValueError("power must be positive")


@dataclass
class FactorEvaluationResult:
    """Structured, JSON-serializable evaluation result.

    ``backtest_result`` remains available for callers that need the package's
    full report tables, while :meth:`to_dict` exposes only the stable research
    contract.
    """

    factors: tuple[str, ...]
    data_quality: dict[str, dict[str, Any]]
    intrinsic: dict[str, dict[str, Any]]
    stability: dict[str, dict[str, Any]]
    correlations: list[dict[str, Any]]
    daily_metrics: pl.DataFrame
    metadata: dict[str, Any]
    backtest_result: FactorBacktestResult = field(repr=False)
    schema_version: str = "factor-evaluation/v1"

    def to_dict(self, *, include_daily: bool = True) -> dict[str, Any]:
        """Return a JSON-safe representation of the stable result contract."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "factors": list(self.factors),
            "data_quality": self.data_quality,
            "intrinsic": self.intrinsic,
            "stability": self.stability,
            "correlations": self.correlations,
            "metadata": self.metadata,
        }
        if include_daily:
            payload["daily_metrics"] = self.daily_metrics.to_dicts()
        return _json_safe(payload)

    def write_json(self, path: str | Path, *, include_daily: bool = True) -> Path:
        """Write the stable result contract as UTF-8 JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(include_daily=include_daily),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output


def evaluate_factors(
    factors: FrameInput,
    forward_returns: FrameInput,
    *,
    factor_names: list[str] | tuple[str, ...] | None = None,
    benchmark: FrameInput | None = None,
    universe: FrameInput | None = None,
    styles: FrameInput | None = None,
    limit_labels: FrameInput | None = None,
    config: FactorEvaluationConfig | None = None,
) -> FactorEvaluationResult:
    """Evaluate one or more factor columns from long-table inputs or paths.

    Factor and return inputs must share ``date`` and ``asset``/``code``/``header``
    keys.  ``forward_returns`` must contain ``return`` (or exactly one value
    column).  Supported paths are CSV, parquet, IPC, Feather, and Arrow files.

    The stable result contains data quality, daily and aggregate Pearson IC and
    RankIC, IC stability, factor autocorrelation, and pairwise factor
    correlation.  The full existing backtest output is retained as
    ``result.backtest_result``.
    """

    evaluation_config = config or FactorEvaluationConfig()
    factor_frame = _read_frame(factors, "factors")
    return_frame = _read_frame(forward_returns, "forward_returns")
    benchmark_frame = _read_frame(benchmark, "benchmark") if benchmark is not None else None
    universe_frame = _read_frame(universe, "universe") if universe is not None else None
    styles_frame = _read_frame(styles, "styles") if styles is not None else None
    limit_label_frame = (
        _read_frame(limit_labels, "limit_labels") if limit_labels is not None else None
    )

    # Normalize through the package's existing data contract.  A zero benchmark
    # is sufficient for intrinsic evaluation and avoids making it a required
    # control-plane input.
    provisional = FactorDataset(
        factors=factor_frame,
        returns=return_frame,
        benchmark=(
            benchmark_frame
            if benchmark_frame is not None
            else _zero_benchmark(return_frame)
        ),
        universe=universe_frame,
        styles=styles_frame,
        limit_labels=limit_label_frame,
    ).normalized()

    available = [
        column for column in provisional.factors.columns if column not in {"date", "asset"}
    ]
    selected = list(factor_names) if factor_names is not None else available
    if not selected:
        raise ValueError("No factor value columns found")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Unknown factor columns: {missing}; available columns: {available}")
    if len(set(selected)) != len(selected):
        raise ValueError("factor_names contains duplicates")

    numeric_factors = provisional.factors.select(
        "date",
        "asset",
        *[pl.col(name).cast(pl.Float64, strict=False).alias(name) for name in selected],
    )
    normalized_styles = provisional.styles
    dataset = FactorDataset(
        factors=numeric_factors,
        returns=provisional.returns,
        benchmark=provisional.benchmark,
        universe=provisional.universe,
        styles=normalized_styles,
        limit_labels=provisional.limit_labels,
    )
    backtest_config = FactorBacktestConfig(
        start_date=evaluation_config.start_date,
        end_date=evaluation_config.end_date,
        buckets=evaluation_config.buckets,
        annualization=evaluation_config.annualization,
        max_lag=evaluation_config.max_stability_lag,
        min_assets=evaluation_config.min_assets,
        topk=evaluation_config.topk,
        power=evaluation_config.power,
        test_basic_features=True,
        test_basic_return=True,
        test_style_features=styles_frame is not None,
        cost_bps=evaluation_config.cost_bps,
    )
    backtest = FactorBacktester(backtest_config).run(dataset)

    filtered_factors = _filter_dates(numeric_factors, evaluation_config)
    filtered_returns = _filter_dates(provisional.returns, evaluation_config)
    filtered_universe = (
        _filter_dates(provisional.universe, evaluation_config)
        if provisional.universe is not None
        else None
    )
    daily_metrics = _daily_intrinsic(backtest, filtered_factors, filtered_returns, selected)
    data_quality = _data_quality(
        backtest,
        filtered_factors,
        filtered_returns,
        filtered_universe,
        selected,
    )
    intrinsic = _intrinsic_summary(backtest, daily_metrics, selected)
    stability = _stability_summary(backtest, daily_metrics, selected)
    correlations = _factor_correlations(filtered_factors, selected)

    return FactorEvaluationResult(
        factors=tuple(selected),
        data_quality=data_quality,
        intrinsic=intrinsic,
        stability=stability,
        correlations=correlations,
        daily_metrics=daily_metrics,
        metadata={
            "return_alignment": "forward return keyed by factor observation date; no implicit shift",
            "start_date": evaluation_config.start_date,
            "end_date": evaluation_config.end_date,
            "min_assets": evaluation_config.min_assets,
            "max_stability_lag": evaluation_config.max_stability_lag,
            "annualization": evaluation_config.annualization,
            "buckets": evaluation_config.buckets,
            "topk": evaluation_config.topk,
            "power": evaluation_config.power,
            "cost_bps": evaluation_config.cost_bps,
            "limit_labels_supplied": backtest.metadata["limit_labels_supplied"],
            "limit_label_coverage": backtest.metadata["limit_label_coverage"],
            "limit_label_hit_rate": backtest.metadata["limit_label_hit_rate"],
            "factor_evaluator_version": "0.1.0",
        },
        backtest_result=backtest,
    )


def _read_frame(value: FrameInput, label: str) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value.clone()
    if isinstance(value, (str, Path)):
        return load_frame(Path(value))
    raise TypeError(f"{label} must be a polars DataFrame or a supported file path")


def _zero_benchmark(returns: pl.DataFrame) -> pl.DataFrame:
    if "date" not in returns.columns:
        raise ValueError("Missing date column in forward_returns")
    return (
        returns.select(pl.col("date").cast(pl.String))
        .unique()
        .sort("date")
        .with_columns(pl.lit(0.0).alias("benchmark_return"))
    )


def _filter_dates(frame: pl.DataFrame | None, config: FactorEvaluationConfig) -> pl.DataFrame:
    if frame is None:
        return pl.DataFrame()
    result = frame
    if config.start_date:
        result = result.filter(pl.col("date") >= config.start_date.replace("-", ""))
    if config.end_date:
        result = result.filter(pl.col("date") <= config.end_date.replace("-", ""))
    return result


def _daily_intrinsic(
    backtest: FactorBacktestResult,
    factors: pl.DataFrame,
    returns: pl.DataFrame,
    factor_names: list[str],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    joined = factors.join(
        returns.select("date", "asset", "return"), on=["date", "asset"], how="inner"
    )
    for factor in factor_names:
        rank_by_date = {
            date: value
            for date, value in backtest.daily[factor].select("date", "ic").iter_rows()
        }
        for day in joined.select("date", factor, "return").partition_by(
            "date", maintain_order=True
        ):
            date = day["date"][0]
            if date not in rank_by_date:
                continue
            left = day[factor].to_numpy()
            right = day["return"].to_numpy()
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "pearson_ic": correlation(left, right),
                    # Use the existing backtester result as the authoritative
                    # RankIC rather than recomputing it in this facade.
                    "rank_ic": rank_by_date[date],
                    "n_assets": int((np.isfinite(left) & np.isfinite(right)).sum()),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.String,
            "factor": pl.String,
            "pearson_ic": pl.Float64,
            "rank_ic": pl.Float64,
            "n_assets": pl.Int64,
        },
    ).sort(["factor", "date"])


def _data_quality(
    backtest: FactorBacktestResult,
    factors: pl.DataFrame,
    returns: pl.DataFrame,
    universe: pl.DataFrame | None,
    factor_names: list[str],
) -> dict[str, dict[str, Any]]:
    if universe is not None:
        value_columns = [
            column for column in universe.columns if column not in {"date", "asset"}
        ]
        eligible = (
            universe.filter(pl.col(value_columns[0]).cast(pl.Boolean))
            if value_columns
            else universe
        )
        base = eligible.select("date", "asset").unique()
    else:
        base = returns.select("date", "asset").unique()
    result: dict[str, dict[str, Any]] = {}
    for factor in factor_names:
        aligned = base.join(factors.select("date", "asset", factor), on=["date", "asset"], how="left")
        values = aligned[factor].to_numpy()
        finite = np.isfinite(values)
        expected = aligned.height
        valid = int(finite.sum())
        daily_coverage = (
            aligned.with_columns(pl.Series("_valid", finite))
            .group_by("date")
            .agg(pl.col("_valid").mean().alias("coverage"))
        )
        distribution = _factor_table_row(backtest, "04_distribution", factor)
        warnings: list[str] = []
        if valid < expected:
            warnings.append("factor contains missing or non-finite values in the evaluation universe")
        if valid == 0:
            warnings.append("factor has no finite observations")
        if distribution.get("std") is None or not _is_finite(distribution.get("std")) or distribution.get("std") == 0:
            warnings.append("factor is constant or has no measurable dispersion")
        result[factor] = {
            "expected_observations": expected,
            "valid_observations": valid,
            "missing_observations": expected - valid,
            "missing_ratio": (expected - valid) / expected if expected else None,
            "n_dates": aligned["date"].n_unique(),
            "n_assets": aligned["asset"].n_unique(),
            "mean_daily_coverage": (
                float(daily_coverage["coverage"].mean()) if daily_coverage.height else None
            ),
            "unique_value_ratio": (
                int(factors.filter(pl.col(factor).is_finite())[factor].n_unique()) / valid
                if valid
                else None
            ),
            "warnings": warnings,
        }
    return result


def _intrinsic_summary(
    backtest: FactorBacktestResult,
    daily: pl.DataFrame,
    factor_names: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for factor in factor_names:
        factor_daily = daily.filter(pl.col("factor") == factor)
        pearson = factor_daily["pearson_ic"].to_numpy()
        rank_summary = _factor_table_row(backtest, "07_ic_summary", factor)
        result[factor] = {
            "pearson_ic": safe_mean(pearson),
            "pearson_icir": _information_ratio(pearson),
            "rank_ic": rank_summary.get("ic"),
            "rank_icir": rank_summary.get("icir"),
            "n_periods": factor_daily.height,
        }
    return result


def _stability_summary(
    backtest: FactorBacktestResult,
    daily: pl.DataFrame,
    factor_names: list[str],
) -> dict[str, dict[str, Any]]:
    autocorrelation = backtest.tables.get("10_autocorrelation", pl.DataFrame())
    result: dict[str, dict[str, Any]] = {}
    for factor in factor_names:
        factor_daily = daily.filter(pl.col("factor") == factor)
        pearson = factor_daily["pearson_ic"].to_numpy()
        rank_ic = factor_daily["rank_ic"].to_numpy()
        lag_rows = (
            autocorrelation.filter(pl.col("factor") == factor).sort("lag").select(
                "lag", "autocorrelation"
            ).to_dicts()
            if not autocorrelation.is_empty()
            else []
        )
        result[factor] = {
            "pearson_ic_std": safe_std(pearson),
            "rank_ic_std": safe_std(rank_ic),
            "pearson_ic_positive_ratio": _positive_ratio(pearson),
            "rank_ic_positive_ratio": _positive_ratio(rank_ic),
            "factor_autocorrelation": lag_rows,
        }
    return result


def _factor_correlations(
    factors: pl.DataFrame, factor_names: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(factor_names):
        for right_name in factor_names[left_index + 1 :]:
            pearson_values: list[float] = []
            rank_values: list[float] = []
            for day in factors.select("date", left_name, right_name).partition_by(
                "date", maintain_order=True
            ):
                left = day[left_name].to_numpy()
                right = day[right_name].to_numpy()
                pearson_values.append(correlation(left, right))
                rank_values.append(rank_correlation(left, right))
            rows.append(
                {
                    "left_factor": left_name,
                    "right_factor": right_name,
                    "mean_cross_sectional_pearson": safe_mean(np.asarray(pearson_values)),
                    "mean_cross_sectional_rank": safe_mean(np.asarray(rank_values)),
                }
            )
    return rows


def _factor_table_row(
    backtest: FactorBacktestResult, table_name: str, factor: str
) -> dict[str, Any]:
    table = backtest.tables.get(table_name, pl.DataFrame())
    if table.is_empty() or "factor" not in table.columns:
        return {}
    matched = table.filter(pl.col("factor") == factor)
    return matched.row(0, named=True) if matched.height else {}


def _information_ratio(values: np.ndarray) -> float:
    deviation = safe_std(values)
    return safe_mean(values) / deviation if deviation > 0 else float("nan")


def _positive_ratio(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float((finite > 0).mean()) if finite.size else float("nan")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value
