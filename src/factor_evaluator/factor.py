from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .data import FactorDataset
from .metrics import (
    annualized_return,
    annualized_volatility,
    correlation,
    max_drawdown,
    rank,
    rank_correlation,
    safe_mean,
    safe_std,
    sharpe_ratio,
)


@dataclass(frozen=True)
class FactorBacktestConfig:
    start_date: str | None = None
    end_date: str | None = None
    out_of_sample_start: str | None = None
    single_interval: bool = True
    buckets: int = 30
    topk: float = 1.0
    power: float = 2.0
    annualization: int = 252
    max_lag: int = 10
    min_assets: int = 5
    test_basic_features: bool = True
    test_basic_return: bool = True
    test_style_features: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.topk <= 1:
            raise ValueError("topk must be in (0, 1]")
        if self.buckets < 2:
            raise ValueError("buckets must be at least 2")
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")


@dataclass
class FactorBacktestResult:
    tables: dict[str, pl.DataFrame]
    daily: dict[str, pl.DataFrame]
    plot_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def export(self, output_dir: str | Path, report_name: str = "factor") -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for factor_name, frame in self.daily.items():
            path = output / f"{factor_name}_base.parquet"
            frame.write_parquet(path)
            files.append(path)
        if self.daily:
            all_daily = pl.concat(
                [frame.with_columns(pl.lit(name).alias("factor")) for name, frame in self.daily.items()],
                how="diagonal_relaxed",
            )
            path = output / "result_all.parquet"
            all_daily.write_parquet(path)
            files.append(path)
        workbook_path = output / "factor_metrics_summary.xlsx"
        self._write_excel(workbook_path)
        files.append(workbook_path)
        svg_path = output / f"{report_name}.svg"
        self._write_svg(svg_path)
        files.append(svg_path)
        return files

    def _write_excel(self, path: Path) -> None:
        try:
            import xlsxwriter
        except ImportError as exc:
            raise RuntimeError(
                "Excel export requires xlsxwriter; install factor-evaluator with its report dependencies"
            ) from exc
        workbook = xlsxwriter.Workbook(path)
        header = workbook.add_format(
            {"bold": True, "font_color": "white", "bg_color": "#17365D", "border": 0}
        )
        percent = workbook.add_format({"num_format": "0.00%;[Red](0.00%);-"})
        number = workbook.add_format({"num_format": "0.0000;[Red](0.0000);-"})
        integer = workbook.add_format({"num_format": "#,##0"})
        for sheet_name, frame in self.tables.items():
            worksheet = workbook.add_worksheet(sheet_name[:31])
            worksheet.hide_gridlines(2)
            worksheet.freeze_panes(1, 0)
            for column_index, column in enumerate(frame.columns):
                worksheet.write(0, column_index, column, header)
                values = frame[column].to_list()
                for row_index, value in enumerate(values, start=1):
                    if isinstance(value, float) and not np.isfinite(value):
                        value = None
                    cell_format = number
                    lowered = column.lower()
                    if "return" in lowered or "drawdown" in lowered or "turnover" in lowered or "coverage" in lowered:
                        cell_format = percent
                    elif lowered in {"year", "lag", "bucket", "count", "n_days"}:
                        cell_format = integer
                    worksheet.write(row_index, column_index, value, cell_format)
                width = min(max(len(column) + 2, 12), 28)
                worksheet.set_column(column_index, column_index, width)
            if frame.height:
                worksheet.autofilter(0, 0, frame.height, max(0, frame.width - 1))
        workbook.close()

    def _write_svg(self, path: Path) -> None:
        from .factor_report import write_factor_report

        write_factor_report(self, path)


class FactorBacktester:
    SHEET_NAMES = {
        "longshort_summary": "01_longshort_summary",
        "lb_by_year": "02_lb_by_year",
        "coverage": "03_coverage",
        "distribution": "04_distribution",
        "factor_return_summary": "05_factor_return_summary",
        "bucket_return": "06_bucket_return",
        "ic_summary": "07_ic_summary",
        "ic_decay": "08_ic_decay",
        "style_correlation": "09_style_correlation",
        "autocorrelation": "10_autocorrelation",
    }

    def __init__(self, config: FactorBacktestConfig | None = None):
        self.config = config or FactorBacktestConfig()

    def run(self, dataset: FactorDataset) -> FactorBacktestResult:
        data = dataset.normalized()
        factors = self._filter_dates(data.factors)
        returns = self._filter_dates(data.returns)
        benchmark = self._filter_dates(data.benchmark)
        factor_columns = [c for c in factors.columns if c not in {"date", "asset"}]
        if not factor_columns:
            raise ValueError("No factor value columns found")
        joined = factors.join(returns.select("date", "asset", "return"), on=["date", "asset"])
        benchmark_map = dict(benchmark.iter_rows())
        universe_counts = self._universe_counts(data, returns)
        table_rows: dict[str, list[dict[str, Any]]] = {
            key: [] for key in self.SHEET_NAMES
        }
        bucket_exposure: dict[str, list[dict[str, Any]]] = {}
        daily_results: dict[str, pl.DataFrame] = {}
        plot_data: dict[str, Any] = {"factor_values": {}, "coverage": {}}
        for factor in factor_columns:
            daily, weights_by_date, bucket_rows = self._daily_factor(
                joined, factor, benchmark_map
            )
            daily_results[factor] = daily
            plot_frame = factors.select("date", "asset", factor).drop_nulls(factor)
            values = plot_frame[factor].to_numpy()
            stride = max(1, len(values) // 200_000)
            plot_data["factor_values"][factor] = values[::stride]
            coverage_rows = []
            for row in plot_frame.group_by("date").agg(
                pl.len().alias("count"),
                pl.col(factor).n_unique().alias("unique"),
            ).sort("date").iter_rows(named=True):
                denominator = universe_counts.get(row["date"], 0)
                if denominator:
                    coverage_rows.append({
                        "date": row["date"],
                        "coverage": row["count"] / denominator,
                        "unique": row["unique"] / denominator,
                    })
            plot_data["coverage"][factor] = pl.DataFrame(coverage_rows)
            horizon = self._horizon(daily)
            if self.config.test_basic_features:
                self._basic_tables(
                    factors, factor, universe_counts, horizon, table_rows
                )
            if self.config.test_basic_return:
                self._return_tables(
                    factors,
                    returns,
                    factor,
                    horizon,
                    daily,
                    bucket_rows,
                    table_rows,
                )
            if self.config.test_style_features and data.styles is not None:
                style_rows, exposure_rows = self._style_tables(
                    joined, data.styles, factor, weights_by_date, horizon
                )
                table_rows["style_correlation"].extend(style_rows)
                for style, rows in exposure_rows.items():
                    bucket_exposure.setdefault(style, []).extend(rows)
        tables = {
            self.SHEET_NAMES[key]: pl.DataFrame(rows)
            for key, rows in table_rows.items()
            if rows
        }
        preferred_styles = ["size", "mid_cap", "liquidity", "residual_volatility"]
        ordered_styles = [
            style for style in preferred_styles if style in bucket_exposure
        ] + sorted(style for style in bucket_exposure if style not in preferred_styles)
        for index, style in enumerate(ordered_styles, start=11):
            rows = bucket_exposure[style]
            tables[f"{index:02d}_bucket_exp_{style}"] = pl.DataFrame(rows)
        return FactorBacktestResult(
            tables=tables,
            daily=daily_results,
            plot_data=plot_data,
            metadata={
                "factor_columns": factor_columns,
                "topk": self.config.topk,
                "power": self.config.power,
                "annualization": self.config.annualization,
            },
        )

    def _filter_dates(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        if self.config.start_date:
            result = result.filter(pl.col("date") >= self.config.start_date.replace("-", ""))
        if self.config.end_date:
            result = result.filter(pl.col("date") <= self.config.end_date.replace("-", ""))
        return result

    def _universe_counts(
        self, data: FactorDataset, returns: pl.DataFrame
    ) -> dict[str, int]:
        universe = data.universe
        if universe is None:
            counts = returns.group_by("date").len()
        else:
            value_columns = [c for c in universe.columns if c not in {"date", "asset"}]
            if value_columns:
                universe = universe.filter(pl.col(value_columns[0]).cast(pl.Boolean))
            counts = self._filter_dates(universe).group_by("date").len()
        return dict(counts.select("date", "len").iter_rows())

    def _daily_factor(
        self,
        joined: pl.DataFrame,
        factor: str,
        benchmark_map: dict[str, float],
    ) -> tuple[pl.DataFrame, dict[str, dict[str, float]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        bucket_observations: list[dict[str, Any]] = []
        weights_by_date: dict[str, dict[str, float]] = {}
        previous_weights: dict[str, float] = {}
        previous_lb_weights: dict[str, float] = {}
        previous_ls_weights: dict[str, float] = {}
        clean = joined.select("date", "asset", factor, "return").drop_nulls([factor, "return"])
        for day in clean.partition_by("date", maintain_order=True):
            date = day["date"][0]
            signal = day[factor].to_numpy()
            asset_return = day["return"].to_numpy()
            assets = day["asset"].to_list()
            finite = np.isfinite(signal) & np.isfinite(asset_return)
            signal, asset_return = signal[finite], asset_return[finite]
            assets = [asset for asset, keep in zip(assets, finite) if keep]
            if signal.size < self.config.min_assets:
                continue
            ranks = rank(signal)
            quantile = (ranks - 1) / max(signal.size - 1, 1)
            bucket = np.minimum(
                (quantile * self.config.buckets).astype(int) + 1,
                self.config.buckets,
            )
            bmk = float(benchmark_map.get(date, 0.0) or 0.0)
            long_mask = bucket == self.config.buckets
            short_mask = bucket == 1
            long_return = safe_mean(asset_return[long_mask])
            short_return = safe_mean(asset_return[short_mask])
            lb_raw = long_mask.astype(float)
            lb_weights_array = lb_raw / lb_raw.sum() if lb_raw.sum() > 0 else lb_raw
            lb_weights = dict(zip(assets, lb_weights_array))
            ls_raw = np.where(long_mask, 0.5, np.where(short_mask, -0.5, 0.0))
            long_count = max(int(long_mask.sum()), 1)
            short_count = max(int(short_mask.sum()), 1)
            ls_weights_array = np.where(long_mask, 0.5 / long_count, np.where(short_mask, -0.5 / short_count, 0.0))
            ls_weights = dict(zip(assets, ls_weights_array))
            lb_turnover = 0.5 * sum(
                abs(lb_weights.get(asset, 0.0) - previous_lb_weights.get(asset, 0.0))
                for asset in set(lb_weights) | set(previous_lb_weights)
            )
            ls_turnover = 0.5 * sum(
                abs(ls_weights.get(asset, 0.0) - previous_ls_weights.get(asset, 0.0))
                for asset in set(ls_weights) | set(previous_ls_weights)
            )
            previous_lb_weights = lb_weights
            previous_ls_weights = ls_weights
            centered = signal - safe_mean(signal)
            denominator = float(np.dot(centered, centered))
            factor_return = (
                float(np.dot(centered, asset_return - safe_mean(asset_return)) / denominator)
                if denominator > 0
                else float("nan")
            )
            cutoff = 1.0 - self.config.topk
            raw_weight = np.where(
                quantile >= cutoff,
                np.power(np.clip((quantile - cutoff) / self.config.topk, 0, 1), self.config.power),
                0.0,
            )
            if raw_weight.sum() <= 0:
                raw_weight = (bucket == self.config.buckets).astype(float)
            weights = raw_weight / raw_weight.sum()
            current_weights = dict(zip(assets, weights))
            all_assets = set(previous_weights) | set(current_weights)
            turnover = 0.5 * sum(
                abs(current_weights.get(asset, 0.0) - previous_weights.get(asset, 0.0))
                for asset in all_assets
            )
            weights_by_date[date] = current_weights
            previous_weights = current_weights
            topk_return = float(np.dot(weights, asset_return))
            rows.append(
                {
                    "date": date,
                    "lb": long_return - bmk,
                    "ls": long_return - short_return,
                    "lb_tvr": lb_turnover,
                    "ls_tvr": ls_turnover,
                    "lb_limit": 0.0,
                    "ls_limit": 0.0,
                    "ic": rank_correlation(signal, asset_return),
                    "factor_return": factor_return,
                    "topk_power_lb": topk_return - bmk,
                    "topk_power_lb_tvr": turnover,
                    "topk_power_lb_limit": 0.0,
                    "topk_power_long_return": topk_return,
                    "benchmark_return": bmk,
                }
            )
            for bucket_number in range(1, self.config.buckets + 1):
                mask = bucket == bucket_number
                if mask.any():
                    bucket_observations.append(
                        {
                            "factor": factor,
                            "date": date,
                            "bucket": bucket_number,
                            "return": safe_mean(asset_return[mask]) - bmk,
                        }
                    )
        schema = {
            "date": pl.String,
            "lb": pl.Float64,
            "ls": pl.Float64,
            "lb_tvr": pl.Float64,
            "ls_tvr": pl.Float64,
            "lb_limit": pl.Float64,
            "ls_limit": pl.Float64,
            "ic": pl.Float64,
            "factor_return": pl.Float64,
            "topk_power_lb": pl.Float64,
            "topk_power_lb_tvr": pl.Float64,
            "topk_power_lb_limit": pl.Float64,
            "topk_power_long_return": pl.Float64,
            "benchmark_return": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema), weights_by_date, bucket_observations

    def _basic_tables(
        self,
        factors: pl.DataFrame,
        factor: str,
        universe_counts: dict[str, int],
        horizon: str,
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        clean = factors.select("date", "asset", factor).drop_nulls(factor)
        by_date = clean.group_by("date").agg(
            pl.len().alias("count"),
            pl.col(factor).n_unique().alias("unique"),
        )
        coverage = [
            count / universe_counts[date]
            for date, count in by_date.select("date", "count").iter_rows()
            if universe_counts.get(date, 0)
        ]
        unique = [
            count / universe_counts[date]
            for date, count in by_date.select("date", "unique").iter_rows()
            if universe_counts.get(date, 0)
        ]
        tables["coverage"].append(
            {
                "factor": factor,
                "mean_coverage": safe_mean(np.asarray(coverage)),
                "mean_unique": safe_mean(np.asarray(unique)),
            }
        )
        values = clean[factor].to_numpy()
        values = values[np.isfinite(values)]
        mean = safe_mean(values)
        std = safe_std(values, ddof=0)
        normalized = (values - mean) / std if std > 0 else np.zeros_like(values)
        tables["distribution"].append(
            {
                "factor": factor,
                "mean": mean,
                "std": std,
                "skew": safe_mean(normalized**3),
                "kurtosis": safe_mean(normalized**4) - 3.0,
                "min": float(np.min(values)) if values.size else float("nan"),
                "max": float(np.max(values)) if values.size else float("nan"),
                "p10": float(np.quantile(values, 0.1)) if values.size else float("nan"),
                "p90": float(np.quantile(values, 0.9)) if values.size else float("nan"),
                "count": int(values.size),
            }
        )
        ordered = factors.select("date", "asset", factor).sort(["asset", "date"])
        for lag in range(1, self.config.max_lag + 1):
            lagged = ordered.with_columns(
                pl.col(factor).shift(lag).over("asset").alias("_lagged")
            ).drop_nulls([factor, "_lagged"])
            tables["autocorrelation"].append(
                {
                    "factor": factor,
                    "horizon": horizon,
                    "lag": lag,
                    "autocorrelation": correlation(
                        lagged[factor].to_numpy(), lagged["_lagged"].to_numpy()
                    ),
                }
            )

    def _return_tables(
        self,
        factors: pl.DataFrame,
        returns: pl.DataFrame,
        factor: str,
        horizon: str,
        daily: pl.DataFrame,
        bucket_rows: list[dict[str, Any]],
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        for strategy, prefix in (("equal_weight", ""), ("topk_power", "topk_power_")):
            for pnl_type in ("lb", "ls"):
                if strategy == "topk_power" and pnl_type == "ls":
                    continue
                column = f"{prefix}{pnl_type}"
                values = daily[column].to_numpy()
                tables["longshort_summary"].append(
                    {
                        "factor": factor,
                        "strategy": strategy,
                        "horizon": "base",
                        "pnl_type": pnl_type,
                        "annual_return": annualized_return(values, self.config.annualization),
                        "sharpe": sharpe_ratio(values, self.config.annualization),
                        "annual_limit_return": (
                            annualized_return(daily[f"{pnl_type}_limit"].to_numpy(), self.config.annualization)
                            if strategy == "equal_weight" and f"{pnl_type}_limit" in daily.columns
                            else 0.0
                        ),
                        "avg_turnover": (
                            safe_mean(daily["topk_power_lb_tvr"].to_numpy())
                            if strategy == "topk_power"
                            else safe_mean(daily[f"{pnl_type}_tvr"].to_numpy())
                        ),
                        "max_drawdown": max_drawdown(values),
                    }
                )
            for year_frame in daily.with_columns(pl.col("date").str.slice(0, 4).alias("year")).partition_by("year"):
                tables["lb_by_year"].append(
                    {
                        "factor": factor,
                        "horizon": "base",
                        "strategy": strategy,
                        "year": int(year_frame["year"][0]),
                        "annual_lb_return": float(year_frame[f"{prefix}lb"].sum()),
                    }
                )
        factor_returns = daily["factor_return"].to_numpy()
        tables["factor_return_summary"].append(
            {
                "factor": factor,
                "annual_return": annualized_return(factor_returns, self.config.annualization),
                "annual_vol": annualized_volatility(factor_returns, self.config.annualization),
                "sharpe": sharpe_ratio(factor_returns, self.config.annualization),
            }
        )
        for bucket in range(1, self.config.buckets + 1):
            values = np.asarray(
                [row["return"] for row in bucket_rows if row["bucket"] == bucket],
                dtype=float,
            )
            tables["bucket_return"].append(
                {
                    "factor": factor,
                    "horizon": horizon,
                    "bucket": bucket,
                    "annual_excess_return": annualized_return(
                        values, self.config.annualization
                    ),
                }
            )
        ic = daily["ic"].to_numpy()
        tables["ic_summary"].append(
            {
                "factor": factor,
                "horizon": horizon,
                "ic": safe_mean(ic),
                "icir": safe_mean(ic) / safe_std(ic) if safe_std(ic) > 0 else float("nan"),
            }
        )
        joined = factors.select("date", "asset", factor).join(
            returns.select("date", "asset", "return"), on=["date", "asset"]
        ).sort(["asset", "date"])
        for lag in range(1, self.config.max_lag + 1):
            shifted = joined.with_columns(
                pl.col("return").shift(-(lag - 1)).over("asset").alias("_future")
            ).drop_nulls([factor, "_future"])
            daily_ic = []
            for day in shifted.partition_by("date", maintain_order=True):
                daily_ic.append(
                    rank_correlation(day[factor].to_numpy(), day["_future"].to_numpy())
                )
            tables["ic_decay"].append(
                {
                    "factor": factor,
                    "horizon": horizon,
                    "lag": lag,
                    "ic_decay": safe_mean(np.asarray(daily_ic)),
                }
            )

    def _style_tables(
        self,
        joined: pl.DataFrame,
        styles: pl.DataFrame,
        factor: str,
        weights_by_date: dict[str, dict[str, float]],
        horizon: str,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        style_columns = [c for c in styles.columns if c not in {"date", "asset"}]
        panel = joined.select("date", "asset", factor, "return").join(
            self._filter_dates(styles), on=["date", "asset"], how="inner"
        ).drop_nulls([factor, "return"])
        correlations = []
        exposures: dict[str, list[dict[str, Any]]] = {}
        for style in style_columns:
            signal_corr: list[float] = []
            style_return: list[float] = []
            long_style_return: list[float] = []
            bucket_values: dict[int, list[float]] = {
                bucket: [] for bucket in range(1, self.config.buckets + 1)
            }
            for day in panel.drop_nulls(style).partition_by("date", maintain_order=True):
                date = day["date"][0]
                signal = day[factor].to_numpy()
                style_value = day[style].to_numpy()
                asset_return = day["return"].to_numpy()
                signal_corr.append(correlation(signal, style_value))
                denominator = float(np.dot(style_value, style_value))
                style_return.append(
                    float(np.dot(style_value, asset_return) / denominator)
                    if denominator > 0
                    else float("nan")
                )
                weights = weights_by_date.get(date, {})
                long_style_return.append(
                    sum(
                        weights.get(asset, 0.0) * value
                        for asset, value in zip(day["asset"].to_list(), style_value)
                    )
                )
                ranks = rank(signal)
                bucket = np.minimum(
                    ((ranks - 1) / max(signal.size - 1, 1) * self.config.buckets).astype(int) + 1,
                    self.config.buckets,
                )
                for bucket_number in bucket_values:
                    mask = bucket == bucket_number
                    if mask.any():
                        bucket_values[bucket_number].append(safe_mean(style_value[mask]))
            factor_ret = []
            long_ret = []
            for day in panel.partition_by("date", maintain_order=True):
                signal = day[factor].to_numpy()
                asset_return = day["return"].to_numpy()
                centered = signal - safe_mean(signal)
                denominator = float(np.dot(centered, centered))
                factor_ret.append(
                    float(np.dot(centered, asset_return) / denominator)
                    if denominator > 0 else float("nan")
                )
                long_ret.append(
                    sum(
                        weights_by_date.get(day["date"][0], {}).get(asset, 0.0) * value
                        for asset, value in zip(day["asset"].to_list(), asset_return)
                    )
                )
            correlations.append(
                {
                    "factor": factor,
                    "style": style,
                    "signal_corr": safe_mean(np.asarray(signal_corr)),
                    "fret_corr": correlation(np.asarray(factor_ret), np.asarray(style_return)),
                    "lbret_corr": correlation(np.asarray(long_ret), np.asarray(long_style_return)),
                }
            )
            exposures[style] = [
                {
                    "factor": factor,
                    "style": style,
                    "bucket": bucket,
                    "exposure": safe_mean(np.asarray(values)),
                }
                for bucket, values in bucket_values.items()
            ]
        return correlations, exposures

    @staticmethod
    def _horizon(daily: pl.DataFrame) -> str:
        if daily.is_empty():
            return "empty"
        return f"{daily['date'].min()}_{daily['date'].max()}"



