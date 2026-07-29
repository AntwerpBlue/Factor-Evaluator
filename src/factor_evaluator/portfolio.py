from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import minimize

from .data import PortfolioDataset, normalize_panel
from .metrics import annualized_return, annualized_volatility, safe_mean, sharpe_ratio


@dataclass(frozen=True)
class IndexEnhancementConfig:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.002
    max_weight: float = 0.03
    max_active_weight: float = 0.01
    tracking_error: float | None = None
    long_only: bool = True


class IndexEnhancementStrategy:
    """Small convex index-enhancement optimizer.

    The optimizer maximizes expected alpha while penalizing active risk and
    turnover. A full covariance matrix can be supplied; identity covariance is
    used as a deterministic fallback for examples and tests.
    """

    def __init__(self, config: IndexEnhancementConfig | None = None):
        self.config = config or IndexEnhancementConfig()

    def optimize(
        self,
        alpha: Mapping[str, float],
        benchmark_weights: Mapping[str, float],
        previous_weights: Mapping[str, float] | None = None,
        covariance: np.ndarray | None = None,
    ) -> dict[str, float]:
        assets = sorted(set(alpha) | set(benchmark_weights))
        if not assets:
            raise ValueError("alpha and benchmark_weights are empty")
        expected = np.asarray([alpha.get(asset, 0.0) for asset in assets], dtype=float)
        benchmark = np.asarray(
            [benchmark_weights.get(asset, 0.0) for asset in assets], dtype=float
        )
        if benchmark.sum() <= 0:
            benchmark = np.full(len(assets), 1.0 / len(assets))
        else:
            benchmark /= benchmark.sum()
        previous = np.asarray(
            [(previous_weights or benchmark_weights).get(asset, 0.0) for asset in assets],
            dtype=float,
        )
        previous = previous / previous.sum() if previous.sum() > 0 else benchmark.copy()
        covariance = (
            np.asarray(covariance, dtype=float)
            if covariance is not None
            else np.eye(len(assets), dtype=float)
        )
        if covariance.shape != (len(assets), len(assets)):
            raise ValueError("covariance shape does not match optimizer universe")

        def objective(weight: np.ndarray) -> float:
            active = weight - benchmark
            risk = float(active @ covariance @ active)
            turnover = np.sqrt((weight - previous) ** 2 + 1e-12).sum()
            return (
                -float(expected @ weight)
                + self.config.risk_aversion * risk
                + self.config.turnover_penalty * turnover
            )

        lower = np.maximum(
            0.0 if self.config.long_only else -self.config.max_weight,
            benchmark - self.config.max_active_weight,
        )
        upper = np.minimum(
            self.config.max_weight, benchmark + self.config.max_active_weight
        )
        if lower.sum() > 1.0 or upper.sum() < 1.0:
            raise ValueError("Weight bounds are infeasible; relax max_weight/max_active_weight")
        constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}
        ]
        if self.config.tracking_error is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weight: float(
                        self.config.tracking_error**2
                        - (weight - benchmark) @ covariance @ (weight - benchmark)
                    ),
                }
            )
        result = minimize(
            objective,
            np.clip(benchmark, lower, upper),
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Index enhancement optimization failed: {result.message}")
        return dict(zip(assets, result.x))

    def build_targets(
        self,
        signals: pl.DataFrame,
        benchmark_weights: pl.DataFrame,
        covariance_by_date: Mapping[str, np.ndarray] | None = None,
    ) -> pl.DataFrame:
        signal_panel = normalize_panel(signals)
        benchmark_panel = normalize_panel(benchmark_weights)
        signal_columns = [c for c in signal_panel.columns if c not in {"date", "asset"}]
        weight_columns = [c for c in benchmark_panel.columns if c not in {"date", "asset"}]
        if len(signal_columns) != 1 or len(weight_columns) != 1:
            raise ValueError("signals and benchmark_weights each require one value column")
        signal_column, weight_column = signal_columns[0], weight_columns[0]
        signal_dates = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day[signal_column].to_list()))
            for day in signal_panel.partition_by("date", maintain_order=True)
        }
        benchmark_dates = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day[weight_column].to_list()))
            for day in benchmark_panel.partition_by("date", maintain_order=True)
        }
        rows = []
        previous: dict[str, float] | None = None
        for date in sorted(set(signal_dates) & set(benchmark_dates)):
            weights = self.optimize(
                signal_dates[date],
                benchmark_dates[date],
                previous,
                (covariance_by_date or {}).get(date),
            )
            rows.extend(
                {"date": date, "asset": asset, "weight": weight}
                for asset, weight in weights.items()
            )
            previous = weights
        return pl.DataFrame(rows)


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    annualization: int = 240
    transaction_cost_bps: float = 8.0
    slot_weights: Mapping[str, float] = field(default_factory=lambda: {"1500_1": 1.0})
    pf_by_time: Mapping[str, float] = field(default_factory=lambda: {"1500_1": 0.4})
    max_deviation: float = 0.005
    max_holding: float = 0.005
    target_volatility: float = 0.05
    charge_initial_trade: bool = False
    rebalance_frequency: int = 1

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be positive")
        if not self.slot_weights or sum(self.slot_weights.values()) <= 0:
            raise ValueError("slot_weights must contain a positive allocation")


@dataclass
class PortfolioBacktestResult:
    portfolio: pl.DataFrame
    trades: pl.DataFrame
    pnl_daily: pl.DataFrame
    trading_pnl_by_time: pl.DataFrame
    summary: dict[str, Any]
    decomposition: pl.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def export(
        self,
        output_dir: str | Path,
        *,
        product_name: str = "ZeroIndex",
        test_name: str = "factor",
    ) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"{product_name}_intraday"
        start = self.pnl_daily["__index_level_0__"].min().strftime("%Y%m%d")
        end = self.pnl_daily["__index_level_0__"].max().strftime("%Y%m%d")
        self.metadata["chart_title"] = (
            f"{test_name}/{product_name} | maxDev={self.metadata.get('max_deviation', 0):.2%}, "
            f"maxHold={self.metadata.get('max_holding', 0):.2%}, "
            f"targetVol={self.metadata.get('target_volatility', 0):.2%} "
            f"[{start} - {end}]"
        )
        outputs = {
            f"{prefix}_port.parquet": self.portfolio,
            f"{prefix}_trade.parquet": self.trades,
            f"{prefix}_pnl_daily.parquet": self.pnl_daily,
            f"{prefix}_trading_pnl_by_time.parquet": self.trading_pnl_by_time,
        }
        files = []
        for filename, frame in outputs.items():
            path = output / filename
            frame.write_parquet(path)
            files.append(path)
        summary_path = output / "intraday_summary.json"
        summary_path.write_text(
            json.dumps({product_name: self.summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files.append(summary_path)
        config_path = output / "factor_config.json"
        config_path.write_text(
            json.dumps(
                {"test_name": test_name, "backtest_config": self.metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files.append(config_path)
        files.extend(self._write_performance_charts(output, prefix))
        if self.decomposition is not None:
            csv_path = output / f"{test_name}_{product_name}_decomp.csv"
            self.decomposition.write_csv(csv_path)
            files.append(csv_path)
            chart_path = output / f"{test_name}_{product_name}_decomp.png"
            self._write_decomposition_chart(chart_path)
            files.append(chart_path)
        done = output / "DONE.ok"
        done.write_text("ok\n", encoding="utf-8")
        files.append(done)
        return files

    def _write_performance_charts(self, output: Path, prefix: str) -> list[Path]:
        from .portfolio_report import write_performance_charts

        return write_performance_charts(self, output, prefix)

    def _write_decomposition_chart(self, path: Path) -> None:
        from .portfolio_report import write_decomposition_chart

        write_decomposition_chart(self, path)


class PortfolioBacktester:
    def __init__(self, config: PortfolioBacktestConfig | None = None):
        self.config = config or PortfolioBacktestConfig()

    def run(
        self, target_weights: pl.DataFrame, dataset: PortfolioDataset
    ) -> PortfolioBacktestResult:
        data = dataset.normalized()
        weights = normalize_panel(target_weights)
        value_columns = [c for c in weights.columns if c not in {"date", "asset"}]
        if "weight" not in weights.columns:
            if len(value_columns) != 1:
                raise ValueError("target_weights require one value column named 'weight'")
            weights = weights.rename({value_columns[0]: "weight"})
        weights = weights.with_columns(pl.col("weight").cast(pl.Float64))
        targets = self._targets_by_date(weights)
        returns_by_date = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day["return"].to_list()))
            for day in data.returns.partition_by("date", maintain_order=True)
        }
        benchmark = dict(data.benchmark.iter_rows())
        dates = sorted(set(returns_by_date) & set(benchmark))
        if not dates:
            raise ValueError("No overlapping return and benchmark dates")
        assets = sorted(
            set(data.returns["asset"].to_list())
            | {asset for target in targets.values() for asset in target}
        )
        portfolio_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        pnl_rows: list[dict[str, Any]] = []
        slot_rows: list[dict[str, Any]] = []
        previous: dict[str, float] = {}
        previous_returns: dict[str, float] = {}
        latest_target: dict[str, float] = {}
        cost_rate = self.config.transaction_cost_bps / 10_000.0
        slot_total = float(sum(self.config.slot_weights.values()))
        for index, date in enumerate(dates):
            daily_returns = returns_by_date[date]
            holding_pnl = (
                sum(previous.get(asset, 0.0) * daily_returns.get(asset, 0.0) for asset in assets)
                if previous
                else 0.0
            )
            if previous:
                denominator = 1.0 + holding_pnl
                pretrade = {
                    asset: previous.get(asset, 0.0)
                    * (1.0 + daily_returns.get(asset, 0.0))
                    / denominator
                    for asset in assets
                }
            else:
                pretrade = {}
            if index % self.config.rebalance_frequency == 0 and date in targets:
                latest_target = targets[date]
            target = latest_target or pretrade
            trade = {
                asset: target.get(asset, 0.0) - pretrade.get(asset, 0.0)
                for asset in assets
            }
            turnover = 0.5 * sum(abs(value) for value in trade.values())
            trading_pnl = -cost_rate * sum(abs(value) for value in trade.values())
            if index == 0 and not self.config.charge_initial_trade:
                trade = {asset: 0.0 for asset in assets}
                turnover = 0.0
                trading_pnl = 0.0
            bmk_ret = float(benchmark[date] or 0.0)
            holding_alpha = holding_pnl - bmk_ret
            alpha = holding_alpha + trading_pnl
            timestamp = self._timestamp(date)
            for slot, allocation in self.config.slot_weights.items():
                fraction = allocation / slot_total
                portfolio_rows.append({
                    **{asset: target.get(asset, 0.0) for asset in assets},
                    "trade_ts": timestamp,
                    "slot_key": slot,
                })
                trade_rows.append({
                    **{asset: value * fraction for asset, value in trade.items()},
                    "trade_ts": timestamp,
                    "slot_key": slot,
                })
            pnl_rows.append(
                {
                    "holding_pnl": holding_pnl,
                    "bmk_ret": bmk_ret,
                    "holding_alpha": holding_alpha,
                    "trading_pnl": trading_pnl,
                    "alpha": alpha,
                    "__index_level_0__": self._date(date),
                }
            )
            slot_rows.append(
                {
                    **{
                        slot: trading_pnl * allocation / slot_total
                        for slot, allocation in self.config.slot_weights.items()
                    },
                    "__index_level_0__": self._date(date),
                }
            )
            previous = target
            previous_returns = daily_returns
        pnl = pl.DataFrame(pnl_rows)
        portfolio_frame = pl.DataFrame(portfolio_rows)
        trade_frame = pl.DataFrame(trade_rows)
        slot_frame = pl.DataFrame(slot_rows)
        from .portfolio_report import build_portfolio_summary
        summary = build_portfolio_summary(
            pnl, trade_frame, self.config.annualization, dict(self.config.pf_by_time)
        )
        decomposition = self._decompose(
            dates, previous_targets=self._lag_targets(dates, targets), dataset=data, pnl=pnl
        )
        return PortfolioBacktestResult(
            portfolio=portfolio_frame,
            trades=trade_frame,
            pnl_daily=pnl,
            trading_pnl_by_time=slot_frame,
            summary=summary,
            decomposition=decomposition,
            metadata=asdict(self.config),
        )

    @staticmethod
    def _targets_by_date(weights: pl.DataFrame) -> dict[str, dict[str, float]]:
        result = {}
        for day in weights.partition_by("date", maintain_order=True):
            raw = np.asarray(day["weight"].to_list(), dtype=float)
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite target weight on {day['date'][0]}")
            total = raw.sum()
            if total <= 0:
                raise ValueError(f"Target weights must sum positive on {day['date'][0]}")
            normalized = raw / total
            result[day["date"][0]] = dict(zip(day["asset"].to_list(), normalized))
        return result

    @staticmethod
    def _date(date: str):
        from datetime import datetime

        return datetime.strptime(date, "%Y%m%d")

    @classmethod
    def _timestamp(cls, date: str):
        return cls._date(date).replace(hour=15)

    @staticmethod
    def _lag_targets(
        dates: list[str], targets: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        lagged: dict[str, dict[str, float]] = {}
        previous: dict[str, float] = {}
        latest: dict[str, float] = {}
        for date in dates:
            lagged[date] = previous
            if date in targets:
                latest = targets[date]
            previous = latest
        return lagged

    def _decompose(
        self,
        dates: list[str],
        previous_targets: dict[str, dict[str, float]],
        dataset: PortfolioDataset,
        pnl: pl.DataFrame,
    ) -> pl.DataFrame | None:
        if dataset.styles is None and dataset.industry_exposures is None:
            return None
        style_columns = (
            [c for c in dataset.styles.columns if c not in {"date", "asset"}]
            if dataset.styles is not None else []
        )
        styles_by_date = (
            {day["date"][0]: day for day in dataset.styles.partition_by("date", maintain_order=True)}
            if dataset.styles is not None else {}
        )
        industry_by_date = (
            {day["date"][0]: day for day in dataset.industry_exposures.partition_by("date", maintain_order=True)}
            if dataset.industry_exposures is not None else {}
        )
        returns_by_date = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day["return"].to_list()))
            for day in dataset.returns.partition_by("date", maintain_order=True)
        }
        benchmark_weights = {}
        if dataset.benchmark_weights is not None:
            weight_columns = [
                c for c in dataset.benchmark_weights.columns if c not in {"date", "asset"}
            ]
            weight_column = "weight" if "weight" in weight_columns else weight_columns[0]
            benchmark_weights = {
                day["date"][0]: dict(zip(day["asset"].to_list(), day[weight_column].to_list()))
                for day in dataset.benchmark_weights.partition_by("date", maintain_order=True)
            }
        style_return_map = self._style_return_map(dataset, style_columns)
        industry_return_map = self._industry_return_map(dataset)
        rows = []
        for index, date in enumerate(dates):
            pnl_row = pnl.row(index, named=True)
            style_frame = styles_by_date.get(date)
            portfolio_weights = previous_targets.get(date, {})
            benchmark_date = benchmark_weights.get(date, {})
            style_contributions = {style: 0.0 for style in style_columns}
            if style_frame is not None:
                for style in style_columns:
                    exposure = sum(
                        (portfolio_weights.get(asset, 0.0) - benchmark_date.get(asset, 0.0))
                        * float(value or 0.0)
                        for asset, value in zip(
                            style_frame["asset"].to_list(), style_frame[style].to_list()
                        )
                    )
                    style_contributions[style] = exposure * style_return_map.get(
                        (date, style), 0.0
                    )
            style_ret = sum(style_contributions.values())
            industry_ret = self._industry_contribution(
                date, industry_by_date.get(date), industry_return_map,
                portfolio_weights, benchmark_date,
            )
            pure_before = pnl_row["holding_alpha"] - style_ret - industry_ret
            rows.append(
                {
                    "TRADE_DT": self._date(date).date(),
                    "SurplusRet": pnl_row["alpha"],
                    "StyleRet": style_ret,
                    "IndustryRet": industry_ret,
                    "pureAlpha_before_fee": pure_before,
                    "pureAlpha_after_fee": pure_before + pnl_row["trading_pnl"],
                    **style_contributions,
                }
            )
        return pl.DataFrame(rows)

    @staticmethod
    def _industry_return_map(
        dataset: PortfolioDataset,
    ) -> dict[tuple[str, str], float]:
        frame = dataset.industry_returns
        if frame is None:
            return {}
        date_column = "date" if "date" in frame.columns else "TRADE_DT"
        if {"industry", "return"}.issubset(frame.columns):
            return {
                (str(row[date_column]).replace("-", ""), str(row["industry"])): float(row["return"] or 0.0)
                for row in frame.iter_rows(named=True)
            }
        industry_columns = [c for c in frame.columns if c != date_column]
        return {
            (str(row[date_column]).replace("-", ""), industry): float(row[industry] or 0.0)
            for row in frame.iter_rows(named=True)
            for industry in industry_columns
        }

    @staticmethod
    def _industry_contribution(
        date: str,
        frame: pl.DataFrame | None,
        return_map: dict[tuple[str, str], float],
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> float:
        if frame is None:
            return 0.0
        if "industry" in frame.columns:
            return sum(
                (portfolio_weights.get(asset, 0.0) - benchmark_weights.get(asset, 0.0))
                * return_map.get((date, str(industry)), 0.0)
                for asset, industry in zip(
                    frame["asset"].to_list(), frame["industry"].to_list()
                )
            )
        result = 0.0
        for industry in [c for c in frame.columns if c not in {"date", "asset"}]:
            exposure = sum(
                (portfolio_weights.get(asset, 0.0) - benchmark_weights.get(asset, 0.0))
                * float(value or 0.0)
                for asset, value in zip(
                    frame["asset"].to_list(), frame[industry].to_list()
                )
            )
            result += exposure * return_map.get((date, industry), 0.0)
        return result

    @staticmethod
    def _style_return_map(
        dataset: PortfolioDataset, style_columns: list[str]
    ) -> dict[tuple[str, str], float]:
        if not style_columns:
            return {}
        if dataset.style_returns is not None:
            frame = dataset.style_returns
            date_column = "date" if "date" in frame.columns else "TRADE_DT"
            return {
                (str(row[date_column]).replace("-", ""), style): float(row[style] or 0.0)
                for row in frame.iter_rows(named=True)
                for style in style_columns
                if style in row
            }
        joined = dataset.returns.join(dataset.styles, on=["date", "asset"], how="inner")
        result: dict[tuple[str, str], float] = {}
        for day in joined.partition_by("date", maintain_order=True):
            y = day["return"].fill_null(0).to_numpy()
            available = [style for style in style_columns if style in day.columns]
            if not available:
                continue
            x = np.column_stack([day[style].fill_null(0).to_numpy() for style in available])
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            for style, coefficient in zip(available, coefficients):
                result[(day["date"][0], style)] = float(coefficient)
        return result







