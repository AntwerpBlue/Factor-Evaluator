from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factor_evaluator import (
    FactorBacktestConfig,
    FactorBacktester,
    FactorDataset,
    PortfolioBacktestConfig,
    PortfolioBacktester,
    PortfolioDataset,
    load_config,
)
from factor_evaluator.data import normalize_panel
from factor_evaluator.config import BacktestFileConfig, load_factor_dataset


def full_synthetic(n_dates: int = 24, n_assets: int = 10):
    rng = np.random.default_rng(11)
    dates = [
        str(value).replace("-", "")
        for value in np.busday_offset("2024-01-02", np.arange(n_dates))
    ]
    assets = [f"{index:06d}.SZ" for index in range(n_assets)]
    factors, returns, benchmark, styles, weights = [], [], [], [], []
    for date in dates:
        signal = rng.normal(size=n_assets)
        size = rng.normal(size=n_assets)
        realized = 0.004 * signal + rng.normal(0, 0.01, n_assets)
        benchmark.append({"date": date, "benchmark_return": float(realized.mean())})
        target = np.exp(signal * 0.1)
        target /= target.sum()
        for index, code in enumerate(assets):
            factors.append({"date": date, "code": code, "alpha": signal[index]})
            returns.append({"date": date, "code": code, "return": realized[index]})
            styles.append(
                {
                    "date": date,
                    "code": code,
                    "size": size[index],
                    "mid_cap": -size[index] ** 2,
                    "liquidity": -size[index],
                    "residual_volatility": abs(size[index]),
                }
            )
            weights.append({"date": date, "code": code, "weight": target[index]})
    return tuple(
        pl.DataFrame(rows)
        for rows in (factors, returns, benchmark, styles, weights)
    )


def test_input_contract_and_config_placeholder():
    valid = pl.DataFrame(
        {"date": ["20250102"], "code": ["000001.SZ"], "alpha": [1.0]}
    )
    assert normalize_panel(valid).columns[:2] == ["date", "asset"]
    with pytest.raises(ValueError, match="YYYYMMDD"):
        normalize_panel(valid.with_columns(pl.lit("2025-01-02").alias("date")))
    with pytest.raises(ValueError, match="security codes"):
        normalize_panel(valid.with_columns(pl.lit("AAPL").alias("code")))
    config = load_config("config/backtest.example.toml")
    with pytest.raises(ValueError, match="placeholder"):
        config.resolve(config.paths["factor_values"])


def test_factor_config_loads_tradability_mask(tmp_path):
    factors = pl.DataFrame(
        {"date": ["20250102"], "code": ["000001.SZ"], "alpha": [1.0]}
    )
    returns = pl.DataFrame(
        {"date": ["20250102"], "code": ["000001.SZ"], "return": [0.01]}
    )
    benchmark = pl.DataFrame(
        {"date": ["20250102"], "benchmark_return": [0.0]}
    )
    tradability = pl.DataFrame(
        {"date": ["20250102"], "code": ["000001.SZ"], "limit_label": [1]}
    )
    for name, frame in {
        "factors.parquet": factors,
        "returns.parquet": returns,
        "benchmark.parquet": benchmark,
        "tradability.parquet": tradability,
    }.items():
        frame.write_parquet(tmp_path / name)

    config = BacktestFileConfig(
        path=tmp_path / "backtest.toml",
        paths={
            "factor_values": "factors.parquet",
            "asset_forward_returns": "returns.parquet",
            "tradability": "tradability.parquet",
        },
        benchmarks={"ZeroIndex": {"returns": "benchmark.parquet"}},
        factor={},
        portfolio={},
        output={},
    )

    dataset = load_factor_dataset(config, "ZeroIndex").normalized()
    assert dataset.limit_labels is not None
    assert dataset.limit_labels["limit_label"].to_list() == [1]


def test_full_factor_svg_contains_reference_panels(tmp_path):
    factors, returns, benchmark, styles, _ = full_synthetic()
    result = FactorBacktester(
        FactorBacktestConfig(buckets=5, max_lag=3)
    ).run(FactorDataset(factors, returns, benchmark, styles=styles))
    result.export(tmp_path, report_name="alpha")
    svg = (tmp_path / "alpha.svg").read_text(encoding="utf-8")
    for title in (
        "Cumulative Return",
        "LB-return by year",
        "PowerRank LB base",
        "Cumulative Factor Return",
        "Alpha Bucket Return",
        "IC decay profile",
        "Coverage",
        "Distribution",
        "Factor Auto-correlation",
        "Style Correlation",
        "Bucket size",
        "Bucket mid_cap",
        "Bucket liquidity",
        "Bucket residual_volatility",
    ):
        assert title in svg


def test_portfolio_full_metrics_and_multiple_slots(tmp_path):
    _, returns, benchmark, styles, weights = full_synthetic()
    result = PortfolioBacktester(
        PortfolioBacktestConfig(
            slot_weights={"1455_1": 0.6, "1500_1": 0.4},
            pf_by_time={"1455_1": 0.3, "1500_1": 0.4},
        )
    ).run(weights, PortfolioDataset(returns, benchmark, styles=styles))
    assert result.trades.height == benchmark.height * 2
    assert set(result.summary) >= {
        "roi",
        "vol",
        "sharpe",
        "holding_alpha",
        "trading",
        "calmar",
        "max_drawdown",
        "max_drawdown_start",
        "max_drawdown_end",
        "annual_turnover",
        "turnover_by_time",
        "pf_by_time",
    }
    result.export(tmp_path, product_name="Zhishu500", test_name="alpha")
    assert (tmp_path / "Zhishu500_intraday_alpha_cum.png").stat().st_size > 10_000
    assert (tmp_path / "alpha_Zhishu500_decomp.png").stat().st_size > 10_000
