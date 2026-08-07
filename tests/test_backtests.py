from __future__ import annotations

import numpy as np
import polars as pl

from factor_evaluator import (
    FactorBacktestConfig,
    FactorBacktester,
    FactorDataset,
    IndexEnhancementConfig,
    IndexEnhancementStrategy,
    PortfolioBacktestConfig,
    PortfolioBacktester,
    PortfolioDataset,
)


def synthetic(seed: int = 1, n_dates: int = 35, n_assets: int = 12):
    rng = np.random.default_rng(seed)
    dates = [str(date).replace("-", "") for date in np.busday_offset("2024-01-02", np.arange(n_dates))]
    assets = [f"{index:06d}.SZ" for index in range(n_assets)]
    factor_rows, return_rows, benchmark_rows, style_rows, weight_rows = [], [], [], [], []
    for date in dates:
        signal = rng.normal(size=n_assets)
        style = rng.normal(size=n_assets)
        returns = 0.004 * signal + 0.001 * style + rng.normal(0, 0.008, n_assets)
        benchmark_rows.append({"date": date, "benchmark_return": float(returns.mean())})
        raw_weight = np.exp(signal * 0.2)
        raw_weight /= raw_weight.sum()
        for i, asset in enumerate(assets):
            factor_rows.append({"date": date, "header": asset, "alpha": signal[i]})
            return_rows.append({"date": date, "header": asset, "return": returns[i]})
            style_rows.append({"date": date, "header": asset, "size": style[i]})
            weight_rows.append({"date": date, "header": asset, "weight": raw_weight[i]})
    return tuple(
        pl.DataFrame(rows)
        for rows in (factor_rows, return_rows, benchmark_rows, style_rows, weight_rows)
    )


def test_factor_schema_and_predictive_ic(tmp_path):
    factors, returns, benchmark, styles, _ = synthetic()
    result = FactorBacktester(
        FactorBacktestConfig(buckets=6, max_lag=3)
    ).run(FactorDataset(factors, returns, benchmark, styles=styles))
    assert set(result.daily["alpha"].columns) == {
        "date",
        "lb",
        "ls",
        "lb_tvr",
        "ls_tvr",
        "lb_limit",
        "ls_limit",
        "ic",
        "factor_return",
        "topk_power_lb",
        "topk_power_lb_tvr",
        "topk_power_lb_limit",
        "topk_power_long_return",
        "benchmark_return",
    }
    assert result.tables["07_ic_summary"]["ic"][0] > 0
    assert "14_bucket_exp_size" not in result.tables
    assert any(name.endswith("bucket_exp_size") for name in result.tables)
    files = result.export(tmp_path, report_name="alpha")
    assert {path.suffix for path in files} >= {".xlsx", ".svg", ".parquet"}


def test_portfolio_accounting_and_export(tmp_path):
    factors, returns, benchmark, styles, weights = synthetic()
    result = PortfolioBacktester(
        PortfolioBacktestConfig(transaction_cost_bps=10, slot_weights={"1500_1": 1.0})
    ).run(weights, PortfolioDataset(returns, benchmark, styles=styles))
    pnl = result.pnl_daily
    np.testing.assert_allclose(
        pnl["alpha"].to_numpy(),
        pnl["holding_alpha"].to_numpy() + pnl["trading_pnl"].to_numpy(),
    )
    assert result.portfolio.height == benchmark.height
    assert result.trades.columns[-2:] == ["trade_ts", "slot_key"]
    assert result.summary["n_days"] == benchmark.height
    files = result.export(tmp_path, product_name="ZeroIndex", test_name="alpha")
    names = {path.name for path in files}
    assert "ZeroIndex_intraday_port.parquet" in names
    assert "intraday_summary.json" in names
    assert "alpha_ZeroIndex_decomp.csv" in names


def test_portfolio_targets_earn_same_date_returns():
    weights = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "weight": 1.0},
            {"date": "20240102", "header": "000002", "weight": 0.0},
            {"date": "20240103", "header": "000001", "weight": 0.0},
            {"date": "20240103", "header": "000002", "weight": 1.0},
        ]
    )
    returns = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "return": 0.10},
            {"date": "20240102", "header": "000002", "return": 0.00},
            {"date": "20240103", "header": "000001", "return": 0.00},
            {"date": "20240103", "header": "000002", "return": 0.20},
        ]
    )
    benchmark = pl.DataFrame(
        [
            {"date": "20240102", "benchmark_return": 0.0},
            {"date": "20240103", "benchmark_return": 0.0},
        ]
    )
    result = PortfolioBacktester(PortfolioBacktestConfig(transaction_cost_bps=0)).run(
        weights, PortfolioDataset(returns, benchmark)
    )

    np.testing.assert_allclose(result.pnl_daily["holding_pnl"].to_numpy(), [0.10, 0.20])


def test_index_enhancement_constraints():
    assets = [f"A{i}" for i in range(8)]
    benchmark = {asset: 1 / len(assets) for asset in assets}
    alpha = {asset: float(i) for i, asset in enumerate(assets)}
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(max_weight=0.25, max_active_weight=0.08)
    )
    weight = strategy.optimize(alpha, benchmark)
    assert abs(sum(weight.values()) - 1.0) < 1e-8
    assert min(weight.values()) >= 0
    assert max(weight.values()) <= 0.25 + 1e-8
    assert max(abs(weight[asset] - benchmark[asset]) for asset in assets) <= 0.08 + 1e-8

def test_index_enhancement_max_turnover_cap():
    assets = [f"A{i}" for i in range(8)]
    benchmark = {asset: 1 / len(assets) for asset in assets}
    previous = benchmark.copy()
    alpha = {asset: float(i) for i, asset in enumerate(assets)}
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(
            max_weight=0.25,
            max_active_weight=0.08,
            turnover_penalty=0.0,
            max_turnover=0.05,
        )
    )
    weight = strategy.optimize(alpha, benchmark, previous_weights=previous)
    one_way_turnover = 0.5 * sum(abs(weight[asset] - previous[asset]) for asset in assets)
    assert abs(sum(weight.values()) - 1.0) < 1e-8
    assert one_way_turnover <= 0.0501

def test_index_enhancement_pf_gate_blocks_uneconomic_turnover():
    benchmark = {"A": 0.5, "B": 0.5}
    previous = benchmark.copy()
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(
            risk_aversion=0.0,
            turnover_penalty=0.0,
            max_weight=1.0,
            max_active_weight=0.5,
            turnover_cost_bps=8.0,
            turnover_cost_alpha_fraction=0.9,
        )
    )

    weight = strategy.optimize({"A": 0.001, "B": 0.0}, benchmark, previous_weights=previous)

    np.testing.assert_allclose([weight["A"], weight["B"]], [0.5, 0.5], atol=1e-8)


def test_index_enhancement_pf_gate_allows_economic_turnover():
    benchmark = {"A": 0.5, "B": 0.5}
    previous = benchmark.copy()
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(
            risk_aversion=0.0,
            turnover_penalty=0.0,
            max_weight=1.0,
            max_active_weight=0.5,
            turnover_cost_bps=8.0,
            turnover_cost_alpha_fraction=0.9,
        )
    )

    weight = strategy.optimize({"A": 0.01, "B": 0.0}, benchmark, previous_weights=previous)
    delta = {asset: weight[asset] - previous[asset] for asset in weight}
    traded = sum(abs(value) for value in delta.values())
    cost = 8.0 / 10_000.0 * traded
    alpha_gain = sum({"A": 0.01, "B": 0.0}[asset] * delta[asset] for asset in delta)

    assert weight["A"] > previous["A"]
    assert cost <= 0.9 * alpha_gain + 1e-12


def test_build_targets_uses_drifted_previous_weights_for_turnover_cap():
    signals = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "alpha": 10.0},
            {"date": "20240102", "header": "000002", "alpha": 0.0},
            {"date": "20240103", "header": "000001", "alpha": 0.0},
            {"date": "20240103", "header": "000002", "alpha": 10.0},
        ]
    )
    benchmark = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "weight": 0.5},
            {"date": "20240102", "header": "000002", "weight": 0.5},
            {"date": "20240103", "header": "000001", "weight": 0.5},
            {"date": "20240103", "header": "000002", "weight": 0.5},
        ]
    )
    returns = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "return": 1.0},
            {"date": "20240102", "header": "000002", "return": 0.0},
        ]
    )
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(
            max_weight=0.8,
            max_active_weight=0.5,
            turnover_penalty=0.0,
            max_turnover=0.01,
        )
    )
    targets = strategy.build_targets(signals, benchmark, asset_returns=returns)
    day1 = dict(targets.filter(pl.col("date") == "20240102").select("asset", "weight").iter_rows())
    day2 = dict(targets.filter(pl.col("date") == "20240103").select("asset", "weight").iter_rows())
    day2_returns = {"000001": 1.0, "000002": 0.0}
    holding_pnl = sum(day1.get(asset, 0.0) * day2_returns.get(asset, 0.0) for asset in day1)
    pretrade = {
        asset: day1.get(asset, 0.0) * (1.0 + day2_returns.get(asset, 0.0)) / (1.0 + holding_pnl)
        for asset in day1
    }
    one_way_turnover = 0.5 * sum(
        abs(day2.get(asset, 0.0) - pretrade.get(asset, 0.0))
        for asset in set(day2) | set(pretrade)
    )
    assert one_way_turnover <= 0.0101

def test_index_enhancement_linear_path_when_risk_is_disabled():
    assets = [f"A{i}" for i in range(6)]
    benchmark = {asset: 1 / len(assets) for asset in assets}
    alpha = {asset: float(i) for i, asset in enumerate(assets)}
    strategy = IndexEnhancementStrategy(
        IndexEnhancementConfig(
            risk_aversion=0.0,
            turnover_penalty=0.0,
            max_weight=0.30,
            max_active_weight=0.20,
        )
    )
    weight = strategy.optimize(alpha, benchmark)
    assert abs(sum(weight.values()) - 1.0) < 1e-8
    assert weight["A5"] >= weight["A0"]
    assert max(weight.values()) <= 0.30 + 1e-8

def test_portfolio_backtest_starts_at_first_target_date():
    weights = pl.DataFrame(
        [
            {"date": "20240103", "header": "000001", "weight": 1.0},
            {"date": "20240103", "header": "000002", "weight": 0.0},
        ]
    )
    returns = pl.DataFrame(
        [
            {"date": "20240102", "header": "000001", "return": 0.50},
            {"date": "20240102", "header": "000002", "return": 0.00},
            {"date": "20240103", "header": "000001", "return": 0.10},
            {"date": "20240103", "header": "000002", "return": 0.00},
        ]
    )
    benchmark = pl.DataFrame(
        [
            {"date": "20240102", "benchmark_return": 0.0},
            {"date": "20240103", "benchmark_return": 0.0},
        ]
    )
    result = PortfolioBacktester(PortfolioBacktestConfig(transaction_cost_bps=0)).run(
        weights, PortfolioDataset(returns, benchmark)
    )
    assert result.summary["n_days"] == 1
    assert result.pnl_daily["__index_level_0__"][0].strftime("%Y%m%d") == "20240103"
    np.testing.assert_allclose(result.pnl_daily["holding_pnl"].to_numpy(), [0.10])
