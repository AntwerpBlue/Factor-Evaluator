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
