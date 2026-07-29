from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from factor_evaluator import (
    FactorBacktestConfig,
    FactorBacktester,
    FactorDataset,
    PortfolioBacktestConfig,
    PortfolioBacktester,
    PortfolioDataset,
)


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = np.busday_offset("2024-01-02", np.arange(80)).astype("datetime64[D]")
    date_text = [str(date).replace("-", "") for date in dates]
    assets = [f"{index:06d}.SZ" for index in range(1, 31)]
    factor_rows, return_rows, style_rows, weight_rows = [], [], [], []
    benchmark_rows = []
    for date in date_text:
        signal = rng.normal(size=len(assets))
        size = rng.normal(size=len(assets))
        residual = rng.normal(scale=0.01, size=len(assets))
        returns = 0.002 * (signal - signal.mean()) + 0.001 * size + residual
        benchmark_rows.append({"date": date, "benchmark_return": float(returns.mean())})
        positive = np.exp(0.6 * signal)
        weights = positive / positive.sum()
        for index, asset in enumerate(assets):
            factor_rows.append({"date": date, "header": asset, "alpha": signal[index]})
            return_rows.append({"date": date, "header": asset, "return": returns[index]})
            style_rows.append({"date": date, "header": asset, "size": size[index]})
            weight_rows.append({"date": date, "header": asset, "weight": weights[index]})
    return (
        pl.DataFrame(factor_rows),
        pl.DataFrame(return_rows),
        pl.DataFrame(benchmark_rows),
        pl.DataFrame(style_rows),
        pl.DataFrame(weight_rows),
    )


def main() -> None:
    factors, returns, benchmark, styles, weights = make_data()
    output = Path("synthetic_report")
    factor_result = FactorBacktester(
        FactorBacktestConfig(buckets=10, max_lag=5)
    ).run(FactorDataset(factors, returns, benchmark, styles=styles))
    factor_result.export(output / "single_factor", report_name="alpha")
    portfolio_result = PortfolioBacktester(
        PortfolioBacktestConfig(slot_weights={"1500_1": 0.4, "1455_1": 0.6})
    ).run(
        weights,
        PortfolioDataset(
            returns=returns,
            benchmark=benchmark,
            styles=styles,
        ),
    )
    portfolio_result.export(
        output / "portfolio", product_name="ZeroIndex", test_name="alpha"
    )
    print(f"Generated reports under {output.resolve()}")


if __name__ == "__main__":
    main()

