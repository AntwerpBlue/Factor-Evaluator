from __future__ import annotations

from pathlib import Path

from factor_evaluator import (
    FactorBacktestConfig,
    FactorBacktester,
    PortfolioBacktestConfig,
    PortfolioBacktester,
    load_config,
    load_factor_dataset,
    load_portfolio_dataset,
    load_target_weights,
)


def run(config_file: str, benchmark: str) -> None:
    config = load_config(config_file)
    output_root = config.resolve(config.output["root"])

    factor_config = FactorBacktestConfig(**config.factor)
    factor_result = FactorBacktester(factor_config).run(
        load_factor_dataset(config, benchmark)
    )
    factor_result.export(
        output_root / config.output.get("single_factor_folder", "single_factor"),
        report_name="factor",
    )

    portfolio_config = PortfolioBacktestConfig(**config.portfolio)
    portfolio_result = PortfolioBacktester(portfolio_config).run(
        load_target_weights(config),
        load_portfolio_dataset(config, benchmark),
    )
    portfolio_result.export(
        output_root / config.output.get("portfolio_folder", "portfolio"),
        product_name=benchmark,
        test_name=Path(config.paths["factor_values"]).stem,
    )


if __name__ == "__main__":
    run("config/backtest.toml", "Zhishu500")
