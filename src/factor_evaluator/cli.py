"""Dispatch all operations through the repository's single entry script."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys


COMMANDS = {
    "factor": "factor",
    "pipeline": "run_end_to_end_backtest",
    "prepare-data": "build_local_backtest_data",
    "fetch-wind": "fetch_wind_index_data",
    "risk-model": "build_risk_model_data",
    "build-targets": "build_index_enhancement_targets",
    "small-account": "build_small_account_enhancement_portfolio",
}


def run_config(config_file: str, benchmark: str, task: str) -> None:
    from .config import load_config, load_factor_dataset, load_portfolio_dataset, load_target_weights
    from .factor import FactorBacktestConfig, FactorBacktester
    from .portfolio import PortfolioBacktestConfig, PortfolioBacktester

    config = load_config(Path(config_file).expanduser())
    output = config.resolve(config.output["root"])
    if task in {"factor", "both"}:
        result = FactorBacktester(FactorBacktestConfig(**config.factor)).run(
            load_factor_dataset(config, benchmark)
        )
        result.export(output / config.output.get("single_factor_folder", "single_factor"))
    if task in {"portfolio", "both"}:
        result = PortfolioBacktester(PortfolioBacktestConfig(**config.portfolio)).run(
            load_target_weights(config), load_portfolio_dataset(config, benchmark)
        )
        result.export(
            output / config.output.get("portfolio_folder", "portfolio"),
            product_name=benchmark,
            test_name=Path(config.paths["target_weights"]).stem,
        )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in COMMANDS:
        module = import_module(f"factor_evaluator._commands.{COMMANDS[args[0]]}")
        module.main(args[1:])
        return
    # Retain the previous --factor invocation through the same entry script.
    if any(arg == "--factor" or arg.startswith("--factor=") for arg in args):
        import_module("factor_evaluator._commands.factor").main(args)
        return
    parser = argparse.ArgumentParser(
        description="Factor Evaluator：统一回测入口",
        epilog=("直接因子文件：run_backtest.py factor --help；"
                "数据适配与原有流程：" + ", ".join(k for k in COMMANDS if k != "factor") +
                "。所有操作均通过本脚本调用。"),
    )
    parser.add_argument("--config", required=True, help="TOML 配置文件")
    parser.add_argument("--task", choices=("factor", "portfolio", "both"), default="factor")
    parser.add_argument("--benchmark", default="Zhishu500", help="配置中的基准名称")
    options = parser.parse_args(args)
    run_config(options.config, options.benchmark, options.task)
