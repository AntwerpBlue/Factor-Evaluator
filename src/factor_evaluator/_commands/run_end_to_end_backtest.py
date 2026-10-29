"""Run end-to-end factor and index-enhancement portfolio backtests.

The script keeps reusable intermediate parquet files under cache-root/local,
cache-root/risk_*, and cache-root/targets, then writes reports under output-root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]

from factor_evaluator import (
    FactorBacktestConfig,
    FactorBacktester,
    FactorDataset,
    PortfolioBacktestConfig,
    PortfolioBacktester,
    PortfolioDataset,
)


LOGGER = logging.getLogger("run_end_to_end_backtest")

BENCHMARK_FILES = {
    "ZeroIndex": (None, None, "ZeroIndex"),
    "zeroindex": (None, None, "ZeroIndex"),
    "300": ("benchmark_300_returns.parquet", "benchmark_300_weights.parquet", "Zhishu300"),
    "500": ("benchmark_500_returns.parquet", "benchmark_500_weights.parquet", "Zhishu500"),
    "1000": ("benchmark_1000_returns.parquet", "benchmark_1000_weights.parquet", "Zhishu1000"),
    "2000": ("benchmark_2000_returns.parquet", "benchmark_2000_weights.parquet", "Zhishu2000"),
    "Zhishu300": ("benchmark_300_returns.parquet", "benchmark_300_weights.parquet", "Zhishu300"),
    "Zhishu500": ("benchmark_500_returns.parquet", "benchmark_500_weights.parquet", "Zhishu500"),
    "Zhishu1000": ("benchmark_1000_returns.parquet", "benchmark_1000_weights.parquet", "Zhishu1000"),
    "Zhishu2000": ("benchmark_2000_returns.parquet", "benchmark_2000_weights.parquet", "Zhishu2000"),
}

DEFAULT_PRODUCTS = ("ZeroIndex", "300", "500", "1000", "2000")

LOCAL_CACHE_FILES = [
    "asset_forward_returns.parquet",
    "universe.parquet",
    "tradability.parquet",
    "style_exposures.parquet",
    "industry_exposures.parquet",
    "security_master.parquet",
]

RISK_CACHE_FILES = [
    "style_factor_returns.parquet",
    "industry_factor_returns.parquet",
    "factor_covariance.parquet",
    "specific_risk.parquet",
]


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def run_command(command: list[str]) -> None:
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run([command[0], str(ROOT / "run_backtest.py"), *command[1:]], cwd=ROOT, check=True)


def all_exist(directory: Path, names: list[str]) -> bool:
    return all((directory / name).exists() for name in names)


def load_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def filter_dates(frame: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    return frame.filter(pl.col("date").cast(pl.String).is_between(pl.lit(start), pl.lit(end)))


def parquet_date_bounds(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    frame = (
        pl.scan_parquet(path)
        .select(
            pl.col("date").cast(pl.String).min().alias("min_date"),
            pl.col("date").cast(pl.String).max().alias("max_date"),
        )
        .collect()
    )
    if frame.is_empty() or frame["min_date"][0] is None or frame["max_date"][0] is None:
        return None
    return str(frame["min_date"][0]), str(frame["max_date"][0])


def cache_date_bounds(directory: Path, names: list[str]) -> tuple[str, str] | None:
    bounds = []
    for name in names:
        item = parquet_date_bounds(directory / name)
        if item is None:
            return None
        bounds.append(item)
    return max(bound[0] for bound in bounds), min(bound[1] for bound in bounds)


def cache_covers(directory: Path, names: list[str], start: str, end: str, *, require_start: bool = True) -> bool:
    bounds = cache_date_bounds(directory, names)
    if bounds is None:
        return False
    min_date, max_date = bounds
    return (min_date <= start or not require_start) and max_date >= end


def product_target_start(data_dir: Path, product: tuple[str | None, str | None, str], start: str) -> str:
    _, weights_name, _ = product
    if weights_name is None:
        return start
    bounds = parquet_date_bounds(data_dir / weights_name)
    if bounds is None:
        return start
    return max(start, bounds[0])


def expanded_cache_range(directory: Path, names: list[str], start: str, end: str) -> tuple[str, str]:
    bounds = cache_date_bounds(directory, names)
    if bounds is None:
        return start, end
    min_date, max_date = bounds
    return min(min_date, start), max(max_date, end)


def next_trading_day(data_dir: Path, date: str) -> str:
    calendar_path = data_dir / "trading_days.txt"
    if not calendar_path.exists():
        calendar_path = expand("~/data/trading_days.txt")
    if not calendar_path.exists():
        return date
    days = sorted(line.strip().replace("-", "") for line in calendar_path.read_text().splitlines() if line.strip())
    for day in days:
        if day > date:
            return day
    return date


def ensure_local_cache(args: argparse.Namespace, local_dir: Path, data_dir: Path, start: str, end: str) -> None:
    cache_end = next_trading_day(data_dir, end)
    if not args.refresh_cache and cache_covers(local_dir, LOCAL_CACHE_FILES, start, end):
        LOGGER.info("Using local cache %s covering %s-%s", local_dir, start, end)
        return
    build_start, build_end = (start, cache_end) if args.refresh_cache else expanded_cache_range(local_dir, LOCAL_CACHE_FILES, start, cache_end)
    LOGGER.info("Building local cache %s for %s-%s", local_dir, build_start, build_end)
    run_command([
        sys.executable,
        "prepare-data",
        "--start", build_start,
        "--end", build_end,
        "--output-dir", str(local_dir),
        "--wind-baseinfo", str(data_dir / "wind_baseinfo.parquet"),
        "--limit-label", str(data_dir / "FLAG_LIMIT_DAILY.parquet"),
        "--trading-days", str(data_dir / "trading_days.txt"),
        "--risk-dir", str(expand(args.risk_dir)),
    ])


def ensure_risk_cache(args: argparse.Namespace, local_dir: Path, risk_dir: Path, start: str, end: str) -> None:
    if args.skip_risk_model:
        return
    if not args.refresh_cache and cache_covers(risk_dir, RISK_CACHE_FILES, start, end, require_start=False):
        LOGGER.info("Using risk cache %s covering through %s", risk_dir, end)
        return
    local_bounds = cache_date_bounds(local_dir, LOCAL_CACHE_FILES)
    build_start, build_end = local_bounds if local_bounds is not None else (start, end)
    LOGGER.info("Building risk cache %s for %s-%s", risk_dir, build_start, build_end)
    run_command([
        sys.executable,
        "risk-model",
        "--exposures", str(local_dir / "style_exposures.parquet"),
        "--industry-exposures", str(local_dir / "industry_exposures.parquet"),
        "--returns", str(local_dir / "asset_forward_returns.parquet"),
        "--weights", str(local_dir / "security_master.parquet"),
        "--output-dir", str(risk_dir),
        "--start", build_start,
        "--end", build_end,
        "--halflife", str(args.risk_halflife),
        "--min-periods", str(args.risk_min_periods),
    ])


def target_cache_path(
    args: argparse.Namespace,
    cache_root: Path,
    product_name: str,
    factor_path: Path,
    max_daily_turnover: float | None,
) -> Path:
    stat = factor_path.stat() if factor_path.exists() else None
    payload = {
        "factor_path": str(factor_path),
        "factor_size": stat.st_size if stat else None,
        "factor_mtime_ns": stat.st_mtime_ns if stat else None,
        "factor_column": args.factor_column,
        "benchmark": product_name,
        "risk_model": not args.skip_risk_model,
        "style_exposure_penalty": args.style_exposure_penalty,
        "style_exposure_limit": args.style_exposure_limit,
        "alpha_transform": args.alpha_transform,
        "alpha_scale": args.alpha_scale,
        "alpha_clip": args.alpha_clip,
        "risk_aversion": args.risk_aversion,
        "turnover_penalty": args.turnover_penalty,
        "turnover_cost_bps": args.turnover_cost_bps,
        "turnover_cost_alpha_fraction": turnover_cost_alpha_fraction,
        "max_daily_turnover": max_daily_turnover,
        "max_weight": args.max_weight,
        "max_active_weight": args.max_active_weight,
        "tracking_error": args.tracking_error,
        "risk_halflife": args.risk_halflife,
        "risk_min_periods": args.risk_min_periods,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / "targets" / product_name / factor_path.stem / f"{digest}.parquet"


def export_factor_result_lenient(result, output_dir: Path, report_name: str) -> list[Path]:
    try:
        return result.export(output_dir, report_name=report_name)
    except RuntimeError as exc:
        if "xlsxwriter" not in str(exc):
            raise
        LOGGER.warning("xlsxwriter is missing; writing parquet/SVG outputs without Excel workbook")
        output_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for factor_name, frame in result.daily.items():
            path = output_dir / f"{factor_name}_base.parquet"
            frame.write_parquet(path)
            files.append(path)
        if result.daily:
            all_daily = pl.concat(
                [frame.with_columns(pl.lit(name).alias("factor")) for name, frame in result.daily.items()],
                how="diagonal_relaxed",
            )
            path = output_dir / "result_all.parquet"
            all_daily.write_parquet(path)
            files.append(path)
        svg_path = output_dir / f"{report_name}.svg"
        result._write_svg(svg_path)
        files.append(svg_path)
        return files



def select_products(selection: str) -> list[tuple[str | None, str | None, str]]:
    names = DEFAULT_PRODUCTS if selection.lower() == "all" else tuple(
        item.strip() for item in selection.split(",") if item.strip()
    )
    if not names:
        raise ValueError("--benchmark must name at least one product")
    products: list[tuple[str | None, str | None, str]] = []
    seen: set[str] = set()
    for name in names:
        if name not in BENCHMARK_FILES:
            allowed = ", ".join(["all", *sorted(BENCHMARK_FILES)])
            raise ValueError(f"Unknown benchmark '{name}'. Allowed: {allowed}")
        spec = BENCHMARK_FILES[name]
        if spec[2] not in seen:
            products.append(spec)
            seen.add(spec[2])
    return products


def fetchable_index_keys(products: list[tuple[str | None, str | None, str]]) -> list[str]:
    keys = []
    for returns_name, weights_name, product_name in products:
        if returns_name is None or weights_name is None:
            continue
        keys.append(product_name.replace("Zhishu", ""))
    return sorted(set(keys), key=lambda key: (len(key), key))


def zero_benchmark_returns(returns: pl.DataFrame) -> pl.DataFrame:
    return returns.select("date").unique().sort("date").with_columns(
        pl.lit(0.0).alias("benchmark_return")
    )


def zero_benchmark_weights(signals: pl.DataFrame) -> pl.DataFrame:
    return signals.select("date", "header").unique().sort(["date", "header"]).with_columns(
        pl.lit(0.0).alias("weight")
    )


def benchmark_inputs(
    data_dir: Path,
    returns: pl.DataFrame,
    signals: pl.DataFrame,
    product: tuple[str | None, str | None, str],
    start: str,
    end: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    returns_name, weights_name, _ = product
    if returns_name is None or weights_name is None:
        return zero_benchmark_returns(returns), zero_benchmark_weights(signals)
    return (
        filter_dates(load_frame(data_dir / returns_name), start, end),
        filter_dates(load_frame(data_dir / weights_name), start, end),
    )


def single_factor_product(products: list[tuple[str | None, str | None, str]]) -> tuple[str | None, str | None, str]:
    for product in products:
        if product[2] == "Zhishu500":
            return product
    for product in products:
        if product[2] != "ZeroIndex":
            return product
    return products[0]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end factor and portfolio backtests.")
    parser.add_argument(
        "--factor-values",
        required=True,
    )
    parser.add_argument("--factor-column", default="alpha")
    parser.add_argument(
        "--benchmark",
        default="all",
        help="Benchmark product, comma-separated products, or 'all' for ZeroIndex,300,500,1000,2000.",
    )
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--risk-dir", required=True, help="Directory of Wind/RQ risk exposure parquet files")
    parser.add_argument("--cache-root", default="~/data/factor_evaluator_cache")
    parser.add_argument("--output-root", default="~/data/factor_evaluator_reports")
    parser.add_argument("--run-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-benchmark-refresh", action="store_true")
    parser.add_argument("--skip-risk-model", action="store_true")
    parser.add_argument("--no-portfolio", action="store_true")
    parser.add_argument("--style-exposure-penalty", type=float, default=1.0)
    parser.add_argument("--style-exposure-limit", type=float)
    parser.add_argument("--alpha-transform", default="zscore", choices=("none", "zscore", "rank_zscore"))
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--alpha-clip", type=float, default=5.0)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--turnover-cost-bps", type=float, default=8.0)
    parser.add_argument("--turnover-cost-alpha-fraction", type=float, default=0.0, help="Cost/return constraint; use alpha-transform none with expected returns. Default disabled.")
    parser.add_argument("--max-daily-turnover", type=float, help="Hard cap for daily one-way turnover, e.g. 0.05 means 5%%.")
    parser.add_argument(
        "--annual-turnover-target",
        type=float,
        default=0.0,
        help="Optional hard annual one-way turnover target converted to a daily cap using 240 trading days; set <=0 to disable.",
    )
    parser.add_argument("--max-weight", type=float, default=0.03)
    parser.add_argument("--max-active-weight", type=float, default=0.01)
    parser.add_argument("--tracking-error", type=float, help="Annualized tracking error, using 252 daily covariance periods per year.")
    parser.add_argument("--risk-halflife", type=float, default=63.0)
    parser.add_argument("--risk-min-periods", type=int, default=60)
    parser.add_argument("--factor-buckets", type=int, default=30)
    parser.add_argument("--factor-topk", type=float, default=1.0)
    parser.add_argument("--factor-power", type=float, default=2.0)
    parser.add_argument(
        "--product-workers",
        type=int,
        default=1,
        help="Number of product target-generation subprocesses to run concurrently.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    start = args.start.replace("-", "")
    end = args.end.replace("-", "")
    products = select_products(args.benchmark)
    max_daily_turnover = args.max_daily_turnover
    if max_daily_turnover is None and args.annual_turnover_target is not None:
        if args.annual_turnover_target > 0:
            max_daily_turnover = args.annual_turnover_target / 240.0
    turnover_cost_alpha_fraction = args.turnover_cost_alpha_fraction
    if turnover_cost_alpha_fraction is not None and turnover_cost_alpha_fraction <= 0:
        turnover_cost_alpha_fraction = None

    data_dir = expand(args.data_dir)
    cache_root = expand(args.cache_root)
    local_cache_dir = cache_root / "local"
    risk_cache_dir = cache_root / f"risk_h{args.risk_halflife:g}_m{args.risk_min_periods}"
    factor_path = expand(args.factor_values)
    product_label = "ALL" if len(products) > 1 else products[0][2]
    output_dir = expand(args.output_root) / args.run_date / f"{start}_{end}_{product_label}"
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    risk_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_benchmark_refresh:
        missing_keys = []
        for returns_name, weights_name, product_name in products:
            if returns_name is None or weights_name is None:
                continue
            if args.refresh_cache or not (data_dir / returns_name).exists() or not (data_dir / weights_name).exists():
                missing_keys.append(product_name.replace("Zhishu", ""))
        if missing_keys:
            run_command([
                sys.executable,
                "fetch-wind",
                "--start", start,
                "--end", end,
                "--indexes", *sorted(set(missing_keys), key=lambda key: (len(key), key)),
                "--calendar", str(data_dir / "trading_days.txt"),
                "--output-dir", str(data_dir),
            ])

    ensure_local_cache(args, local_cache_dir, data_dir, start, end)
    ensure_risk_cache(args, local_cache_dir, risk_cache_dir, start, end)

    factor_frame = filter_dates(load_frame(factor_path), start, end).select("date", "header", args.factor_column)
    returns = filter_dates(load_frame(local_cache_dir / "asset_forward_returns.parquet"), start, end)
    universe = filter_dates(load_frame(local_cache_dir / "universe.parquet"), start, end)
    styles = filter_dates(load_frame(local_cache_dir / "style_exposures.parquet"), start, end)

    factor_benchmark_returns, _ = benchmark_inputs(
        data_dir,
        returns,
        factor_frame,
        single_factor_product(products),
        start,
        end,
    )
    LOGGER.info("Running single-factor backtest")
    factor_config = FactorBacktestConfig(
        start_date=start,
        end_date=end,
        buckets=args.factor_buckets,
        topk=args.factor_topk,
        power=args.factor_power,
    )
    factor_result = FactorBacktester(factor_config).run(
        FactorDataset(
            factors=factor_frame,
            returns=returns,
            benchmark=factor_benchmark_returns,
            universe=universe,
            styles=styles,
            limit_labels=load_frame(local_cache_dir / "tradability.parquet"),
        )
    )
    factor_files = export_factor_result_lenient(
        factor_result,
        output_dir / "single_factor",
        report_name=factor_path.stem,
    )
    LOGGER.info("Single-factor outputs: %s", output_dir / "single_factor")

    if args.no_portfolio:
        LOGGER.info("Skipping portfolio backtest")
        LOGGER.info("Local cache: %s", local_cache_dir)
        LOGGER.info("Generated %s factor files", len(factor_files))
        return

    industry_exposures = filter_dates(load_frame(local_cache_dir / "industry_exposures.parquet"), start, end)
    style_returns = load_frame(risk_cache_dir / "style_factor_returns.parquet") if (risk_cache_dir / "style_factor_returns.parquet").exists() else None
    industry_returns = load_frame(risk_cache_dir / "industry_factor_returns.parquet") if (risk_cache_dir / "industry_factor_returns.parquet").exists() else None
    portfolio_files: list[Path] = []
    target_jobs: list[tuple[tuple[str | None, str | None, str], Path, list[str], bool]] = []

    for product in products:
        _, _, product_name = product
        target_path = target_cache_path(args, cache_root, product_name, factor_path, max_daily_turnover)
        required_target_start = product_target_start(data_dir, product, start)
        target_start, target_end = (
            (required_target_start, end)
            if args.refresh_cache
            else expanded_cache_range(target_path.parent, [target_path.name], required_target_start, end)
        )
        target_cmd = [
            sys.executable,
            "build-targets",
            "--factor-values", str(factor_path),
            "--factor-column", args.factor_column,
            "--benchmark", product_name,
            "--data-dir", str(data_dir),
            "--start", target_start,
            "--end", target_end,
            "--universe", str(local_cache_dir / "universe.parquet"),
            "--tradability", str(local_cache_dir / "tradability.parquet"),
            "--asset-returns", str(local_cache_dir / "asset_forward_returns.parquet"),
            "--output", str(target_path),
            "--max-weight", str(args.max_weight),
            "--max-active-weight", str(args.max_active_weight),
            "--risk-aversion", str(args.risk_aversion),
            "--turnover-penalty", str(args.turnover_penalty),
            "--turnover-cost-bps", str(args.turnover_cost_bps),
            "--style-exposure-penalty", str(args.style_exposure_penalty),
            "--alpha-transform", args.alpha_transform,
            "--alpha-scale", str(args.alpha_scale),
            "--alpha-clip", str(args.alpha_clip),
            "--risk-exposures", str(local_cache_dir / "style_exposures.parquet"),
        ]
        if args.style_exposure_limit is not None:
            target_cmd.extend(["--style-exposure-limit", str(args.style_exposure_limit)])
        if product_name == "ZeroIndex":
            target_cmd.append("--allow-off-benchmark")
        if turnover_cost_alpha_fraction is not None:
            target_cmd.extend(["--turnover-cost-alpha-fraction", str(turnover_cost_alpha_fraction)])
        if max_daily_turnover is not None:
            target_cmd.extend(["--max-daily-turnover", str(max_daily_turnover)])
        if args.tracking_error is not None:
            target_cmd.extend(["--tracking-error", str(args.tracking_error)])
        if not args.skip_risk_model:
            target_cmd.extend([
                "--use-risk-model",
                "--industry-exposures", str(local_cache_dir / "industry_exposures.parquet"),
                "--factor-covariance", str(risk_cache_dir / "factor_covariance.parquet"),
                "--specific-risk", str(risk_cache_dir / "specific_risk.parquet"),
            ])
        needs_build = args.refresh_cache or not cache_covers(
            target_path.parent,
            [target_path.name],
            required_target_start,
            end,
        )
        target_jobs.append((product, target_path, target_cmd, needs_build))

    build_jobs = [job for job in target_jobs if job[3]]
    if build_jobs:
        workers = max(1, int(args.product_workers))
        LOGGER.info("Generating %s target files with %s worker(s)", len(build_jobs), workers)
        if workers == 1:
            for product, _, command, _ in build_jobs:
                LOGGER.info("Running portfolio target generation for %s", product[2])
                run_command(command)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_command, command): product[2] for product, _, command, _ in build_jobs}
                for future in as_completed(futures):
                    product_name = futures[future]
                    future.result()
                    LOGGER.info("Finished portfolio target generation for %s", product_name)
    for product, target_path, _, needs_build in target_jobs:
        if not needs_build:
            LOGGER.info("Using cached target weights: %s", target_path)
        _, _, product_name = product
        target_frame = filter_dates(load_frame(target_path), start, end)
        benchmark_returns, benchmark_weights = benchmark_inputs(data_dir, returns, factor_frame, product, start, end)
        if target_frame.is_empty() or benchmark_returns.is_empty():
            LOGGER.warning(
                "Skipping %s because target or benchmark data is empty in %s-%s",
                product_name,
                start,
                end,
            )
            continue
        LOGGER.info("Running portfolio backtest for %s", product_name)
        portfolio_result = PortfolioBacktester(PortfolioBacktestConfig()).run(
            target_frame,
            PortfolioDataset(
                returns=returns,
                benchmark=benchmark_returns,
                benchmark_weights=benchmark_weights,
                styles=styles,
                industry_exposures=industry_exposures,
                style_returns=style_returns,
                industry_returns=industry_returns,
            ),
        )
        portfolio_files.extend(portfolio_result.export(
            output_dir / "portfolio",
            product_name=product_name,
            test_name=factor_path.stem,
        ))
        LOGGER.info("%s target cache: %s", product_name, target_path)

    LOGGER.info("Portfolio outputs: %s", output_dir / "portfolio")
    LOGGER.info("Local cache: %s", local_cache_dir)
    LOGGER.info("Risk cache: %s", risk_cache_dir)
    LOGGER.info("Generated %s factor files and %s portfolio files", len(factor_files), len(portfolio_files))
