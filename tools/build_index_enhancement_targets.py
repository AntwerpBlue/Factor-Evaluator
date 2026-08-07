#!/usr/bin/env python3
"""Build index-enhancement target weights from factor signals.

The optimizer is factor_evaluator.portfolio.IndexEnhancementStrategy. Inputs are
long panel parquet/csv files using date + code/header plus one value column.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_evaluator import ExposureMatrixProvider, IndexEnhancementConfig, IndexEnhancementStrategy, RiskMatrixProvider
from factor_evaluator.data import normalize_panel


LOGGER = logging.getLogger("build_index_enhancement_targets")

BENCHMARK_WEIGHT_FILES = {
    "300": "benchmark_300_weights.parquet",
    "500": "benchmark_500_weights.parquet",
    "1000": "benchmark_1000_weights.parquet",
    "2000": "benchmark_2000_weights.parquet",
    "Zhishu300": "benchmark_300_weights.parquet",
    "Zhishu500": "benchmark_500_weights.parquet",
    "Zhishu1000": "benchmark_1000_weights.parquet",
    "Zhishu2000": "benchmark_2000_weights.parquet",
    "ZeroIndex": None,
    "zeroindex": None,
}


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path)))


def load_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def value_column(frame: pl.DataFrame, requested: str | None, label: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"{label} column not found: {requested}")
        return requested
    values = [c for c in frame.columns if c not in {"date", "asset"}]
    if len(values) != 1:
        raise ValueError(f"{label} must have exactly one value column or pass an explicit option")
    return values[0]


def date_filter(frame: pl.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    if start:
        frame = frame.filter(pl.col("date").cast(pl.String) >= start.replace("-", ""))
    if end:
        frame = frame.filter(pl.col("date").cast(pl.String) <= end.replace("-", ""))
    return frame


def filter_by_panel_flag(frame: pl.DataFrame, flag: pl.DataFrame, column: str) -> pl.DataFrame:
    flag_panel = normalize_panel(flag)
    if column not in flag_panel.columns:
        raise ValueError(f"Flag column not found: {column}")
    keep = flag_panel.select("date", "asset", pl.col(column).cast(pl.Boolean).alias("_keep"))
    return frame.join(keep, on=["date", "asset"], how="inner").filter(pl.col("_keep")).drop("_keep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized index-enhancement target weights.")
    parser.add_argument("--factor-values", required=True, help="External factor parquet/csv")
    parser.add_argument("--factor-column", help="Factor column to optimize on")
    parser.add_argument("--benchmark", default="500", choices=sorted(BENCHMARK_WEIGHT_FILES))
    parser.add_argument("--data-dir", default="~/data")
    parser.add_argument("--benchmark-weights", help="Override benchmark weights file")
    parser.add_argument("--universe", default="~/data/universe.parquet")
    parser.add_argument("--tradability", default="~/data/tradability.parquet")
    parser.add_argument("--asset-returns", help="Asset return panel used to drift previous target weights before optimization.")
    parser.add_argument("--output", default="~/data/target_weights.parquet")
    parser.add_argument("--use-risk-model", action="store_true", help="Use X F X' + D risk matrices from EWMA risk files.")
    parser.add_argument("--risk-exposures", default="~/data/style_exposures.parquet")
    parser.add_argument("--industry-exposures", default="~/data/industry_exposures.parquet")
    parser.add_argument("--factor-covariance", default="~/data/factor_covariance.parquet")
    parser.add_argument("--specific-risk", default="~/data/specific_risk.parquet")
    parser.add_argument("--specific-var-floor", type=float, default=1e-8)
    parser.add_argument("--style-exposure-penalty", type=float, default=0.0, help="Soft penalty for active style exposure squared norm.")
    parser.add_argument("--style-exposure-limit", type=float, help="Hard active style exposure band, matching style_bands semantics.")
    parser.add_argument("--alpha-transform", default="zscore", choices=("none", "zscore", "rank_zscore"))
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--alpha-clip", type=float, default=5.0)
    parser.add_argument("--allow-off-benchmark", action="store_true", help="Allow assets outside the benchmark constituent file.")
    parser.add_argument("--max-weight", type=float, default=0.03)
    parser.add_argument("--max-active-weight", type=float, default=0.01)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
    parser.add_argument("--turnover-cost-bps", type=float, default=8.0)
    parser.add_argument("--turnover-cost-alpha-fraction", type=float, default=0.9, help="PF soft turnover gate: require turnover cost <= fraction * expected alpha improvement; set <=0 to disable.")
    parser.add_argument("--max-daily-turnover", type=float, help="Hard cap for daily one-way turnover, e.g. 0.05 means 5%%.")
    parser.add_argument("--tracking-error", type=float)
    parser.add_argument("--long-short", action="store_true", help="Allow short positions")
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    data_dir = expand(args.data_dir)
    zero_benchmark = BENCHMARK_WEIGHT_FILES[args.benchmark] is None
    benchmark_path = (
        expand(args.benchmark_weights)
        if args.benchmark_weights
        else None if zero_benchmark else data_dir / BENCHMARK_WEIGHT_FILES[args.benchmark]
    )

    signals = normalize_panel(load_frame(expand(args.factor_values)))
    factor_col = value_column(signals, args.factor_column, "factor_values")
    signals = signals.select("date", "asset", pl.col(factor_col).cast(pl.Float64).alias("alpha"))
    if args.start:
        signals = signals.filter(pl.col("date") >= args.start.replace("-", ""))
    if args.end:
        signals = signals.filter(pl.col("date") <= args.end.replace("-", ""))

    universe_path = expand(args.universe)
    if universe_path.exists():
        signals = filter_by_panel_flag(signals, load_frame(universe_path), "eligible")
    tradability_path = expand(args.tradability)
    if tradability_path.exists():
        signals = filter_by_panel_flag(signals, load_frame(tradability_path), "tradable")

    asset_returns = date_filter(normalize_panel(load_frame(expand(args.asset_returns))), args.start, args.end) if args.asset_returns else None

    if zero_benchmark:
        benchmark_weights = signals.select("date", "asset").unique().with_columns(pl.lit(0.0).alias("weight"))
    else:
        assert benchmark_path is not None
        benchmark_weights = normalize_panel(load_frame(benchmark_path))
        weight_col = value_column(benchmark_weights, "weight" if "weight" in benchmark_weights.columns else None, "benchmark_weights")
        benchmark_weights = benchmark_weights.select("date", "asset", pl.col(weight_col).cast(pl.Float64).alias("weight"))
        if not args.allow_off_benchmark:
            benchmark_assets = benchmark_weights.select("date", "asset").unique()
            signals = signals.join(benchmark_assets, on=["date", "asset"], how="inner")

    exposure_provider = None
    if args.style_exposure_penalty > 0 or args.style_exposure_limit is not None:
        style_exposures = date_filter(normalize_panel(load_frame(expand(args.risk_exposures))), args.start, args.end)
        exposure_provider = ExposureMatrixProvider(style_exposures)
        LOGGER.info(
            "Using style exposure controls: penalty=%s limit=%s",
            args.style_exposure_penalty,
            args.style_exposure_limit,
        )

    covariance_provider = None
    if args.use_risk_model:
        risk_exposures = date_filter(normalize_panel(load_frame(expand(args.risk_exposures))), args.start, args.end)
        industry_path = expand(args.industry_exposures)
        if industry_path.exists():
            industry_exposures = date_filter(normalize_panel(load_frame(industry_path)), args.start, args.end)
            risk_exposures = risk_exposures.join(industry_exposures, on=["date", "asset"], how="inner")
        factor_covariance = date_filter(load_frame(expand(args.factor_covariance)), args.start, args.end)
        specific_risk = date_filter(load_frame(expand(args.specific_risk)), args.start, args.end)
        covariance_provider = RiskMatrixProvider(
            risk_exposures,
            factor_covariance,
            specific_risk,
            specific_var_floor=args.specific_var_floor,
        )
        LOGGER.info("Using EWMA risk model for optimization")

    config = IndexEnhancementConfig(
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        max_weight=args.max_weight,
        max_active_weight=args.max_active_weight,
        tracking_error=args.tracking_error,
        long_only=not args.long_short,
        active_exposure_penalty=args.style_exposure_penalty,
        active_exposure_limit=args.style_exposure_limit,
        turnover_cost_bps=args.turnover_cost_bps,
        turnover_cost_alpha_fraction=(
            args.turnover_cost_alpha_fraction
            if args.turnover_cost_alpha_fraction is None or args.turnover_cost_alpha_fraction > 0
            else None
        ),
        max_turnover=args.max_daily_turnover,
        zero_benchmark=zero_benchmark,
        alpha_transform=args.alpha_transform,
        alpha_scale=args.alpha_scale,
        alpha_clip=args.alpha_clip,
    )
    targets = IndexEnhancementStrategy(config).build_targets(
        signals,
        benchmark_weights,
        covariance_by_date=covariance_provider,
        exposure_by_date=exposure_provider,
        asset_returns=asset_returns,
    )
    output = expand(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    targets.rename({"asset": "code"}).write_parquet(output)
    LOGGER.info("Wrote %s (%s rows)", output, targets.height)


if __name__ == "__main__":
    main()
