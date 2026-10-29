"""Build EWMA risk-model data from local Factor Evaluator input parquet files."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import polars as pl


from factor_evaluator.risk import (
    estimate_factor_returns,
    ewma_factor_covariance,
    ewma_specific_risk,
)


LOGGER = logging.getLogger("build_risk_model_data")

STYLE_COLUMNS = [
    "liquidity",
    "leverage",
    "earnings_variability",
    "earnings_quality",
    "profitability",
    "investment_quality",
    "book_to_price",
    "earnings_yield",
    "longterm_reversal",
    "growth",
    "momentum",
    "mid_cap",
    "size",
    "beta",
    "residual_volatility",
    "dividend_yield",
    "comovement",
]


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


def date_filter(frame: pl.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    if start:
        frame = frame.filter(pl.col("date").cast(pl.String) >= start.replace("-", ""))
    if end:
        frame = frame.filter(pl.col("date").cast(pl.String) <= end.replace("-", ""))
    return frame


def write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    LOGGER.info("Wrote %s (%s rows, %s cols)", path, frame.height, len(frame.columns))


def split_factor_returns(factor_returns: pl.DataFrame, style_columns: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    styles = [column for column in style_columns if column in factor_returns.columns]
    industries = [column for column in factor_returns.columns if column not in {"date", *styles}]
    return factor_returns.select("date", *styles), factor_returns.select("date", *industries)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factor returns and EWMA risk estimates.")
    parser.add_argument("--exposures", default="~/data/style_exposures.parquet")
    parser.add_argument("--industry-exposures", default="~/data/industry_exposures.parquet")
    parser.add_argument("--returns", default="~/data/asset_forward_returns.parquet")
    parser.add_argument("--weights", default="~/data/security_master.parquet")
    parser.add_argument("--weight-column", default="S_DQ_MV")
    parser.add_argument("--output-dir", default="~/data")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--halflife", type=float, default=63.0)
    parser.add_argument("--min-periods", type=int, default=60)
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Use current day factor/specific returns in same-date EWMA estimates.",
    )
    parser.add_argument(
        "--styles-only",
        action="store_true",
        help="Do not append industry exposure columns to the regression design.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    output_dir = expand(args.output_dir)
    exposures = date_filter(load_frame(expand(args.exposures)), args.start, args.end)
    if not args.styles_only:
        industry = date_filter(load_frame(expand(args.industry_exposures)), args.start, args.end)
        exposures = exposures.join(industry, on=["date", "code"], how="inner")
    returns = date_filter(load_frame(expand(args.returns)), args.start, args.end)

    weights_path = expand(args.weights)
    weights = date_filter(load_frame(weights_path), args.start, args.end) if weights_path.exists() else None
    if weights is not None and args.weight_column not in weights.columns:
        LOGGER.warning("Weight column %s not found in %s; using equal weights", args.weight_column, weights_path)
        weights = None

    factor_returns, specific_returns = estimate_factor_returns(
        exposures,
        returns,
        weights=weights,
        weight_column=args.weight_column,
    )
    factor_covariance = ewma_factor_covariance(
        factor_returns,
        halflife=args.halflife,
        min_periods=args.min_periods,
        include_current=args.include_current,
    )
    specific_risk = ewma_specific_risk(
        specific_returns,
        halflife=args.halflife,
        min_periods=args.min_periods,
        include_current=args.include_current,
    )
    style_returns, industry_returns = split_factor_returns(factor_returns, STYLE_COLUMNS)

    write_frame(factor_returns, output_dir / "risk_factor_returns.parquet")
    write_frame(style_returns, output_dir / "style_factor_returns.parquet")
    write_frame(industry_returns, output_dir / "industry_factor_returns.parquet")
    write_frame(specific_returns, output_dir / "specific_return_daily.parquet")
    write_frame(factor_covariance, output_dir / "factor_covariance.parquet")
    write_frame(specific_risk, output_dir / "specific_risk.parquet")
