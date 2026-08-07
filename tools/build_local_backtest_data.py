#!/usr/bin/env python3
"""Build Factor Evaluator input tables from local Wind/RQ data.

This script produces framework-ready parquet files under ~/data by default:

    asset_forward_returns.parquet
    universe.parquet
    tradability.parquet
    execution_prices.parquet
    market_volume.parquet
    corporate_actions.parquet
    security_master.parquet
    trading_calendar.parquet
    style_exposures.parquet
    industry_exposures.parquet

Factor values remain an external input. Benchmark returns/weights are generated
by tools/fetch_wind_index_data.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import polars as pl


LOGGER = logging.getLogger("build_local_backtest_data")

DEFAULT_WIND_BASEINFO = "~/data/wind_baseinfo.parquet"
DEFAULT_LIMIT_LABEL = "~/data/FLAG_LIMIT_DAILY.parquet"
DEFAULT_TRADING_DAYS = "~/data/trading_days.txt"
DEFAULT_RISK_DIR = "/mnt/hdd/researchshare/RQRiskData/BarraStyleFactor"
DEFAULT_OUTPUT = "~/data"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local backtest input parquet files.")
    parser.add_argument("--wind-baseinfo", default=DEFAULT_WIND_BASEINFO)
    parser.add_argument("--limit-label", default=DEFAULT_LIMIT_LABEL)
    parser.add_argument("--trading-days", default=DEFAULT_TRADING_DAYS)
    parser.add_argument("--risk-dir", default=DEFAULT_RISK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--start", help="Start date YYYYMMDD")
    parser.add_argument("--end", help="End date YYYYMMDD")
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="Skip style_exposures and industry_exposures generation.",
    )
    return parser.parse_args()


def normalize_date_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.replace_all("-", "").str.slice(0, 8)


def date_filter(frame: pl.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    if start:
        frame = frame.filter(pl.col("date") >= start.replace("-", ""))
    if end:
        frame = frame.filter(pl.col("date") <= end.replace("-", ""))
    return frame


def read_trading_days(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    days = [line.strip().replace("-", "") for line in path.read_text().splitlines() if line.strip()]
    bad = [day for day in days if not re.fullmatch(r"\d{8}", day)]
    if bad:
        raise ValueError(f"Invalid trading day values: {bad[:5]}")
    return pl.DataFrame({"date": sorted(set(days))})


def read_wind_baseinfo(path: Path, start: str | None, end: str | None) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).with_columns(
        normalize_date_expr("TRADE_DT").alias("date"),
        pl.col("S_INFO_WINDCODE").cast(pl.String).alias("code"),
    )
    return date_filter(frame, start, end).sort(["code", "date"])


def write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    LOGGER.info("Wrote %s (%s rows, %s cols)", path, frame.height, len(frame.columns))


def build_asset_forward_returns(wind: pl.DataFrame) -> pl.DataFrame:
    return (
        wind.select(["date", "code", "S_DQ_ADJCLOSE"])
        .sort(["code", "date"])
        .with_columns(
            (pl.col("S_DQ_ADJCLOSE").shift(-1).over("code") / pl.col("S_DQ_ADJCLOSE") - 1.0)
            .alias("return")
        )
        .select("date", "code", pl.col("return").cast(pl.Float64))
        .drop_nulls("return")
        .sort(["date", "code"])
    )


def build_universe(wind: pl.DataFrame) -> pl.DataFrame:
    eligible = (
        pl.col("validflag").fill_null(False)
        & pl.col("MASK_STOCKPOOL").fill_null(False)
        & ~pl.col("MASK_ST").fill_null(False)
        & ~pl.col("MASK_MKTCAPISNA").fill_null(True)
        & ~pl.col("delistflag").fill_null(False)
    )
    if "MASK_LIST30D" in wind.columns:
        eligible = eligible & (pl.col("MASK_LIST30D").is_null() | pl.col("MASK_LIST30D"))
    return wind.select("date", "code", eligible.alias("eligible")).sort(["date", "code"])


def build_tradability(wind: pl.DataFrame, limit_path: Path | None) -> pl.DataFrame:
    tradable = wind.select(
        "date",
        "code",
        (
            pl.col("validflag").fill_null(False)
            & pl.col("mask_tradable").fill_null(False)
            & ~pl.col("delistflag").fill_null(False)
        ).alias("tradable"),
        pl.col("S_DQ_TRADESTATUSCODE").alias("trade_status"),
        pl.col("S_DQ_LIMIT").alias("limit_up_price"),
        pl.col("S_DQ_STOPPING").alias("limit_down_price"),
    )
    if limit_path is not None and limit_path.exists():
        labels = pl.read_parquet(limit_path).select(
            normalize_date_expr("trade_date").alias("date"),
            pl.col("st_code").cast(pl.String).alias("code"),
            pl.col("limit_label").cast(pl.Int8),
        )
        tradable = tradable.join(labels, on=["date", "code"], how="left")
    else:
        tradable = tradable.with_columns(pl.lit(None, dtype=pl.Int8).alias("limit_label"))
    return tradable.sort(["date", "code"])


def build_execution_prices(wind: pl.DataFrame) -> pl.DataFrame:
    cols = {
        "S_DQ_ADJOPEN": "adj_open",
        "S_DQ_ADJCLOSE": "adj_close",
        "S_DQ_AVGPRICE": "avg_price",
        "S_DQ_OPEN": "open",
        "S_DQ_CLOSE": "close",
    }
    return wind.select(
        "date",
        "code",
        *[pl.col(src).alias(dst) for src, dst in cols.items() if src in wind.columns],
    ).sort(["date", "code"])


def build_market_volume(wind: pl.DataFrame) -> pl.DataFrame:
    return wind.select(
        "date",
        "code",
        pl.col("S_DQ_VOLUME").alias("volume"),
        pl.col("S_DQ_AMOUNT").alias("amount"),
    ).sort(["date", "code"])


def build_corporate_actions(wind: pl.DataFrame) -> pl.DataFrame:
    return wind.select(
        "date",
        "code",
        pl.col("S_DQ_ADJFACTOR").alias("adj_factor"),
    ).sort(["date", "code"])


def build_security_master(wind: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "date",
        "code",
        "validflag",
        "delistflag",
        "MASK_ST",
        "MASK_LIST30D",
        "MASK_LIST90D",
        "MASK_STOCKPOOL",
        "FREE_SHARES_TODAY",
        "S_VAL_MV",
        "S_DQ_MV",
    ]
    return wind.select([c for c in cols if c in wind.columns]).sort(["date", "code"])


def risk_files(risk_dir: Path) -> list[Path]:
    if not risk_dir.exists():
        raise FileNotFoundError(risk_dir)
    return sorted(risk_dir.glob("*.parquet"))


def build_risk_exposures(risk_dir: Path, start: str | None, end: str | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    files = risk_files(risk_dir)
    if start:
        files = [p for p in files if p.stem.replace("-", "") >= start.replace("-", "")]
    if end:
        files = [p for p in files if p.stem.replace("-", "") <= end.replace("-", "")]
    if not files:
        raise RuntimeError("No risk exposure parquet files in requested range")
    frame = pl.concat(
        [pl.read_parquet(p) for p in files],
        how="diagonal_relaxed",
    ).with_columns(
        normalize_date_expr("TRADE_DT").alias("date"),
        pl.col("S_INFO_WINDCODE").cast(pl.String).alias("code"),
    )
    styles = [c for c in STYLE_COLUMNS if c in frame.columns]
    industry_cols = [
        c for c in frame.columns
        if c not in {"TRADE_DT", "S_INFO_WINDCODE", "date", "code", *STYLE_COLUMNS}
    ]
    style_exposures = frame.select("date", "code", *styles).sort(["date", "code"])
    industry_exposures = frame.select("date", "code", *industry_cols).sort(["date", "code"])
    return style_exposures, industry_exposures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = expand(args.output_dir)

    wind = read_wind_baseinfo(expand(args.wind_baseinfo), args.start, args.end)
    trading_calendar = date_filter(read_trading_days(expand(args.trading_days)), args.start, args.end)

    write_frame(build_asset_forward_returns(wind), output_dir / "asset_forward_returns.parquet")
    write_frame(build_universe(wind), output_dir / "universe.parquet")
    write_frame(build_tradability(wind, expand(args.limit_label)), output_dir / "tradability.parquet")
    write_frame(build_execution_prices(wind), output_dir / "execution_prices.parquet")
    write_frame(build_market_volume(wind), output_dir / "market_volume.parquet")
    write_frame(build_corporate_actions(wind), output_dir / "corporate_actions.parquet")
    write_frame(build_security_master(wind), output_dir / "security_master.parquet")
    write_frame(trading_calendar, output_dir / "trading_calendar.parquet")

    if not args.no_risk:
        style_exposures, industry_exposures = build_risk_exposures(expand(args.risk_dir), args.start, args.end)
        write_frame(style_exposures, output_dir / "style_exposures.parquet")
        write_frame(industry_exposures, output_dir / "industry_exposures.parquet")


if __name__ == "__main__":
    main()
