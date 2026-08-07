#!/usr/bin/env python3
"""
Fetch benchmark index returns and constituent weights from the local Wind DB.

Default outputs are written to ~/data and follow config/backtest.example.toml:

    benchmark_300_returns.parquet
    benchmark_300_weights.parquet
    benchmark_500_returns.parquet
    benchmark_500_weights.parquet
    benchmark_1000_returns.parquet
    benchmark_1000_weights.parquet
    benchmark_2000_returns.parquet
    benchmark_2000_weights.parquet

By default the script is incremental: if output parquet files already exist, it
only fetches missing trading days and merges them into the existing files. Use
--full-refresh to rebuild the selected outputs from scratch.

Returns are close-to-next-close by default: the row keyed by date T contains
the return earned after forming a position on T.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml
from sqlalchemy import create_engine, text


LOGGER = logging.getLogger("fetch_wind_index_data")

DEFAULT_CONFIG = "config/wind_info.yaml"
DEFAULT_CALENDAR = "~/data/trading_days.txt"
DEFAULT_OUTPUT = "~/data"

INDEXES = {
    "300": "000300.SH",
    "500": "000905.SH",
    "1000": "000852.SH",
    "2000": "932000.CSI",
}

INDEX_NAME_ALIASES = {
    "Zhishu300": "300",
    "Zhishu500": "500",
    "Zhishu1000": "1000",
    "Zhishu2000": "2000",
}

INDEX_EOD_TABLE = "AINDEXEODPRICES"
INDEX_WEIGHT_TABLE = "AINDEXHS300FREEWEIGHT"
CODE_RE = re.compile(r"^[A-Za-z0-9.]+$")


@dataclass(frozen=True)
class DateRange:
    start: str
    end: str


@dataclass(frozen=True)
class OutputPaths:
    returns: Path
    weights: Path


@dataclass(frozen=True)
class FetchPlan:
    key: str
    code: str
    dates: DateRange
    existing_returns: pl.DataFrame | None
    existing_weights: pl.DataFrame | None


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sql_engine(db_cfg: dict):
    driver = "ODBC Driver 18 for SQL Server"
    server = db_cfg["server"]
    database = db_cfg["database"]
    uid = db_cfg.get("uid", "")
    pwd = db_cfg.get("pwd", "")
    if uid:
        conn_str = (
            f"mssql+pyodbc://{uid}:{pwd}@{server}/{database}"
            f"?driver={driver}&TrustServerCertificate=yes"
        )
    else:
        conn_str = f"mssql+pyodbc://{server}/{database}?driver={driver}"
    return create_engine(conn_str, fast_executemany=True, future=True)


def read_calendar(path: str | Path) -> list[str]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    days = [
        line.strip().replace("-", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad = [day for day in days if not re.fullmatch(r"\d{8}", day)]
    if bad:
        raise ValueError(f"Invalid trading-day values in {path}: {bad[:5]}")
    return sorted(set(days))


def resolve_date_range(
    trading_days: list[str],
    start: str | None,
    end: str | None,
) -> DateRange:
    if not trading_days and (not start or not end):
        raise ValueError("start/end are required when no trading calendar is supplied")
    actual_start = (start or trading_days[0]).replace("-", "")
    actual_end = (end or trading_days[-1]).replace("-", "")
    if not re.fullmatch(r"\d{8}", actual_start) or not re.fullmatch(r"\d{8}", actual_end):
        raise ValueError("start and end must use YYYYMMDD")
    if actual_start > actual_end:
        raise ValueError(f"start is after end: {actual_start} > {actual_end}")
    return DateRange(actual_start, actual_end)


def next_trading_day(trading_days: list[str], date: str) -> str | None:
    for day in trading_days:
        if day > date:
            return day
    return None


def previous_trading_day(trading_days: list[str], date: str) -> str | None:
    previous = None
    for day in trading_days:
        if day >= date:
            return previous
        previous = day
    return previous


def normalize_index_selection(names: Iterable[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in names:
        key = INDEX_NAME_ALIASES.get(name, name)
        if key not in INDEXES:
            allowed = ", ".join([*INDEXES, *INDEX_NAME_ALIASES])
            raise ValueError(f"Unknown index '{name}'. Allowed: {allowed}")
        selected[key] = INDEXES[key]
    return selected


def quoted_codes(codes: Iterable[str]) -> str:
    values = []
    for code in codes:
        if not CODE_RE.fullmatch(code):
            raise ValueError(f"Unsafe Wind code: {code}")
        values.append(f"'{code}'")
    return ", ".join(values)


def output_paths(output_dir: Path, key: str) -> OutputPaths:
    return OutputPaths(
        returns=output_dir / f"benchmark_{key}_returns.parquet",
        weights=output_dir / f"benchmark_{key}_weights.parquet",
    )


def read_existing(path: Path, required_columns: set[str]) -> pl.DataFrame | None:
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.with_columns(pl.col("date").cast(pl.String))


def max_date(frame: pl.DataFrame | None) -> str | None:
    if frame is None or frame.is_empty():
        return None
    return frame.select(pl.max("date")).item()


def min_present_date(frame: pl.DataFrame | None) -> str | None:
    if frame is None or frame.is_empty():
        return None
    return frame.select(pl.min("date")).item()


def desired_return_end(trading_days: list[str], dates: DateRange, *, forward: bool) -> str:
    if not forward:
        return dates.end
    previous = previous_trading_day(trading_days, dates.end)
    if previous is None:
        raise ValueError(f"No prior trading day before {dates.end}; cannot build forward return")
    return previous


def missing_start(
    existing: pl.DataFrame | None,
    dates: DateRange,
    trading_days: list[str],
    desired_end: str,
) -> str | None:
    last = max_date(existing)
    if last is None:
        return dates.start
    if last >= desired_end:
        return None
    return next_trading_day(trading_days, last) or dates.start


def build_fetch_plans(
    indexes: dict[str, str],
    output_dir: Path,
    dates: DateRange,
    trading_days: list[str],
    *,
    forward_returns: bool,
    full_refresh: bool,
) -> list[FetchPlan]:
    plans: list[FetchPlan] = []
    ret_end = desired_return_end(trading_days, dates, forward=forward_returns)
    for key, code in indexes.items():
        paths = output_paths(output_dir, key)
        existing_returns = None if full_refresh else read_existing(
            paths.returns, {"date", "benchmark_return"}
        )
        existing_weights = None if full_refresh else read_existing(
            paths.weights, {"date", "code", "weight"}
        )

        ret_start = dates.start if full_refresh else missing_start(
            existing_returns, dates, trading_days, ret_end
        )
        weight_start = dates.start if full_refresh else missing_start(
            existing_weights, dates, trading_days, dates.end
        )
        starts = [day for day in (ret_start, weight_start) if day is not None]
        if not starts:
            LOGGER.info("%s is already up to date through %s", key, dates.end)
            continue

        fetch_start = min(starts)
        if fetch_start > dates.end:
            LOGGER.info("%s has no fetchable dates in %s -> %s", key, dates.start, dates.end)
            continue
        LOGGER.info("%s incremental fetch start: %s", key, fetch_start)
        plans.append(
            FetchPlan(
                key=key,
                code=code,
                dates=DateRange(fetch_start, dates.end),
                existing_returns=existing_returns,
                existing_weights=existing_weights,
            )
        )
    return plans


def fetch_index_prices(engine, codes: dict[str, str], dates: DateRange) -> pl.DataFrame:
    query = f"""
        SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_CLOSE, S_DQ_PCTCHANGE
        FROM WINDFILESYNC.dbo.{INDEX_EOD_TABLE}
        WHERE S_INFO_WINDCODE IN ({quoted_codes(codes.values())})
          AND TRADE_DT >= :start_date
          AND TRADE_DT <= :end_date
    """
    with engine.connect() as conn:
        frame = pl.read_database(
            text(query),
            conn,
            execute_options={"parameters": {"start_date": dates.start, "end_date": dates.end}},
        )
    if frame.is_empty():
        raise RuntimeError("Wind index price query returned no rows")
    return frame.with_columns(
        pl.col("TRADE_DT").cast(pl.String),
        pl.col("S_INFO_WINDCODE").cast(pl.String),
        pl.col("S_DQ_CLOSE").cast(pl.Float64),
        pl.col("S_DQ_PCTCHANGE").cast(pl.Float64),
    ).sort(["S_INFO_WINDCODE", "TRADE_DT"])


def build_benchmark_returns(
    prices: pl.DataFrame,
    index_code: str,
    dates: DateRange,
    *,
    forward: bool,
    trading_days: list[str],
) -> pl.DataFrame:
    frame = prices.filter(pl.col("S_INFO_WINDCODE") == index_code).sort("TRADE_DT")
    if frame.is_empty():
        raise RuntimeError(f"No index prices for {index_code}")
    if forward:
        returns = frame.with_columns(
            (
                pl.col("S_DQ_CLOSE").shift(-1) / pl.col("S_DQ_CLOSE") - 1.0
            ).alias("benchmark_return")
        )
        end = desired_return_end(trading_days, dates, forward=True)
    else:
        returns = frame.with_columns(
            (pl.col("S_DQ_PCTCHANGE") / 100.0).alias("benchmark_return")
        )
        end = dates.end
    return (
        returns.select(
            pl.col("TRADE_DT").alias("date"),
            pl.col("benchmark_return").cast(pl.Float64),
        )
        .filter(pl.col("date").is_between(pl.lit(dates.start), pl.lit(end)))
        .drop_nulls("benchmark_return")
        .sort("date")
    )


def fetch_raw_weights(engine, codes: dict[str, str], end_date: str) -> pl.DataFrame:
    # Pull all available snapshots up to end_date so the first requested day can
    # be filled from the latest earlier Wind weight snapshot.
    query = f"""
        SELECT S_INFO_WINDCODE, S_CON_WINDCODE, TRADE_DT, I_WEIGHT
        FROM WINDFILESYNC.dbo.{INDEX_WEIGHT_TABLE}
        WHERE S_INFO_WINDCODE IN ({quoted_codes(codes.values())})
          AND TRADE_DT <= :end_date
    """
    with engine.connect() as conn:
        frame = pl.read_database(
            text(query),
            conn,
            execute_options={"parameters": {"end_date": end_date}},
        )
    if frame.is_empty():
        raise RuntimeError("Wind index weight query returned no rows")
    return frame.with_columns(
        pl.col("S_INFO_WINDCODE").cast(pl.String),
        pl.col("S_CON_WINDCODE").cast(pl.String),
        pl.col("TRADE_DT").cast(pl.String),
        pl.col("I_WEIGHT").cast(pl.Float64),
    ).sort(["S_INFO_WINDCODE", "TRADE_DT", "S_CON_WINDCODE"])


def normalize_weight_scale(frame: pl.DataFrame) -> pl.DataFrame:
    sums = frame.group_by(["S_INFO_WINDCODE", "TRADE_DT"]).agg(
        pl.col("I_WEIGHT").sum().alias("_sum_weight")
    )
    scaled = frame.join(sums, on=["S_INFO_WINDCODE", "TRADE_DT"], how="left")
    return scaled.with_columns(
        pl.when(pl.col("_sum_weight") > 2.0)
        .then(pl.col("I_WEIGHT") / 100.0)
        .otherwise(pl.col("I_WEIGHT"))
        .alias("weight")
    ).drop("_sum_weight")


def build_daily_weights(
    raw_weights: pl.DataFrame,
    index_code: str,
    trading_days: list[str],
    dates: DateRange,
    *,
    daily_fill: bool,
) -> pl.DataFrame:
    frame = normalize_weight_scale(
        raw_weights.filter(pl.col("S_INFO_WINDCODE") == index_code)
    )
    if frame.is_empty():
        raise RuntimeError(f"No index weights for {index_code}")

    if not daily_fill:
        return (
            frame.filter(pl.col("TRADE_DT").is_between(pl.lit(dates.start), pl.lit(dates.end)))
            .select(
                pl.col("TRADE_DT").alias("date"),
                pl.col("S_CON_WINDCODE").alias("code"),
                pl.col("weight"),
            )
            .sort(["date", "code"])
        )

    calendar = pl.DataFrame(
        {"date": [day for day in trading_days if dates.start <= day <= dates.end]}
    )
    if calendar.is_empty():
        raise RuntimeError("No trading days in requested range")

    snapshots = frame.select("TRADE_DT").unique().sort("TRADE_DT").rename(
        {"TRADE_DT": "source_date"}
    )
    date_map = calendar.join_asof(
        snapshots,
        left_on="date",
        right_on="source_date",
        strategy="backward",
    ).drop_nulls("source_date")
    if date_map.is_empty():
        raise RuntimeError(f"No weight snapshot on or before requested range for {index_code}")

    return (
        date_map.join(frame, left_on="source_date", right_on="TRADE_DT", how="inner")
        .select(
            "date",
            pl.col("S_CON_WINDCODE").alias("code"),
            pl.col("weight").cast(pl.Float64),
        )
        .sort(["date", "code"])
    )


def merge_returns(existing: pl.DataFrame | None, new: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in (existing, new) if frame is not None and not frame.is_empty()]
    if not frames:
        return pl.DataFrame(schema={"date": pl.String, "benchmark_return": pl.Float64})
    return pl.concat(frames, how="vertical").unique(subset=["date"], keep="last").sort("date")


def merge_weights(existing: pl.DataFrame | None, new: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in (existing, new) if frame is not None and not frame.is_empty()]
    if not frames:
        return pl.DataFrame(schema={"date": pl.String, "code": pl.String, "weight": pl.Float64})
    return (
        pl.concat(frames, how="vertical")
        .unique(subset=["date", "code"], keep="last")
        .sort(["date", "code"])
    )


def write_outputs(
    output_dir: Path,
    key: str,
    returns: pl.DataFrame,
    weights: pl.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, key)
    returns.write_parquet(paths.returns)
    weights.write_parquet(paths.weights)
    LOGGER.info("Wrote %s (%s rows)", paths.returns, returns.height)
    LOGGER.info("Wrote %s (%s rows)", paths.weights, weights.height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wind benchmark returns and index constituent weights."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Wind YAML config path")
    parser.add_argument("--calendar", default=DEFAULT_CALENDAR, help="Trading day text file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--start", help="Start date YYYYMMDD; defaults to calendar first day")
    parser.add_argument("--end", help="End date YYYYMMDD; defaults to calendar last day")
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=list(INDEXES),
        help="Indexes to fetch: 300 500 1000 2000 or Zhishu300/Zhishu500/...",
    )
    parser.add_argument(
        "--same-day-returns",
        action="store_true",
        help="Use Wind same-day pctchange instead of close-to-next-close forward returns.",
    )
    parser.add_argument(
        "--snapshot-weights",
        action="store_true",
        help="Write only Wind weight snapshot dates instead of daily forward-filled weights.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Rebuild selected output files instead of appending only missing dates.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    config = load_config(args.config)
    trading_days = read_calendar(args.calendar)
    dates = resolve_date_range(trading_days, args.start, args.end)
    indexes = normalize_index_selection(args.indexes)
    output_dir = Path(os.path.expanduser(args.output_dir))
    forward_returns = not args.same_day_returns

    LOGGER.info("Date range: %s -> %s", dates.start, dates.end)
    LOGGER.info("Indexes: %s", ", ".join(f"{k}={v}" for k, v in indexes.items()))
    LOGGER.info("Mode: %s", "full refresh" if args.full_refresh else "incremental")

    plans = build_fetch_plans(
        indexes,
        output_dir,
        dates,
        trading_days,
        forward_returns=forward_returns,
        full_refresh=args.full_refresh,
    )
    if not plans:
        LOGGER.info("No missing data to fetch")
        return

    global_dates = DateRange(min(plan.dates.start for plan in plans), dates.end)
    active_indexes = {plan.key: plan.code for plan in plans}

    engine = sql_engine(config["database"])
    prices = fetch_index_prices(engine, active_indexes, global_dates)
    raw_weights = fetch_raw_weights(engine, active_indexes, global_dates.end)

    for plan in plans:
        new_returns = build_benchmark_returns(
            prices,
            plan.code,
            plan.dates,
            forward=forward_returns,
            trading_days=trading_days,
        )
        new_weights = build_daily_weights(
            raw_weights,
            plan.code,
            trading_days,
            plan.dates,
            daily_fill=not args.snapshot_weights,
        )
        returns = merge_returns(plan.existing_returns, new_returns)
        weights = merge_weights(plan.existing_weights, new_weights)
        write_outputs(output_dir, plan.key, returns, weights)


if __name__ == "__main__":
    main()
