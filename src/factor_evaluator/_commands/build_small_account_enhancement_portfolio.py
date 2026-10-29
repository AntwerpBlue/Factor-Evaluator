"""Build a small-account 60/40 HS300-plus-alpha holdings list.

The default portfolio is 60% HS300 index instrument and 40% a concentrated
stock enhancement sleeve. The stock sleeve solves a long-only optimization over
factor-ranked candidates, then rounds to board lots for a cash account.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize


from factor_evaluator.data import normalize_panel


LOGGER = logging.getLogger("build_small_account_enhancement_portfolio")

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

BENCHMARK_WEIGHT_FILES = {
    "300": "benchmark_300_weights.parquet",
    "500": "benchmark_500_weights.parquet",
    "1000": "benchmark_1000_weights.parquet",
    "2000": "benchmark_2000_weights.parquet",
    "Zhishu300": "benchmark_300_weights.parquet",
    "Zhishu500": "benchmark_500_weights.parquet",
    "Zhishu1000": "benchmark_1000_weights.parquet",
    "Zhishu2000": "benchmark_2000_weights.parquet",
}


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


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
    values = [column for column in frame.columns if column not in {"date", "asset"}]
    if len(values) != 1:
        raise ValueError(f"{label} requires one value column or an explicit column argument")
    return values[0]


def latest_panel(frame: pl.DataFrame, date: str, *, require_unique: bool = False) -> pl.DataFrame:
    panel = normalize_panel(frame, require_unique=require_unique).filter(pl.col("date") <= date)
    if panel.is_empty():
        raise ValueError(f"No rows on or before {date}")
    return panel.sort(["asset", "date"]).group_by("asset", maintain_order=True).tail(1)


def normalize_date_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.replace_all("-", "").str.slice(0, 8)


def load_prices(args: argparse.Namespace, date: str) -> pl.DataFrame:
    execution_path = expand(args.execution_prices) if args.execution_prices else None
    if execution_path and execution_path.exists():
        prices = latest_panel(load_frame(execution_path), date)
        price_col = args.price_column if args.price_column in prices.columns else "close"
        if price_col not in prices.columns:
            raise ValueError(f"Price column not found in {execution_path}: {args.price_column}")
        return prices.select("asset", pl.col(price_col).cast(pl.Float64).alias("price"), pl.col("date").alias("price_date"))

    wind_path = expand(args.wind_baseinfo)
    wind = pl.read_parquet(wind_path).select(
        normalize_date_expr("TRADE_DT").alias("date"),
        pl.col("S_INFO_WINDCODE").cast(pl.String).str.slice(0, 6).alias("asset"),
        pl.col(args.wind_price_column).cast(pl.Float64).alias("price"),
    )
    return (
        wind.filter(pl.col("date") <= date)
        .sort(["asset", "date"])
        .group_by("asset", maintain_order=True)
        .tail(1)
        .rename({"date": "price_date"})
    )


def filter_flag(frame: pl.DataFrame, path: Path, column: str, date: str) -> pl.DataFrame:
    if not path.exists():
        LOGGER.warning("%s not found; skipping %s filter", path, column)
        return frame
    flag = latest_panel(load_frame(path), date)
    if column not in flag.columns:
        raise ValueError(f"{column} not found in {path}")
    keep = flag.select("asset", pl.col(column).cast(pl.Boolean).alias("_keep"))
    return frame.join(keep, on="asset", how="inner").filter(pl.col("_keep")).drop("_keep")


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.nan_to_num((values - mean) / std, nan=0.0)


def exposure_matrix(
    assets: list[str],
    frame: pl.DataFrame | None,
    columns: list[str],
) -> tuple[np.ndarray | None, list[str]]:
    if frame is None or frame.is_empty():
        return None, []
    available = [column for column in columns if column in frame.columns]
    if not available:
        return None, []
    lookup = {row["asset"]: row for row in frame.select("asset", *available).iter_rows(named=True)}
    matrix = np.asarray(
        [[float(lookup.get(asset, {}).get(column, 0.0) or 0.0) for column in available] for asset in assets],
        dtype=float,
    )
    return matrix, available


def style_exposure_matrix(assets: list[str], frame: pl.DataFrame | None) -> tuple[np.ndarray | None, list[str]]:
    matrix, columns = exposure_matrix(assets, frame, STYLE_COLUMNS)
    if matrix is None:
        return None, []
    return np.column_stack([zscore(matrix[:, idx]) for idx in range(matrix.shape[1])]), columns


def benchmark_target(
    benchmark: pl.DataFrame,
    exposures: pl.DataFrame | None,
    columns: list[str],
    *,
    zscore_like: bool,
) -> np.ndarray | None:
    if exposures is None or exposures.is_empty() or not columns:
        return None
    assets = benchmark["asset"].to_list()
    matrix, actual_columns = exposure_matrix(assets, exposures, columns)
    if matrix is None:
        return None
    if zscore_like:
        matrix = np.column_stack([zscore(matrix[:, idx]) for idx in range(matrix.shape[1])])
    weights = benchmark["weight"].to_numpy()
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))
    return weights @ matrix[:, : len(actual_columns)]


def optimize_sleeve(
    alpha: np.ndarray,
    style_matrix: np.ndarray | None,
    style_target: np.ndarray | None,
    industry_matrix: np.ndarray | None,
    industry_target: np.ndarray | None,
    benchmark_weights: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    n_assets = len(alpha)
    if n_assets == 0:
        raise ValueError("No optimization candidates")
    if args.max_sleeve_weight * n_assets < 1.0 - 1e-12:
        raise ValueError("max_sleeve_weight is too low for the selected number of stocks")

    alpha = zscore(alpha)
    benchmark_weights = benchmark_weights / benchmark_weights.sum() if benchmark_weights.sum() > 0 else np.full(n_assets, 1 / n_assets)

    def objective(weight: np.ndarray) -> float:
        value = -float(alpha @ weight)
        value += args.benchmark_penalty * float(np.sum((weight - benchmark_weights) ** 2))
        if style_matrix is not None and style_target is not None:
            diff = style_matrix.T @ weight - style_target
            value += args.style_penalty * float(diff @ diff)
        if industry_matrix is not None and industry_target is not None:
            diff = industry_matrix.T @ weight - industry_target
            value += args.industry_penalty * float(diff @ diff)
        return value

    def jac(weight: np.ndarray) -> np.ndarray:
        grad = -alpha + 2.0 * args.benchmark_penalty * (weight - benchmark_weights)
        if style_matrix is not None and style_target is not None:
            grad += 2.0 * args.style_penalty * (style_matrix @ (style_matrix.T @ weight - style_target))
        if industry_matrix is not None and industry_target is not None:
            grad += 2.0 * args.industry_penalty * (industry_matrix @ (industry_matrix.T @ weight - industry_target))
        return grad

    initial = np.minimum(np.full(n_assets, 1.0 / n_assets), args.max_sleeve_weight)
    initial = initial / initial.sum()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=jac,
        bounds=[(0.0, args.max_sleeve_weight)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0), "jac": lambda weight: np.ones(n_assets)}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Stock sleeve optimization failed: {result.message}")
    return result.x / result.x.sum()


def lot_round_stocks(frame: pl.DataFrame, capital: float, sleeve_weight: float, args: argparse.Namespace) -> pl.DataFrame:
    sleeve_cash = capital * sleeve_weight
    rows = frame.sort("sleeve_weight", descending=True).to_dicts()
    for row in rows:
        target_amount = sleeve_cash * float(row["sleeve_weight"])
        lot_value = float(row["price"]) * args.stock_lot_size
        row["shares"] = 0
        row["notional"] = 0.0
        row["target_amount"] = target_amount
        row["lot_value"] = lot_value

    cash = sleeve_cash
    max_notional = sleeve_cash * args.max_sleeve_weight * 1.05
    if args.seed_one_lot:
        seeded = sorted(rows, key=lambda row: (float(row["alpha_z"]), float(row["sleeve_weight"])), reverse=True)
        for row in seeded:
            if row["lot_value"] <= cash and row["lot_value"] <= max_notional:
                row["shares"] = args.stock_lot_size
                row["notional"] = row["lot_value"]
                cash -= row["lot_value"]

    for row in rows:
        if row["lot_value"] <= 0:
            continue
        remaining_target = max(row["target_amount"] - row["notional"], 0.0)
        additional_lots = int(np.floor(remaining_target / row["lot_value"]))
        additional_shares = additional_lots * args.stock_lot_size
        additional_notional = additional_shares * float(row["price"])
        if additional_notional > cash:
            additional_shares = int(np.floor(cash / row["lot_value"]) * args.stock_lot_size)
            additional_notional = additional_shares * float(row["price"])
        if row["notional"] + additional_notional > max_notional:
            room = max(max_notional - row["notional"], 0.0)
            additional_shares = int(np.floor(room / row["lot_value"]) * args.stock_lot_size)
            additional_notional = additional_shares * float(row["price"])
        row["shares"] += additional_shares
        row["notional"] += additional_notional
        cash -= additional_notional

    while True:
        affordable = [row for row in rows if row["lot_value"] <= cash and row["notional"] + row["lot_value"] <= max_notional]
        if not affordable:
            break
        affordable.sort(key=lambda row: (float(row["alpha_z"]), -abs((row["notional"] + row["lot_value"]) - row["target_amount"])), reverse=True)
        row = affordable[0]
        row["shares"] += args.stock_lot_size
        row["notional"] += row["lot_value"]
        cash -= row["lot_value"]
    result = pl.DataFrame(rows).with_columns(
        (pl.col("notional") / capital).alias("actual_overall_weight"),
        (pl.col("notional") / sleeve_cash).alias("actual_sleeve_weight"),
        (pl.col("target_amount") / capital).alias("target_overall_weight"),
    )
    return result.sort("notional", descending=True)


def index_row(args: argparse.Namespace, capital: float) -> dict[str, object]:
    target_amount = capital * args.index_weight
    row: dict[str, object] = {
        "type": "index",
        "asset": args.index_instrument,
        "target_overall_weight": args.index_weight,
        "target_amount": target_amount,
        "price": args.index_price,
        "shares": None,
        "notional": target_amount,
        "actual_overall_weight": args.index_weight,
    }
    if args.index_price is not None and args.index_price > 0:
        shares = int(np.floor(target_amount / (args.index_price * args.index_lot_size)) * args.index_lot_size)
        notional = shares * args.index_price
        row.update({"shares": shares, "notional": notional, "actual_overall_weight": notional / capital})
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 60/40 HS300 plus concentrated stock enhancement holdings list.")
    parser.add_argument("--factor-values", required=True, help="Factor parquet/csv with date + code/header + factor column")
    parser.add_argument("--factor-column", default="alpha")
    parser.add_argument("--date", help="Portfolio date YYYYMMDD; defaults to latest factor date")
    parser.add_argument("--capital", type=float, default=200000.0)
    parser.add_argument("--index-weight", type=float, default=0.60)
    parser.add_argument("--stock-sleeve-weight", type=float, default=0.40)
    parser.add_argument("--holdings", type=int, default=15)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--benchmark", default="300", choices=sorted(BENCHMARK_WEIGHT_FILES))
    parser.add_argument("--data-dir", default="~/data")
    parser.add_argument("--benchmark-weights", help="Override benchmark weights file")
    parser.add_argument("--universe", default="~/data/factor_evaluator_cache/local/universe.parquet")
    parser.add_argument("--tradability", default="~/data/factor_evaluator_cache/local/tradability.parquet")
    parser.add_argument("--execution-prices", default="~/data/factor_evaluator_cache/local/execution_prices.parquet")
    parser.add_argument("--wind-baseinfo", default="~/data/wind_baseinfo.parquet")
    parser.add_argument("--price-column", default="close")
    parser.add_argument("--wind-price-column", default="S_DQ_CLOSE")
    parser.add_argument("--style-exposures", default="~/data/factor_evaluator_cache/local/style_exposures.parquet")
    parser.add_argument("--industry-exposures", default="~/data/factor_evaluator_cache/local/industry_exposures.parquet")
    parser.add_argument("--allow-off-benchmark", action="store_true")
    parser.add_argument("--max-sleeve-weight", type=float, default=0.12, help="Max weight within the 40%% stock sleeve")
    parser.add_argument("--style-penalty", type=float, default=5.0)
    parser.add_argument("--industry-penalty", type=float, default=2.0)
    parser.add_argument("--benchmark-penalty", type=float, default=0.05)
    parser.add_argument("--stock-lot-size", type=int, default=100)
    parser.add_argument("--no-seed-one-lot", dest="seed_one_lot", action="store_false", help="Do not force one board lot for each selected stock before filling target weights.")
    parser.set_defaults(seed_one_lot=True)
    parser.add_argument("--index-instrument", default="510300.SH")
    parser.add_argument("--index-price", type=float, help="ETF/index instrument price; if omitted, index row is amount-only")
    parser.add_argument("--index-lot-size", type=int, default=100)
    parser.add_argument("--output", default="~/data/small_account_portfolios/hs300_plus_alpha.parquet")
    parser.add_argument("--summary-output", help="Optional JSON summary path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    if not np.isclose(args.index_weight + args.stock_sleeve_weight, 1.0):
        raise ValueError("index_weight + stock_sleeve_weight must equal 1")
    if args.holdings < 1:
        raise ValueError("holdings must be positive")

    data_dir = expand(args.data_dir)
    factor = normalize_panel(load_frame(expand(args.factor_values)))
    factor_col = value_column(factor, args.factor_column, "factor_values")
    date = args.date.replace("-", "") if args.date else str(factor["date"].max())
    factor = factor.filter(pl.col("date") == date).select("asset", pl.col(factor_col).cast(pl.Float64).alias("alpha"))
    if factor.is_empty():
        raise ValueError(f"No factor values for {date}")

    benchmark_path = expand(args.benchmark_weights) if args.benchmark_weights else data_dir / BENCHMARK_WEIGHT_FILES[args.benchmark]
    benchmark = latest_panel(load_frame(benchmark_path), date)
    weight_col = "weight" if "weight" in benchmark.columns else value_column(benchmark, None, "benchmark_weights")
    benchmark = benchmark.select("asset", pl.col(weight_col).cast(pl.Float64).alias("weight"))
    benchmark = benchmark.with_columns((pl.col("weight") / pl.col("weight").sum()).alias("weight"))

    candidates = factor
    if not args.allow_off_benchmark:
        candidates = candidates.join(benchmark.select("asset"), on="asset", how="inner")
    universe_path = expand(args.universe)
    if universe_path.exists():
        candidates = filter_flag(candidates, universe_path, "eligible", date)
    tradability_path = expand(args.tradability)
    if tradability_path.exists():
        candidates = filter_flag(candidates, tradability_path, "tradable", date)

    prices = load_prices(args, date)
    candidates = candidates.join(prices, on="asset", how="inner").filter(pl.col("price") > 0)
    candidates = candidates.with_columns(pl.col("alpha").rank("average").alias("alpha_rank"))
    candidates = candidates.join(benchmark, on="asset", how="left").with_columns(pl.col("weight").fill_null(0.0).alias("benchmark_weight"))
    max_buyable_price = args.capital * args.stock_sleeve_weight * args.max_sleeve_weight / args.stock_lot_size
    candidates = candidates.filter(pl.col("price") <= max_buyable_price)
    if candidates.height < args.holdings:
        raise ValueError(f"Only {candidates.height} buyable candidates remain; reduce holdings or max_sleeve_weight")

    style_frame = latest_panel(load_frame(expand(args.style_exposures)), date) if expand(args.style_exposures).exists() else None
    industry_frame = latest_panel(load_frame(expand(args.industry_exposures)), date) if expand(args.industry_exposures).exists() else None
    style_target = benchmark_target(benchmark, style_frame, STYLE_COLUMNS, zscore_like=True)
    industry_columns = [column for column in (industry_frame.columns if industry_frame is not None else []) if column not in {"date", "asset"}]
    industry_target = benchmark_target(benchmark, industry_frame, industry_columns, zscore_like=False)

    pool = candidates.sort("alpha", descending=True).head(max(args.candidate_pool, args.holdings))
    pool_assets = pool["asset"].to_list()
    pool_style, style_columns = style_exposure_matrix(pool_assets, style_frame)
    pool_industry, actual_industry_columns = exposure_matrix(pool_assets, industry_frame, industry_columns)
    pool_weight = optimize_sleeve(
        pool["alpha"].to_numpy(),
        pool_style,
        style_target[: len(style_columns)] if style_target is not None and style_columns else None,
        pool_industry,
        industry_target[: len(actual_industry_columns)] if industry_target is not None and actual_industry_columns else None,
        pool["benchmark_weight"].to_numpy(),
        args,
    )
    selected = pool.with_columns(pl.Series("preselect_weight", pool_weight)).sort("preselect_weight", descending=True).head(args.holdings)

    selected_assets = selected["asset"].to_list()
    selected_style, style_columns = style_exposure_matrix(selected_assets, style_frame)
    selected_industry, actual_industry_columns = exposure_matrix(selected_assets, industry_frame, industry_columns)
    sleeve_weight = optimize_sleeve(
        selected["alpha"].to_numpy(),
        selected_style,
        style_target[: len(style_columns)] if style_target is not None and style_columns else None,
        selected_industry,
        industry_target[: len(actual_industry_columns)] if industry_target is not None and actual_industry_columns else None,
        selected["benchmark_weight"].to_numpy(),
        args,
    )
    alpha_z = zscore(selected["alpha"].to_numpy())
    selected = selected.with_columns(
        pl.Series("sleeve_weight", sleeve_weight),
        pl.Series("alpha_z", alpha_z),
        (pl.Series("sleeve_weight", sleeve_weight) * args.stock_sleeve_weight).alias("target_overall_weight"),
    )
    rounded = lot_round_stocks(selected, args.capital, args.stock_sleeve_weight, args)
    stock_rows = rounded.with_columns(
        pl.lit("stock").alias("type"),
        pl.lit(date).alias("date"),
    ).select(
        "date", "type", "asset", "price", "price_date", "alpha", "alpha_z", "benchmark_weight",
        "sleeve_weight", "target_overall_weight", "target_amount", "shares", "notional",
        "actual_sleeve_weight", "actual_overall_weight",
    )

    index = index_row(args, args.capital)
    index_frame = pl.DataFrame([{**index, "date": date, "price_date": date, "alpha": None, "alpha_z": None, "benchmark_weight": None, "sleeve_weight": None, "actual_sleeve_weight": None}]).select(stock_rows.columns)
    output_frame = pl.concat([index_frame, stock_rows], how="vertical_relaxed")
    output = expand(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.write_parquet(output)
    csv_path = output.with_suffix(".csv")
    output_frame.write_csv(csv_path)

    total_notional = float(output_frame["notional"].fill_null(0).sum())
    stock_notional = float(stock_rows["notional"].sum())
    summary = {
        "date": date,
        "capital": args.capital,
        "index_instrument": args.index_instrument,
        "index_target_weight": args.index_weight,
        "stock_sleeve_target_weight": args.stock_sleeve_weight,
        "requested_stock_count": args.holdings,
        "actual_stock_count": int(stock_rows.filter(pl.col("shares") > 0).height),
        "stock_target_cash": args.capital * args.stock_sleeve_weight,
        "stock_notional": stock_notional,
        "total_notional": total_notional,
        "cash_left": args.capital - total_notional,
        "output": str(output),
        "csv_output": str(csv_path),
    }
    summary_path = expand(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", output, csv_path)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
