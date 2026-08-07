from __future__ import annotations

from dataclasses import dataclass

import polars as pl


DATE_ALIASES = ("date",)
ASSET_ALIASES = ("asset", "code", "header")


def _find_column(frame: pl.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    for column in aliases:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing {label} column; accepted names: {aliases}")


def normalize_panel(
    frame: pl.DataFrame,
    *,
    require_unique: bool = True,
) -> pl.DataFrame:
    """Normalize long panel keys to ``date`` and ``asset``."""
    date_col = _find_column(frame, DATE_ALIASES, "date")
    asset_col = _find_column(frame, ASSET_ALIASES, "asset")
    rename = {}
    if date_col != "date":
        rename[date_col] = "date"
    if asset_col != "asset":
        rename[asset_col] = "asset"
    result = frame.rename(rename).with_columns(
        pl.col("date").cast(pl.String),
        pl.col("asset").cast(pl.String).str.slice(0, 6),
    )
    invalid_dates = result.filter(~pl.col("date").str.contains(r"^\d{8}$")).height
    if invalid_dates:
        raise ValueError(
            f"Panel contains {invalid_dates} invalid dates; expected YYYYMMDD strings"
        )
    invalid_assets = result.filter(
        ~pl.col("asset").str.contains(r"^\d{6}(\.[A-Za-z]{2})?$")
    ).height
    if invalid_assets:
        raise ValueError(
            f"Panel contains {invalid_assets} invalid security codes; "
            "expected 6 digits or 9 characters such as 000001.SZ"
        )
    if require_unique:
        duplicated = (
            result.group_by(["date", "asset"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicated:
            raise ValueError(f"Panel contains {duplicated} duplicate date/asset keys")
    return result.sort(["date", "asset"])


def normalize_daily(frame: pl.DataFrame, value_name: str) -> pl.DataFrame:
    date_col = _find_column(frame, DATE_ALIASES, "date")
    result = frame.rename({date_col: "date"}) if date_col != "date" else frame
    value_columns = [column for column in result.columns if column != "date"]
    if value_name not in result.columns:
        if len(value_columns) != 1:
            raise ValueError(f"Expected a '{value_name}' column")
        result = result.rename({value_columns[0]: value_name})
    daily = result.select(
        pl.col("date").cast(pl.String),
        pl.col(value_name).cast(pl.Float64),
    ).sort("date")
    invalid_dates = daily.filter(~pl.col("date").str.contains(r"^\d{8}$")).height
    if invalid_dates:
        raise ValueError(
            f"Daily data contains {invalid_dates} invalid dates; expected YYYYMMDD strings"
        )
    return daily


@dataclass(frozen=True)
class FactorDataset:
    """Inputs for a factor test.

    ``returns`` must contain the return earned *after* a position is formed on
    its keyed date. This explicit contract keeps alignment and look-ahead
    handling outside the statistics engine.
    """

    factors: pl.DataFrame
    returns: pl.DataFrame
    benchmark: pl.DataFrame
    universe: pl.DataFrame | None = None
    styles: pl.DataFrame | None = None

    def normalized(self) -> "FactorDataset":
        factors = normalize_panel(self.factors)
        returns = normalize_panel(self.returns)
        if "return" not in returns.columns:
            values = [c for c in returns.columns if c not in {"date", "asset"}]
            if len(values) != 1:
                raise ValueError("returns must contain exactly one value column named 'return'")
            returns = returns.rename({values[0]: "return"})
        returns = returns.with_columns(pl.col("return").cast(pl.Float64))
        benchmark = normalize_daily(self.benchmark, "benchmark_return")
        universe = normalize_panel(self.universe) if self.universe is not None else None
        styles = normalize_panel(self.styles) if self.styles is not None else None
        return FactorDataset(factors, returns, benchmark, universe, styles)


@dataclass(frozen=True)
class PortfolioDataset:
    """Inputs for a portfolio simulation, all in long form."""

    returns: pl.DataFrame
    benchmark: pl.DataFrame
    benchmark_weights: pl.DataFrame | None = None
    styles: pl.DataFrame | None = None
    industry_exposures: pl.DataFrame | None = None
    style_returns: pl.DataFrame | None = None
    industry_returns: pl.DataFrame | None = None

    def normalized(self) -> "PortfolioDataset":
        returns = normalize_panel(self.returns)
        if "return" not in returns.columns:
            values = [c for c in returns.columns if c not in {"date", "asset"}]
            if len(values) != 1:
                raise ValueError("returns must contain exactly one value column named 'return'")
            returns = returns.rename({values[0]: "return"})
        benchmark = normalize_daily(self.benchmark, "benchmark_return")
        benchmark_weights = (
            normalize_panel(self.benchmark_weights)
            if self.benchmark_weights is not None
            else None
        )
        styles = normalize_panel(self.styles) if self.styles is not None else None
        industry_exposures = (
            normalize_panel(self.industry_exposures)
            if self.industry_exposures is not None else None
        )
        style_returns = self.style_returns
        industry_returns = self.industry_returns
        return PortfolioDataset(
            returns=returns,
            benchmark=benchmark,
            benchmark_weights=benchmark_weights,
            styles=styles,
            industry_exposures=industry_exposures,
            style_returns=style_returns,
            industry_returns=industry_returns,
        )






