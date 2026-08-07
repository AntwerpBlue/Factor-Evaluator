from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .metrics import annualized_return, annualized_volatility, sharpe_ratio


def _metric(values: np.ndarray, periods: int) -> dict[str, float]:
    return {
        "roi": annualized_return(values, periods),
        "vol": annualized_volatility(values, periods),
        "sharpe": sharpe_ratio(values, periods),
    }


def build_portfolio_summary(
    pnl: pl.DataFrame,
    trades: pl.DataFrame,
    periods: int,
    pf_by_time: dict[str, float],
) -> dict[str, Any]:
    holding = pnl["holding_alpha"].to_numpy()
    trading = pnl["trading_pnl"].to_numpy()
    alpha = pnl["alpha"].to_numpy()
    cumulative = np.cumsum(np.nan_to_num(alpha, nan=0.0))
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    drawdown = cumulative - peaks
    end = int(np.argmin(drawdown))
    start = int(np.argmax(cumulative[: end + 1]))
    max_dd = float(drawdown[end])
    dates = pnl["__index_level_0__"].to_list()
    asset_columns = [c for c in trades.columns if c not in {"trade_ts", "slot_key"}]
    turnover_by_time: dict[str, float] = {}
    if asset_columns:
        turnover = trades.with_columns(
            (0.5 * pl.sum_horizontal([pl.col(c).abs() for c in asset_columns])).alias("_turnover")
        )
        for slot in turnover["slot_key"].unique().to_list():
            turnover_by_time[slot] = float(
                turnover.filter(pl.col("slot_key") == slot)["_turnover"].mean()
            )
    alpha_metric = _metric(alpha, periods)
    peak_value = float(cumulative[start]) if len(cumulative) else float("nan")
    trough_value = float(cumulative[end]) if len(cumulative) else float("nan")
    return {
        **alpha_metric,
        "n_days": len(alpha),
        "holding_alpha": _metric(holding, periods),
        "trading": _metric(trading, periods),
        "calmar": alpha_metric["roi"] / abs(max_dd) if max_dd < 0 else float("nan"),
        "max_drawdown": max_dd,
        "max_drawdown_start": str(dates[start].date()),
        "max_drawdown_end": str(dates[end].date()),
        "max_drawdown_start_value": peak_value,
        "max_drawdown_end_value": trough_value,
        "annual_turnover": sum(turnover_by_time.values()) * periods,
        "turnover_by_time": turnover_by_time,
        "pf_by_time": dict(pf_by_time),
    }


def _label(name: str, metric: dict[str, float]) -> str:
    return (
        f"{name} ROI={metric['roi']:.2%} VOL={metric['vol']:.2%} "
        f"Sharpe={metric['sharpe']:.2f}"
    )


def write_performance_charts(result: Any, output: Path, prefix: str) -> list[Path]:
    pnl = result.pnl_daily
    dates = pnl["__index_level_0__"].to_list()
    periods = result.metadata["annualization"]
    cumulative_path = output / f"{prefix}_alpha_cum.png"
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True
    )
    top.plot(
        dates,
        np.cumsum(pnl["holding_alpha"].fill_null(0).to_numpy()),
        label=_label("holding_alpha", result.summary["holding_alpha"]),
    )
    for slot in [c for c in result.trading_pnl_by_time.columns if c != "__index_level_0__"]:
        values = result.trading_pnl_by_time[slot].fill_null(0).to_numpy()
        label = _label(f"trading@{slot}", _metric(values, periods))
        if slot in result.summary["pf_by_time"]:
            label += f" PF={result.summary['pf_by_time'][slot]:.4f}"
        top.plot(dates, np.cumsum(values), label=label)
    alpha_label = _label("alpha", result.summary)
    alpha_label += (
        f" Calmar={result.summary['calmar']:.2f} "
        f"maxDD={result.summary['max_drawdown']:.2%} "
        f"[{result.summary['max_drawdown_start']} - {result.summary['max_drawdown_end']}] "
        f"AnnualTVR={result.summary['annual_turnover']:.2f}"
    )
    cumulative_alpha = np.cumsum(pnl["alpha"].fill_null(0).to_numpy())
    top.plot(dates, cumulative_alpha, linewidth=2, label=alpha_label)
    dd_start = result.summary.get("max_drawdown_start")
    dd_end = result.summary.get("max_drawdown_end")
    if dd_start and dd_end:
        dd_start_dt = np.datetime64(dd_start).astype("datetime64[D]").astype(object)
        dd_end_dt = np.datetime64(dd_end).astype("datetime64[D]").astype(object)
        top.axvspan(dd_start_dt, dd_end_dt, color="#C00000", alpha=0.10, label="maxDD window")
        start_value = result.summary.get("max_drawdown_start_value")
        end_value = result.summary.get("max_drawdown_end_value")
        if start_value is not None and end_value is not None and np.isfinite(start_value) and np.isfinite(end_value):
            top.scatter([dd_start_dt, dd_end_dt], [start_value, end_value], color="#C00000", s=28, zorder=5)
            top.annotate(
                f"maxDD {result.summary['max_drawdown']:.2%}\n{dd_start} -> {dd_end}",
                xy=(dd_end_dt, end_value),
                xytext=(8, -24),
                textcoords="offset points",
                fontsize=8,
                color="#8B0000",
                arrowprops={"arrowstyle": "->", "color": "#8B0000", "linewidth": 0.8},
            )
    top.axhline(0, color="#888888", linewidth=0.7)
    top.grid(alpha=0.2)
    top.legend(fontsize=8)
    top.set_title(result.metadata.get("chart_title", prefix))

    asset_columns = [c for c in result.trades.columns if c not in {"trade_ts", "slot_key"}]
    if asset_columns:
        turnover = result.trades.with_columns(
            (0.5 * pl.sum_horizontal([pl.col(c).abs() for c in asset_columns])).alias("_turnover")
        )
        for slot in turnover["slot_key"].unique().to_list():
            subset = turnover.filter(pl.col("slot_key") == slot)
            bottom.plot(
                subset["trade_ts"].to_list(),
                subset["_turnover"].to_numpy(),
                label=f"{slot} mean={subset['_turnover'].mean():.6f}",
            )
    bottom.set_ylim(bottom=0)
    bottom.set_ylabel("turnover")
    bottom.grid(alpha=0.2)
    bottom.legend(fontsize=8)
    fig.savefig(cumulative_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    yearly_path = output / f"{prefix}_alpha_yearly_sum.png"
    alpha = pnl["alpha"].fill_null(0).to_numpy()
    years = np.asarray([date.year for date in dates])
    unique_years = np.unique(years)
    sums = [alpha[years == year].sum() for year in unique_years]
    fig, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(unique_years.astype(str), sums, color="#4472C4")
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in sums], padding=3)
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set_title(f"{prefix.replace('_intraday', '')} yearly alpha sum")
    axis.set_xlabel("year")
    fig.savefig(yearly_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [cumulative_path, yearly_path]


def write_decomposition_chart(result: Any, path: Path) -> None:
    frame = result.decomposition
    assert frame is not None
    periods = result.metadata["annualization"]
    columns = (
        "SurplusRet", "StyleRet", "IndustryRet",
        "pureAlpha_before_fee", "pureAlpha_after_fee",
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fill_null(0).to_numpy()
        axis.plot(
            np.cumsum(values),
            label=(
                f"{column} ANN={annualized_return(values, periods):.2%} "
                f"VOL={annualized_volatility(values, periods):.2%} "
                f"Sharpe={sharpe_ratio(values, periods):.2f}"
            ),
        )
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.set_title(f"attribution - {result.metadata.get('chart_title', '')}")
    axis.set_xlabel("TRADE_DT")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
