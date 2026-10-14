from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _table(result: Any, name: str) -> pl.DataFrame:
    return result.tables.get(name, pl.DataFrame())


def _date_axis(frame: pl.DataFrame) -> np.ndarray:
    values = frame["date"].to_list()
    return np.asarray([f"{v[:4]}-{v[4:6]}-{v[6:8]}" for v in values], dtype="datetime64[D]")


def _format_full_date_axis(axis: plt.Axes, dates: np.ndarray, *, max_ticks: int = 8) -> None:
    if dates.size == 0:
        return
    if dates.size <= max_ticks:
        indexes = np.arange(dates.size)
    else:
        indexes = np.unique(np.linspace(0, dates.size - 1, max_ticks).round().astype(int))
    ticks = dates[indexes]
    labels = [str(np.datetime_as_string(value, unit="D")) for value in ticks]
    if dates.size > 1:
        axis.set_xlim(dates[0], dates[-1])
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)


def _horizon(frame: pl.DataFrame) -> str:
    if frame.is_empty() or "date" not in frame.columns:
        return ""
    return f"{frame['date'].min()}-{frame['date'].max()}"


def _fmt_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def _fmt_num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def _metric_value(frame: pl.DataFrame, factor: str, strategy: str, pnl_type: str, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    subset = frame.filter(
        (pl.col("factor") == factor)
        & (pl.col("strategy") == strategy)
        & (pl.col("pnl_type") == pnl_type)
    )
    if subset.is_empty():
        return None
    value = subset[column][0]
    return float(value) if value is not None else None


def _factor_summary_value(frame: pl.DataFrame, factor: str, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    subset = frame.filter(pl.col("factor") == factor)
    if subset.is_empty():
        return None
    value = subset[column][0]
    return float(value) if value is not None else None


def _return_label(summary: pl.DataFrame, factor: str, strategy: str, pnl_type: str, name: str) -> str:
    annual = _metric_value(summary, factor, strategy, pnl_type, "annual_return")
    sharpe = _metric_value(summary, factor, strategy, pnl_type, "sharpe")
    limit = _metric_value(summary, factor, strategy, pnl_type, "annual_limit_return")
    turnover = _metric_value(summary, factor, strategy, pnl_type, "avg_turnover")
    is_net = name.endswith("net")
    if is_net:
        net_annual = _metric_value(summary, factor, strategy, pnl_type, "annual_return_net")
        net_sharpe = _metric_value(summary, factor, strategy, pnl_type, "sharpe_net")
        if net_annual is not None:
            annual = net_annual
        if net_sharpe is not None:
            sharpe = net_sharpe
    return f"{factor}-{name}:{_fmt_pct(annual)}|{_fmt_num(sharpe)}|{_fmt_pct(limit)}|{_fmt_pct(None if turnover is None else 2.0 * turnover)}"


def _distribution_text(frame: pl.DataFrame, factor: str) -> str:
    subset = frame.filter(pl.col("factor") == factor) if not frame.is_empty() else pl.DataFrame()
    if subset.is_empty():
        return ""
    row = subset.row(0, named=True)
    labels = [
        ("Mean", "mean", "{:.6g}"),
        ("Std", "std", "{:.4g}"),
        ("Skew", "skew", "{:.4g}"),
        ("Kurtosis", "kurtosis", "{:.4g}"),
        ("Min", "min", "{:.4g}"),
        ("Max", "max", "{:.4g}"),
        ("P10", "p10", "{:.4g}"),
        ("P90", "p90", "{:.4g}"),
    ]
    lines = []
    for label, key, fmt in labels:
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            lines.append(f"{label}: {fmt.format(float(value))}")
    return "\n".join(lines)


def _style_abbrev(style: str) -> str:
    mapping = {
        "beta": "BT",
        "book_to_price": "BP",
        "dividend_yield": "DY",
        "earnings_quality": "EQ",
        "earnings_variability": "EV",
        "earnings_yield": "EY",
        "growth": "GR",
        "investment_quality": "IQ",
        "leverage": "LV",
        "liquidity": "LQ",
        "longterm_reversal": "LR",
        "mid_cap": "NL",
        "momentum": "MO",
        "profitability": "PF",
        "residual_volatility": "RV",
        "size": "SZ",
        "comovement": "CM",
    }
    return mapping.get(style, style[:2].upper())


def write_factor_report(result: Any, path: Path) -> None:
    """Render the 14 effective panels found in the reference SVG."""
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), constrained_layout=True)
    panels = axes.flat
    summary = _table(result, "01_longshort_summary")

    for factor, daily in result.daily.items():
        x = _date_axis(daily)
        has_cost = float(result.metadata.get("cost_bps", 0.0) or 0.0) > 0.0
        panels[0].plot(x, np.cumsum(daily["lb"].fill_null(0).to_numpy()), label=_return_label(summary, factor, "equal_weight", "lb", "LB"))
        panels[0].plot(x, np.cumsum(daily["ls"].fill_null(0).to_numpy()), "--", label=_return_label(summary, factor, "equal_weight", "ls", "LS"))
        if has_cost:
            panels[0].plot(x, np.cumsum(daily["lb_net"].fill_null(0).to_numpy()), ":", label=_return_label(summary, factor, "equal_weight", "lb", "LBnet"))
            panels[0].plot(x, np.cumsum(daily["ls_net"].fill_null(0).to_numpy()), ":", label=_return_label(summary, factor, "equal_weight", "ls", "LSnet"))
        panels[2].plot(x, np.cumsum(daily["topk_power_lb"].fill_null(0).to_numpy()), label=_return_label(summary, factor, "topk_power", "lb", "PowerLB"))
        if has_cost and "topk_power_lb_net" in daily.columns:
            panels[2].plot(x, np.cumsum(daily["topk_power_lb_net"].fill_null(0).to_numpy()), "--", label=_return_label(summary, factor, "topk_power", "lb", "PowerLBnet"))
        factor_summary = _table(result, "05_factor_return_summary")
        annual = _factor_summary_value(factor_summary, factor, "annual_return")
        sharpe = _factor_summary_value(factor_summary, factor, "sharpe")
        panels[3].plot(x, np.cumsum(daily["factor_return"].fill_null(0).to_numpy()), label=f"{factor} |{_fmt_pct(annual)}|{_fmt_num(sharpe)}")
    first_daily = next(iter(result.daily.values()), pl.DataFrame())
    horizon = _horizon(first_daily)
    panels[0].set_title(f"Cumulative Return {horizon}")
    panels[2].set_title(
        f"PowerRank LB base | k={result.metadata.get('topk', 1.0):.2f} | "
        f"power={result.metadata.get('power', 2.0):.2f}"
    )
    panels[3].set_title(f"Cumulative Factor Return {horizon}")

    yearly = _table(result, "02_lb_by_year")
    if not yearly.is_empty():
        subset = yearly.filter(pl.col("strategy") == "equal_weight").sort("year")
        panels[1].bar(subset["year"].cast(pl.String).to_list(), subset["annual_lb_return"].to_numpy())
        mean_return = float(subset["annual_lb_return"].mean()) if subset.height else float("nan")
        panels[1].legend([f"LB|{_fmt_pct(mean_return)}"], fontsize=8)
        panels[1].tick_params(axis="x", rotation=45)
    panels[1].set_title(f"LB-return by year {horizon}")

    bucket = _table(result, "06_bucket_return")
    if not bucket.is_empty():
        width = 0.8 / max(len(bucket["factor"].unique()), 1)
        for index, factor in enumerate(bucket["factor"].unique().to_list()):
            subset = bucket.filter(pl.col("factor") == factor).sort("bucket")
            x_bucket = subset["bucket"].to_numpy()
            y = subset["annual_excess_return"].to_numpy()
            panels[4].bar(x_bucket + (index - 0.5) * width, y, width=width, label=factor)
            value_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else float("nan")
            panels[4].text(0.02, 0.95 - 0.08 * index, f"{factor}|RANGE:{value_range:.2%}", transform=panels[4].transAxes, va="top", fontsize=8)
    panels[4].set_title("Alpha Bucket Return")

    ic_decay = _table(result, "08_ic_decay")
    ic_summary = _table(result, "07_ic_summary")
    if not ic_decay.is_empty():
        for factor in ic_decay["factor"].unique().to_list():
            subset = ic_decay.filter(pl.col("factor") == factor).sort("lag")
            ic = _factor_summary_value(ic_summary, factor, "ic")
            icir = _factor_summary_value(ic_summary, factor, "icir")
            panels[5].plot(subset["lag"].to_numpy(), subset["ic_decay"].to_numpy(), marker="o", label=f"{factor}@{horizon.replace('-', '_')}|{_fmt_num(ic)}|{_fmt_num(icir)}")
    panels[5].set_title("IC decay profile")

    coverage_plot = result.plot_data.get("coverage", {})
    coverage_table = _table(result, "03_coverage")
    for factor, frame in coverage_plot.items():
        if frame.is_empty():
            continue
        # Coverage records may be sparse; use their dates if present.
        x = _date_axis(frame) if "date" in frame.columns else np.arange(frame.height)
        cov = _factor_summary_value(coverage_table, factor, "mean_coverage")
        unique = _factor_summary_value(coverage_table, factor, "mean_unique")
        panels[6].plot(x, frame["coverage"].to_numpy(), label=f"coverage|{_fmt_pct(cov)}")
        panels[6].plot(x, frame["unique"].to_numpy(), "--", label=f"unique|{_fmt_pct(unique)}")
    panels[6].set_ylim(0, 1.05)
    panels[6].set_title(f"Coverage {horizon}")

    distribution = _table(result, "04_distribution")
    for index, (factor, values) in enumerate(result.plot_data.get("factor_values", {}).items()):
        panels[7].hist(values, bins=60, density=True, alpha=0.35, label=factor)
        text = _distribution_text(distribution, factor)
        if text:
            panels[7].text(0.02, 0.95 - 0.02 * index, text, transform=panels[7].transAxes, va="top", fontsize=8)
    panels[7].set_title(f"Distribution {horizon}")
    panels[7].set_ylabel("Density")

    autocorr = _table(result, "10_autocorrelation")
    if not autocorr.is_empty():
        for factor in autocorr["factor"].unique().to_list():
            subset = autocorr.filter(pl.col("factor") == factor).sort("lag")
            panels[8].plot(subset["lag"].to_numpy(), subset["autocorrelation"].to_numpy(), marker="o", label=f"{factor}@{horizon.replace('-', '_')}")
    panels[8].set_title("Factor Auto-correlation")

    styles = _table(result, "09_style_correlation")
    if not styles.is_empty():
        labels = [_style_abbrev(style) for style in styles["style"].to_list()]
        x = np.arange(len(labels))
        for offset, column in zip((-0.25, 0.0, 0.25), ("signal_corr", "fret_corr", "lbret_corr")):
            panels[9].bar(x + offset, styles[column].to_numpy(), width=0.24, label=column)
        panels[9].set_xticks(x, labels, rotation=75)
    panels[9].set_title(f"Style Correlation {horizon}")

    for panel, style in zip(range(10, 14), ("size", "mid_cap", "liquidity", "residual_volatility")):
        frames = [frame for name, frame in result.tables.items() if name.endswith(f"bucket_exp_{style}")]
        if frames:
            frame = frames[0]
            factors = frame["factor"].unique().to_list()
            factor_count = max(len(factors), 1)
            width = 0.8 / factor_count
            all_buckets: list[int] = []
            for index, factor in enumerate(factors):
                subset = frame.filter(pl.col("factor") == factor).sort("bucket")
                buckets = subset["bucket"].to_numpy()
                exposure = subset["exposure"].to_numpy()
                centered = exposure - np.nanmean(exposure)
                offset = (index - (factor_count - 1) / 2.0) * width
                panels[panel].bar(buckets + offset, centered, width=width, label=factor)
                all_buckets.extend(int(bucket) for bucket in buckets)
            if all_buckets:
                unique_buckets = sorted(set(all_buckets))
                panels[panel].set_xticks(unique_buckets)
        panels[panel].set_title(f"Bucket {style}")

    for panel in range(14):
        panels[panel].axhline(0, color="#888888", linewidth=0.7)
        panels[panel].grid(alpha=0.2)
        if panels[panel].get_legend_handles_labels()[0]:
            panels[panel].legend(fontsize=8)
    full_dates = _date_axis(first_daily) if not first_daily.is_empty() else np.asarray([], dtype="datetime64[D]")
    for panel in (0, 2, 3):
        _format_full_date_axis(panels[panel], full_dates)
    for panel in (0, 2, 3, 6):
        panels[panel].tick_params(axis="x", rotation=30)
    panels[14].axis("off")
    panels[15].axis("off")
    fig.suptitle(f"{', '.join(result.daily)} summary", fontsize=16)
    fig.savefig(path, format="svg")
    plt.close(fig)
