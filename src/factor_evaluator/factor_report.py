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


def write_factor_report(result: Any, path: Path) -> None:
    """Render the 14 effective panels found in the reference SVG."""
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), constrained_layout=True)
    panels = axes.flat

    for factor, daily in result.daily.items():
        panels[0].plot(np.cumsum(daily["lb"].fill_null(0).to_numpy()), label=f"{factor}-LB")
        panels[0].plot(np.cumsum(daily["ls"].fill_null(0).to_numpy()), "--", label=f"{factor}-LS")
        panels[2].plot(np.cumsum(daily["topk_power_lb"].fill_null(0).to_numpy()), label=factor)
        panels[3].plot(np.cumsum(daily["factor_return"].fill_null(0).to_numpy()), label=factor)
    panels[0].set_title("Cumulative Return")
    panels[2].set_title(
        f"PowerRank LB base | k={result.metadata.get('topk', 1.0):.2f} | "
        f"power={result.metadata.get('power', 2.0):.2f}"
    )
    panels[3].set_title("Cumulative Factor Return")

    yearly = _table(result, "02_lb_by_year")
    if not yearly.is_empty():
        subset = yearly.filter(pl.col("strategy") == "equal_weight").sort("year")
        panels[1].bar(subset["year"].cast(pl.String).to_list(), subset["annual_lb_return"].to_numpy())
        panels[1].tick_params(axis="x", rotation=45)
    panels[1].set_title("LB-return by year")

    series_specs = (
        (4, "06_bucket_return", "bucket", "annual_excess_return", "Alpha Bucket Return"),
        (5, "08_ic_decay", "lag", "ic_decay", "IC decay profile"),
        (8, "10_autocorrelation", "lag", "autocorrelation", "Factor Auto-correlation"),
    )
    for panel, table_name, x_col, y_col, title in series_specs:
        frame = _table(result, table_name)
        if not frame.is_empty():
            for factor in frame["factor"].unique().to_list():
                subset = frame.filter(pl.col("factor") == factor).sort(x_col)
                panels[panel].plot(subset[x_col].to_numpy(), subset[y_col].to_numpy(), marker="o", label=factor)
        panels[panel].set_title(title)

    coverage_plot = result.plot_data.get("coverage", {})
    for factor, frame in coverage_plot.items():
        panels[6].plot(frame["coverage"].to_numpy(), label=f"{factor} coverage")
        panels[6].plot(frame["unique"].to_numpy(), "--", label=f"{factor} unique")
    panels[6].set_ylim(0, 1.05)
    panels[6].set_title("Coverage")

    for factor, values in result.plot_data.get("factor_values", {}).items():
        panels[7].hist(values, bins=60, density=True, alpha=0.35, label=factor)
    panels[7].set_title("Distribution")
    panels[7].set_ylabel("Density")

    styles = _table(result, "09_style_correlation")
    if not styles.is_empty():
        labels = styles["style"].to_list()
        x = np.arange(len(labels))
        for offset, column in zip((-0.25, 0.0, 0.25), ("signal_corr", "fret_corr", "lbret_corr")):
            panels[9].bar(x + offset, styles[column].to_numpy(), width=0.24, label=column)
        panels[9].set_xticks(x, labels, rotation=75)
    panels[9].set_title("Style Correlation")

    for panel, style in zip(range(10, 14), ("size", "mid_cap", "liquidity", "residual_volatility")):
        frames = [frame for name, frame in result.tables.items() if name.endswith(f"bucket_exp_{style}")]
        if frames:
            frame = frames[0]
            for factor in frame["factor"].unique().to_list():
                subset = frame.filter(pl.col("factor") == factor).sort("bucket")
                panels[panel].plot(subset["bucket"].to_numpy(), subset["exposure"].to_numpy(), marker="o", label=factor)
        panels[panel].set_title(f"Bucket {style}")

    for panel in range(14):
        panels[panel].axhline(0, color="#888888", linewidth=0.7)
        panels[panel].grid(alpha=0.2)
        if panels[panel].get_legend_handles_labels()[0]:
            panels[panel].legend(fontsize=8)
    panels[14].axis("off")
    panels[15].axis("off")
    fig.suptitle(f"{', '.join(result.daily)} summary", fontsize=16)
    fig.savefig(path, format="svg")
    plt.close(fig)
