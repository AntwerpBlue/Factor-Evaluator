from __future__ import annotations

import numpy as np
import polars as pl

from factor_evaluator import FactorBacktestConfig, FactorBacktester, FactorDataset
from factor_evaluator.factor_report import _return_label


def test_limit_return_contribution_uses_signed_portfolio_weights():
    returns = np.asarray([0.01, 0.02, 0.03, 0.04])
    labels = np.asarray([0.0, -1.0, 0.0, 1.0])

    equal_long = np.asarray([0.0, 0.0, 0.5, 0.5])
    long_short = np.asarray([-0.25, -0.25, 0.25, 0.25])
    power_rank = np.asarray([0.0, 1.0 / 14.0, 4.0 / 14.0, 9.0 / 14.0])

    assert np.isclose(
        FactorBacktester._limit_return_contribution(equal_long, returns, labels), 0.02
    )
    assert np.isclose(
        FactorBacktester._limit_return_contribution(long_short, returns, labels), 0.005
    )
    assert np.isclose(
        FactorBacktester._limit_return_contribution(power_rank, returns, labels),
        0.02 / 14.0 + 0.36 / 14.0,
    )


def test_limit_return_contribution_distinguishes_zero_from_unknown():
    weights = np.asarray([0.5, 0.5])
    returns = np.asarray([0.01, 0.02])

    assert FactorBacktester._limit_return_contribution(
        weights, returns, np.asarray([0.0, 0.0])
    ) == 0.0
    assert FactorBacktester._limit_return_contribution(
        weights, returns, np.asarray([0.0, np.nan])
    ) is None


def test_limit_contribution_flows_to_equal_weight_and_power_rank_summary():
    date = "20240102"
    assets = [f"{index:06d}.SZ" for index in range(4)]
    factors = pl.DataFrame(
        {"date": [date] * 4, "header": assets, "alpha": [0.0, 1.0, 2.0, 3.0]}
    )
    returns = pl.DataFrame(
        {"date": [date] * 4, "header": assets, "return": [0.01, 0.02, 0.03, 0.04]}
    )
    benchmark = pl.DataFrame({"date": [date], "benchmark_return": [0.0]})
    labels = pl.DataFrame(
        {"date": [date] * 4, "header": assets, "limit_label": [0, -1, 0, 1]}
    )

    result = FactorBacktester(
        FactorBacktestConfig(
            buckets=2,
            topk=1.0,
            power=2.0,
            annualization=1,
            max_lag=1,
            min_assets=4,
        )
    ).run(FactorDataset(factors, returns, benchmark, limit_labels=labels))

    row = result.daily["alpha"].row(0, named=True)
    assert np.isclose(row["lb_limit"], 0.02)
    assert np.isclose(row["ls_limit"], 0.005)
    assert np.isclose(row["topk_power_lb_limit"], 0.38 / 14.0)

    power_summary = result.tables["01_longshort_summary"].filter(
        (pl.col("factor") == "alpha")
        & (pl.col("strategy") == "topk_power")
        & (pl.col("pnl_type") == "lb")
    )
    assert np.isclose(power_summary["annual_limit_return"][0], 0.38 / 14.0)
    assert result.metadata["limit_label_coverage"] == 1.0
    assert result.metadata["limit_label_hit_rate"] == 0.5


def test_missing_limit_mask_is_reported_as_unknown_instead_of_zero():
    date = "20240102"
    assets = [f"{index:06d}.SZ" for index in range(4)]
    factors = pl.DataFrame(
        {"date": [date] * 4, "header": assets, "alpha": [0.0, 1.0, 2.0, 3.0]}
    )
    returns = pl.DataFrame(
        {"date": [date] * 4, "header": assets, "return": [0.01, 0.02, 0.03, 0.04]}
    )
    benchmark = pl.DataFrame({"date": [date], "benchmark_return": [0.0]})

    result = FactorBacktester(
        FactorBacktestConfig(buckets=2, max_lag=1, min_assets=4)
    ).run(FactorDataset(factors, returns, benchmark))

    row = result.daily["alpha"].row(0, named=True)
    assert row["lb_limit"] is None
    assert row["ls_limit"] is None
    assert row["topk_power_lb_limit"] is None
    assert result.metadata["limit_labels_supplied"] is False


def test_partial_limit_mask_makes_aggregate_contribution_unknown():
    dates = ["20240102", "20240103"]
    assets = [f"{index:06d}.SZ" for index in range(4)]
    factor_rows = []
    return_rows = []
    label_rows = []
    for date_index, date in enumerate(dates):
        for asset_index, asset in enumerate(assets):
            factor_rows.append(
                {"date": date, "header": asset, "alpha": float(asset_index)}
            )
            return_rows.append(
                {
                    "date": date,
                    "header": asset,
                    "return": 0.01 * (asset_index + 1),
                }
            )
            if not (date_index == 1 and asset_index == 3):
                label_rows.append(
                    {
                        "date": date,
                        "header": asset,
                        "limit_label": 1 if asset_index == 3 else 0,
                    }
                )

    result = FactorBacktester(
        FactorBacktestConfig(
            buckets=2, annualization=1, max_lag=1, min_assets=4
        )
    ).run(
        FactorDataset(
            pl.DataFrame(factor_rows),
            pl.DataFrame(return_rows),
            pl.DataFrame(
                {"date": dates, "benchmark_return": [0.0, 0.0]}
            ),
            limit_labels=pl.DataFrame(label_rows),
        )
    )

    daily = result.daily["alpha"]
    assert np.isclose(daily["lb_limit"][0], 0.02)
    assert daily["lb_limit"][1] is None
    summary = result.tables["01_longshort_summary"].filter(
        (pl.col("strategy") == "equal_weight") & (pl.col("pnl_type") == "lb")
    )
    assert np.isnan(summary["annual_limit_return"][0])


def test_limit_label_validation_rejects_unknown_encoding():
    date = "20240102"
    factors = pl.DataFrame({"date": [date], "header": ["000001.SZ"], "alpha": [1.0]})
    returns = pl.DataFrame({"date": [date], "header": ["000001.SZ"], "return": [0.01]})
    benchmark = pl.DataFrame({"date": [date], "benchmark_return": [0.0]})
    labels = pl.DataFrame(
        {"date": [date], "header": ["000001.SZ"], "limit_label": [2]}
    )

    with np.testing.assert_raises_regex(ValueError, "limit_label"):
        FactorDataset(factors, returns, benchmark, limit_labels=labels).normalized()


def test_net_report_label_preserves_real_zero_metrics():
    summary = pl.DataFrame(
        {
            "factor": ["alpha"],
            "strategy": ["equal_weight"],
            "pnl_type": ["lb"],
            "annual_return": [0.12],
            "annual_return_net": [0.0],
            "sharpe": [1.2],
            "sharpe_net": [0.0],
            "annual_limit_return": [0.01],
            "avg_turnover": [0.1],
        }
    )

    label = _return_label(summary, "alpha", "equal_weight", "lb", "LBnet")
    assert label.startswith("alpha-LBnet:0.00%|0.00|")
