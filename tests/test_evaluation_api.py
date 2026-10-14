from __future__ import annotations

import json

import numpy as np
import polars as pl

from factor_evaluator import FactorEvaluationConfig, evaluate_factors


def evaluation_inputs(n_dates: int = 8, n_assets: int = 6):
    dates = [
        str(value).replace("-", "")
        for value in np.busday_offset("2025-01-02", np.arange(n_dates))
    ]
    factors: list[dict[str, object]] = []
    returns: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for asset_index in range(n_assets):
            signal = float(asset_index + date_index)
            factors.append(
                {
                    "date": date,
                    "code": f"{asset_index:06d}.SZ",
                    "alpha": signal,
                    "inverse": -signal,
                }
            )
            returns.append(
                {
                    "date": date,
                    "code": f"{asset_index:06d}.SZ",
                    "return": signal / 100.0,
                }
            )
    return pl.DataFrame(factors), pl.DataFrame(returns)


def test_evaluate_factors_returns_structured_real_metrics():
    factors, returns = evaluation_inputs()
    result = evaluate_factors(
        factors,
        returns,
        config=FactorEvaluationConfig(min_assets=4, max_stability_lag=2),
    )

    assert result.schema_version == "factor-evaluation/v1"
    assert result.intrinsic["alpha"]["pearson_ic"] == 1.0
    assert result.intrinsic["alpha"]["rank_ic"] == 1.0
    assert result.intrinsic["inverse"]["rank_ic"] == -1.0
    assert result.data_quality["alpha"]["missing_ratio"] == 0.0
    assert result.stability["alpha"]["factor_autocorrelation"][0]["lag"] == 1
    assert result.correlations[0]["mean_cross_sectional_pearson"] == -1.0
    assert result.daily_metrics.height == 16
    json.dumps(result.to_dict(), allow_nan=False)


def test_evaluate_factors_accepts_paths_and_reports_missing_values(tmp_path):
    factors, returns = evaluation_inputs()
    factors = factors.with_columns(
        pl.when((pl.col("date") == factors["date"][0]) & (pl.col("code") == "000000.SZ"))
        .then(None)
        .otherwise(pl.col("alpha"))
        .alias("alpha")
    ).select("date", "code", "alpha")
    factor_path = tmp_path / "factors.parquet"
    return_path = tmp_path / "returns.csv"
    factors.write_parquet(factor_path)
    returns.write_csv(return_path)

    result = evaluate_factors(
        factor_path,
        return_path,
        config=FactorEvaluationConfig(min_assets=4, max_stability_lag=1),
    )

    quality = result.data_quality["alpha"]
    assert quality["expected_observations"] == 48
    assert quality["valid_observations"] == 47
    assert quality["missing_observations"] == 1
    assert quality["warnings"]
    output = result.write_json(tmp_path / "evaluation.json")
    assert json.loads(output.read_text(encoding="utf-8"))["factors"] == ["alpha"]


def test_evaluate_factors_can_select_one_column():
    factors, returns = evaluation_inputs()
    result = evaluate_factors(
        factors,
        returns,
        factor_names=["alpha"],
        config=FactorEvaluationConfig(min_assets=4, max_stability_lag=1),
    )

    assert result.factors == ("alpha",)
    assert result.correlations == []
    assert result.backtest_result.metadata["factor_columns"] == ["alpha"]


def test_evaluate_factors_forwards_limit_labels_and_cost_config():
    factors, returns = evaluation_inputs()
    labels = returns.select("date", "code").with_columns(
        pl.when(pl.col("code") == "000005.SZ")
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("limit_label")
    )

    result = evaluate_factors(
        factors.select("date", "code", "alpha"),
        returns,
        limit_labels=labels,
        config=FactorEvaluationConfig(
            min_assets=4,
            max_stability_lag=1,
            cost_bps=8.0,
            topk=0.5,
            power=3.0,
        ),
    )

    assert result.metadata["limit_labels_supplied"] is True
    assert result.metadata["limit_label_coverage"] == 1.0
    assert np.isclose(result.metadata["limit_label_hit_rate"], 1.0 / 6.0)
    assert result.metadata["cost_bps"] == 8.0
    assert result.metadata["topk"] == 0.5
    assert result.metadata["power"] == 3.0
    assert result.backtest_result.daily["alpha"]["topk_power_lb_limit"].is_null().sum() == 0
