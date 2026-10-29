"""Small output-only checks for optimizer constraints and public output."""
import json

import numpy as np
import polars as pl
import pytest

from factor_evaluator import AssetConstraints, IndexEnhancementConfig, IndexEnhancementStrategy
from factor_evaluator._commands.build_index_enhancement_targets import main


def strategy(**options):
    return IndexEnhancementStrategy(IndexEnhancementConfig(**{
        "risk_aversion": 0.0, "turnover_penalty": 0.0,
        "max_weight": 1.0, "max_active_weight": 1.0, **options,
    }))


def test_annual_tracking_error_limits_output():
    result = strategy(tracking_error=0.05).optimize_result(
        {"A": 0.1, "B": 0.0}, {"A": 0.5, "B": 0.5},
        covariance=np.eye(2) * 0.0001,
    )
    x = 0.05 / np.sqrt(2 * 0.0001 * 252)
    assert result.weights["A"] == pytest.approx(0.5 + x, abs=1e-6)
    assert result.diagnostics["tracking_error_annual"] == pytest.approx(0.05, abs=1e-7)


def test_turnover_cap_preserves_changed_weight_bounds():
    result = strategy(max_weight=0.6, max_turnover=0.25).optimize_result(
        {"A": 0.1, "B": 0.0}, {"A": 0.5, "B": 0.5}, {"A": 0.8, "B": 0.2},
    )
    assert result.weights == pytest.approx({"A": 0.6, "B": 0.4})
    assert result.diagnostics["turnover"] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="optimization failed"):
        strategy(max_weight=0.6, max_turnover=0.1).optimize(
            {"A": 0.1, "B": 0.0}, {"A": 0.5, "B": 0.5}, {"A": 0.8, "B": 0.2},
        )


def test_exposure_and_trade_bounds_hold_together():
    result = strategy(active_exposure_limit=0.1, max_trade_weight=0.03).optimize_result(
        {"A": 0.1, "B": 0.0}, {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5},
        exposure_matrix=np.array([[1.0], [-1.0]]),
    )
    assert result.weights == pytest.approx({"A": 0.53, "B": 0.47})
    assert result.diagnostics["active_exposures"][0] == pytest.approx(0.06)


def test_calibrated_alpha_and_initial_holdings_in_batch():
    optimizer = strategy(alpha_return_scale=0.002, turnover_cost_alpha_fraction=0.9)
    signals = pl.DataFrame({"date": ["20240102"] * 2, "asset": ["000001", "000002"], "alpha": [1.0, -1.0]})
    benchmark = signals.select("date", "asset").with_columns(pl.lit(0.5).alias("weight"))
    target = optimizer.build_targets(signals, benchmark, initial_weights={"000001": 0.6, "000002": 0.4})
    assert target["weight"].to_list() == pytest.approx([1.0, 0.0])
    report = optimizer.diagnostics[0]
    assert report["expected_alpha"] == pytest.approx(0.002)
    assert report["turnover"] == pytest.approx(0.4)
    assert report["constraint_slacks"]["cost_alpha"] >= 0


def test_cli_writes_weights_and_diagnostics(tmp_path):
    signal = tmp_path / "signal.csv"
    benchmark = tmp_path / "benchmark.csv"
    previous = tmp_path / "previous.csv"
    signal.write_text("date,code,alpha\n20240102,000001,0.01\n20240102,000002,0.0\n")
    benchmark.write_text("date,code,weight\n20240102,000001,0.5\n20240102,000002,0.5\n")
    previous.write_text("date,code,weight\n20240102,000001,0.6\n20240102,000002,0.4\n")
    output = tmp_path / "weights.parquet"
    main([
        "--factor-values", str(signal), "--benchmark-weights", str(benchmark),
        "--previous-weights", str(previous), "--output", str(output),
        "--universe", str(tmp_path / "unused"), "--tradability", str(tmp_path / "unused"),
        "--alpha-transform", "none", "--max-weight", "1", "--max-active-weight", "1",
        "--risk-aversion", "0", "--turnover-penalty", "0", "--max-daily-turnover", "0.1",
    ])
    assert pl.read_parquet(output)["weight"].to_list() == pytest.approx([0.7, 0.3])
    report = json.loads(output.with_suffix(".diagnostics.json").read_text())[0]
    assert report["alpha_units"] == "return"
    assert report["constraints_satisfied"]
    assert report["turnover"] == pytest.approx(0.1)


@pytest.mark.parametrize("risk_aversion", [0.0, 0.02])
def test_named_neutrality_and_industry_band(risk_aversion):
    result = strategy(risk_aversion=risk_aversion, exposure_bounds={
        "size": (0.0, 0.0), "industry:Bank": (-0.1, 0.1),
    }).optimize_result(
        {"A": 0.04, "B": 0.03, "C": 0.02, "D": 0.0},
        dict.fromkeys("ABCD", 0.25),
        exposure_matrix=np.array([[1., 1.], [-1., 1.], [1., 0.], [-1., 0.]]),
        exposure_names=["size", "industry:Bank"],
    )
    exposure = result.diagnostics["named_active_exposures"]
    assert exposure["size"] == pytest.approx(0.0, abs=1e-7)
    assert -0.1000001 <= exposure["industry:Bank"] <= 0.1000001
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_stock_bounds_and_trade_permissions_are_intersected():
    previous = dict.fromkeys("ABCDE", 0.2)
    restrictions = AssetConstraints(
        weight_bounds={"D": (0.1, 0.25)}, allowed_assets=set("ABCD"),
        frozen=("A",), no_buy=("B",), no_sell=("C",),
    )
    result = strategy().optimize_result(
        {"A": 8., "B": 10., "C": 1., "D": 9., "E": 100.}, previous, previous,
        asset_constraints=restrictions,
    )
    assert result.weights == pytest.approx({"A": .2, "B": .2, "C": .35, "D": .25, "E": 0.})
    with pytest.raises(ValueError, match="conflicting assets"):
        strategy().optimize({"A": 1.}, previous, previous,
            asset_constraints=AssetConstraints(allowed_assets=set("ABCD"), frozen=("E",)))


def test_dated_freeze_uses_drifted_holdings():
    signals = pl.DataFrame({
        "date": ["20240102", "20240102", "20240103", "20240103"],
        "asset": ["000001", "000002"] * 2, "alpha": [1., 0., 1., 0.],
    })
    benchmark = signals.select("date", "asset").with_columns(pl.lit(0.5).alias("weight"))
    returns = signals.select("date", "asset").with_columns(pl.Series("return", [.1, 0., 0., 0.]))
    output = strategy().build_targets(
        signals, benchmark, asset_returns=returns,
        initial_weights={"000001": .5, "000002": .5},
        asset_constraints_by_date={
            "20240102": AssetConstraints(frozen=("000001", "000002")),
            "20240103": AssetConstraints(frozen=("000001",)),
        },
    )
    day = output.filter(pl.col("date") == "20240103")
    assert day["weight"].to_list() == pytest.approx([.55 / 1.05, .5 / 1.05])


def test_cli_industry_constraints_and_strict_universe(tmp_path):
    signal = tmp_path / "signal.csv"
    benchmark = tmp_path / "benchmark.csv"
    universe = tmp_path / "universe.csv"
    assets = tmp_path / "assets.csv"
    industries = tmp_path / "industry.csv"
    bands = tmp_path / "bands.toml"
    output = tmp_path / "weights.parquet"
    signal.write_text("date,code,alpha\n20240102,000001,1\n20240102,000002,2\n20240102,000003,100\n")
    benchmark.write_text("date,code,weight\n20240102,000001,0.4\n20240102,000002,0.3\n20240102,000003,0.3\n")
    universe.write_text("date,code,eligible\n20240102,000001,true\n20240102,000002,true\n20240102,000003,false\n")
    assets.write_text("date,code,frozen,min_weight\n20240102,000001,true,0\n20240102,000002,false,0.6\n")
    industries.write_text("date,code,industry\n20240102,000001,Bank\n20240102,000002,Tech\n20240102,000003,Tech\n")
    bands.write_text('[exposure_bounds]\n"industry:Bank" = [0.0, 0.0]\n')
    main([
        "--factor-values", str(signal), "--benchmark-weights", str(benchmark),
        "--previous-weights", str(benchmark), "--output", str(output),
        "--universe", str(universe), "--tradability", str(tmp_path / "unused"),
        "--asset-constraints", str(assets), "--industry-exposures", str(industries),
        "--exposure-bounds", str(bands), "--max-weight", "1", "--max-active-weight", "1",
        "--risk-aversion", "0", "--turnover-penalty", "0",
    ])
    assert pl.read_parquet(output)["weight"].to_list() == pytest.approx([.4, .6, 0.])
    report = json.loads(output.with_suffix(".diagnostics.json").read_text())[0]
    assert report["named_active_exposures"]["industry:Bank"] == pytest.approx(0.0)
