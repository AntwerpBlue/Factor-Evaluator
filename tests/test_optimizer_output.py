"""Small output-only checks for optimizer constraints and public output."""
import json

import numpy as np
import polars as pl
import pytest

from factor_evaluator import AssetConstraints, IndexEnhancementConfig, IndexEnhancementStrategy
from factor_evaluator._commands.build_index_enhancement_targets import main
from factor_evaluator import RankReturnCalibrator, RiskMatrixProvider, FactorCovarianceMatrix
from factor_evaluator.alpha_calibration import signal_ranks
from factor_evaluator.portfolio import constraints_from_trade_status


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
    retained = strategy().optimize({"A": 1.}, previous, previous,
        asset_constraints=AssetConstraints(allowed_assets=set("ABCD"), frozen=("E",)))
    assert retained["E"] == pytest.approx(.2)


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
    industries.write_text("date,code,industry\n20240101,000001,Bank\n20240101,000002,Tech\n20240101,000003,Tech\n")
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


def barra_provider(assets=("000001", "000002")):
    # Deliberately different current-day exposures: any leakage changes the output.
    exposures, covariance, specific = [], [], []
    for date, scale in [("20261106", 1.), ("20261109", 2.), ("20261110", 100.)]:
        for i, asset in enumerate(assets):
            exposures.append({"date": date, "code": asset, "size": scale * (-1 if i % 2 else 1)})
            specific.append({"date": date, "code": asset, "specific_var": .0001})
        covariance.append({"date": date, "factor_i": "size", "factor_j": "size", "covariance": .00001})
    return RiskMatrixProvider(pl.DataFrame(exposures), pl.DataFrame(covariance), pl.DataFrame(specific))


def normal_status(assets=("000001", "000002")):
    return {asset: {"observed_at": "2026-11-10T09:59:00", "status": "normal"} for asset in assets}


def test_rank_calibration_posterior_forgetting_and_smoothing():
    calibrator = RankReturnCalibrator()
    state = calibrator.initial_state(prior_std=.01, observation_std=.01, halflife_days=1)
    original = state.to_dict()
    assert len(state.means) == 20
    assert signal_ranks({"A": 1, "B": 1}) == {"A": .5, "B": .5}
    assert calibrator.predict({"A": 0, "B": 1}, state) == pytest.approx({"A": -.000114, "B": .000114})
    first = calibrator.update(state, [{"rank": .525, "realized_at": "2026-11-06T15:00", "raw_return": .02}])
    posterior = (state.prior_means[10] + .02) / 2
    assert first.means[10] == pytest.approx(.5 * posterior + .25 * (state.prior_means[9] + state.prior_means[11]))
    assert first.means[9] > state.means[9]
    second = calibrator.update(first, [{"rank": .525, "realized_at": "2026-11-07T15:00", "raw_return": 0.}])
    assert second.counts[10] == pytest.approx(1.5)
    assert second.sums[10] == pytest.approx(.01)
    assert second.state_id == "rank-return:2"
    assert state.to_dict() == original
    prediction = calibrator.predict({"A": .525}, second, return_uncertainty=True)
    assert prediction["uncertainty"]["A"] > 0
    neutral = calibrator.initial_state(basis="barra_neutral", prior_std=.01, observation_std=.01)
    neutral = calibrator.update(neutral, [{"rank": .525, "realized_at": "2026-11-09T15:00", "raw_return": .5, "barra_neutral_return": -.02}])
    assert neutral.means[10] < 0
    with pytest.raises(ValueError, match="realized"):
        calibrator.update(state, [{"rank": .5, "realized_at": "2026-11-10", "available_at": "2026-11-09", "raw_return": .1}])


@pytest.mark.parametrize("time", ["2026-11-10T10:00", "2026-11-10T14:30", "2026-11-10T02:00+00:00"])
def test_intraday_risk_always_uses_prior_barra(time):
    snapshot = barra_provider().snapshot(time, ["000001", "000002"])
    assert snapshot.asof_date == "20261109"
    assert snapshot.covariance.exposures[:, 0] == pytest.approx([2, -2])
    assert barra_provider().snapshot("2026-11-09T10:00", ["000001"]).asof_date == "20261106"
    with pytest.raises(ValueError, match="strictly before"):
        barra_provider().snapshot("2026-11-06T10:00", ["000001"])


def test_trade_status_cutoff_and_unknown_policy():
    records = pl.DataFrame([
        {"asset": "A", "observed_at": "2026-11-10T09:59", "status": "limit_up"},
        {"asset": "A", "observed_at": "2026-11-10T10:00", "status": "normal"},
        {"asset": "B", "observed_at": "2026-11-10T10:01", "status": "normal"},
    ])
    with pytest.raises(ValueError, match="Unknown"):
        constraints_from_trade_status(records, "2026-11-10T10:00", ["A", "B"])
    limits = constraints_from_trade_status(records, "2026-11-10T10:00", ["A", "B"], unknown_policy="freeze")
    assert "A" in limits.no_buy
    assert "B" in limits.frozen


def test_high_level_limits_preserve_excluded_unsellable_holdings():
    assets = ["000001", "000002", "000003", "000004", "000005"]
    current = dict.fromkeys(assets, .2)
    status = normal_status(assets)
    for asset, label in zip(assets, ["limit_up", "limit_down", "suspended", "normal", "normal"]):
        status[asset]["status"] = label
    result = strategy(tracking_error=.5).optimize_signal(
        signal=dict(zip(assets, [10., -10., 20., 5., 0.])), alpha_transform="none",
        decision_time="2026-11-10T10:00", benchmark_weights=current, current_weights=current,
        risk_data=barra_provider(assets), trade_status=status,
        asset_constraints=AssetConstraints(allowed_assets=set(assets) - {"000002", "000003"}),
    )
    assert result.weights["000001"] <= .2000001
    assert result.weights["000002"] == pytest.approx(.2, abs=1e-7)
    assert result.weights["000003"] == pytest.approx(.2, abs=1e-7)
    assert result.weights["000004"] == pytest.approx(.4, abs=1e-7)


def test_sparse_cost_objective_and_named_hard_bounds(monkeypatch):
    def no_dense(*args, **kwargs):
        raise AssertionError("Factor-form optimization must not create dense stock covariance")
    monkeypatch.setattr(FactorCovarianceMatrix, "to_numpy", no_dense)
    previous = {"000001": .5, "000002": .5}
    options = dict(signal={"000001": .01, "000002": 0.}, alpha_transform="none",
        decision_time="2026-11-10T10:00", benchmark_weights=previous, current_weights=previous,
        risk_data=barra_provider(), trade_status=normal_status())
    result = strategy(tracking_error=.05, buy_cost_bps=10, sell_cost_bps=20,
        exposure_bounds={"size": (-.1, .1)}).optimize_signal(**options)
    assert result.weights == pytest.approx({"000001": .525, "000002": .475}, abs=1e-6)
    report = result.diagnostics
    assert report["expected_cost"] == pytest.approx(.025 * .003, abs=1e-9)
    assert report["active_exposures"]["size"] == pytest.approx(.1, abs=1e-6)
    assert report["factor_risk"] + report["specific_risk"] == pytest.approx(report["predicted_te"]**2)
    assert report["barra_asof_date"] == "20261109"
    assert report["solver"] == "clarabel"
    assert min(report["constraint_slacks"].values()) >= -1e-7
    # Moving a unit earns 1bp but costs 30bp: the optimum keeps actual holdings.
    options["signal"] = {"000001": .0001, "000002": 0.}
    unchanged = strategy(tracking_error=.05, buy_cost_bps=10, sell_cost_bps=20).optimize_signal(**options)
    assert unchanged.weights == pytest.approx(previous, abs=1e-6)


def test_sparse_te_budget_binds():
    current = {"A": .5, "B": .5}
    covariance = FactorCovarianceMatrix(np.zeros((2, 1)), np.zeros((1, 1)), np.full(2, .0001))
    result = strategy(objective_mode="te_cost", tracking_error=.05).optimize_result(
        {"A": .01, "B": 0.}, current, current, covariance=covariance)
    expected = .5 + .05 / np.sqrt(2 * .0001 * 252)
    assert result.weights["A"] == pytest.approx(expected, abs=1e-6)
    assert result.diagnostics["predicted_te"] == pytest.approx(.05, abs=1e-7)


def test_calibrated_entrypoint_reads_state_and_rejects_future_inputs():
    calibrator = RankReturnCalibrator()
    state = calibrator.initial_state()
    before = state.to_dict()
    current = {"000001": .5, "000002": .5}
    options = dict(signal={"000001": 5., "000002": 1.}, decision_time="2026-11-10T10:00",
        benchmark_weights=current, current_weights=current, risk_data=barra_provider(), trade_status=normal_status())
    optimizer = strategy(tracking_error=.05)
    result = optimizer.optimize_signal(**options, calibration_state=state)
    assert result.diagnostics["predicted_returns"] == calibrator.predict(signal_ranks(options["signal"]), state)
    assert result.diagnostics["alpha_calibration_state_id"] == state.state_id
    assert state.to_dict() == before
    future = calibrator.update(state, [{"rank": .9, "realized_at": "2026-11-10T15:00", "raw_return": .01}])
    with pytest.raises(ValueError, match="not available"):
        optimizer.optimize_signal(**options, calibration_state=future)
    options["signal"] = {"observed_at": "2026-11-10T10:01", "values": options["signal"]}
    with pytest.raises(ValueError, match="not available"):
        optimizer.optimize_signal(**options, calibration_state=state)


def test_single_decision_cli_uses_current_holdings_and_prior_snapshot(tmp_path):
    def save(name, text):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)
    signal = save("signal.csv", "code,signal_time,alpha\n000001,2026-11-10T09:58,1\n000002,2026-11-10T09:58,0\n")
    # Old positions must not be carried forward into a newer complete snapshot.
    positions = save("positions.csv", "date,code,weight\n20261109,000001,0.8\n20261110,000001,0.6\n20261110,000002,0.4\n")
    benchmark = save("benchmark.csv", "code,weight\n000001,0.5\n000002,0.5\n")
    status = save("status.csv", "code,observed_at,status\n000001,2026-11-10T09:59,limit_up\n000002,2026-11-10T09:59,normal\n")
    x = save("x.csv", "date,code,size\n20261109,000001,1\n20261109,000002,-1\n")
    omega = save("omega.csv", "date,factor_i,factor_j,covariance\n20261109,size,size,0.00001\n")
    d = save("d.csv", "date,code,specific_var\n20261109,000001,0.0001\n20261109,000002,0.0001\n")
    output = tmp_path / "targets.parquet"
    main(["--decision-time", "2026-11-10T10:00", "--factor-values", signal,
        "--previous-weights", positions, "--benchmark-weights", benchmark, "--trade-status", status,
        "--risk-exposures", x, "--factor-covariance", omega, "--specific-risk", d,
        "--industry-exposures", str(tmp_path / "unused"), "--universe", str(tmp_path / "unused"),
        "--alpha-transform", "none", "--tracking-error", ".2", "--max-weight", "1",
        "--max-active-weight", "1", "--output", str(output)])
    assert pl.read_parquet(output)["weight"].to_list() == pytest.approx([.6, .4], abs=1e-6)
    report = json.loads(output.with_suffix(".diagnostics.json").read_text(encoding="utf-8"))
    assert report["barra_asof_date"] == "20261109"
    assert report["objective_mode"] == "te_cost"
