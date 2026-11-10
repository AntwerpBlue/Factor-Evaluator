"""Build index-enhancement target weights from factor signals.

The optimizer is factor_evaluator.portfolio.IndexEnhancementStrategy. Inputs are
long panel parquet/csv files using date + code/header plus one value column.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tomllib
from pathlib import Path

import polars as pl


from factor_evaluator import AssetConstraints, CalibrationState, ExposureMatrixProvider, IndexEnhancementConfig, IndexEnhancementStrategy, RiskMatrixProvider
from factor_evaluator.data import normalize_panel
from factor_evaluator.risk import decision_timestamp


LOGGER = logging.getLogger("build_index_enhancement_targets")

BENCHMARK_WEIGHT_FILES = {
    "300": "benchmark_300_weights.parquet",
    "500": "benchmark_500_weights.parquet",
    "1000": "benchmark_1000_weights.parquet",
    "2000": "benchmark_2000_weights.parquet",
    "Zhishu300": "benchmark_300_weights.parquet",
    "Zhishu500": "benchmark_500_weights.parquet",
    "Zhishu1000": "benchmark_1000_weights.parquet",
    "Zhishu2000": "benchmark_2000_weights.parquet",
    "ZeroIndex": None,
    "zeroindex": None,
}


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path)))


def load_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream))
        return pl.read_csv(path, schema_overrides={name: pl.String for name in columns if name in {"code", "asset", "header"}})
    raise ValueError(f"Unsupported file type: {path}")


def value_column(frame: pl.DataFrame, requested: str | None, label: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"{label} column not found: {requested}")
        return requested
    values = [c for c in frame.columns if c not in {"date", "asset"}]
    if len(values) != 1:
        raise ValueError(f"{label} must have exactly one value column or pass an explicit option")
    return values[0]


def date_filter(frame: pl.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    if start:
        frame = frame.filter(pl.col("date").cast(pl.String) >= start.replace("-", ""))
    if end:
        frame = frame.filter(pl.col("date").cast(pl.String) <= end.replace("-", ""))
    return frame


def constraint_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Format constraint keys without scanning for input consistency."""
    code = next(name for name in ("asset", "code", "header") if name in frame.columns)
    return frame.rename({code: "asset"}).with_columns(
        pl.col("date").cast(pl.String), pl.col("asset").cast(pl.String).str.slice(0, 6)
    )


def asset_constraint_provider(
    benchmark: pl.DataFrame,
    universe: pl.DataFrame | None,
    panels: list[pl.DataFrame],
    allow_off_benchmark: bool,
):
    members = {
        day["date"][0]: set(day.filter(pl.col("weight") > 0)["asset"].to_list())
        for day in benchmark.partition_by("date")
    }
    allowed = None if universe is None else {
        day["date"][0]: set(day.filter(pl.col("eligible"))["asset"].to_list())
        for day in universe.partition_by("date")
    }
    records: dict[str, list[dict]] = {}
    for panel in panels:
        for row in panel.iter_rows(named=True):
            records.setdefault(row["date"], []).append(row)

    def for_date(date: str) -> AssetConstraints:
        eligible = None if allowed is None else set(allowed[date])
        if not allow_off_benchmark:
            eligible = set(members[date]) if eligible is None else eligible & members[date]
        bounds, frozen, no_buy, no_sell = {}, set(), set(), set()
        for row in records.get(date, []):
            asset = row["asset"]
            lo, hi = bounds.get(asset, (-float("inf"), float("inf")))
            if row.get("min_weight") is not None:
                lo = max(lo, row["min_weight"])
            if row.get("max_weight") is not None:
                hi = min(hi, row["max_weight"])
            if row.get("fixed_weight") is not None:
                lo, hi = max(lo, row["fixed_weight"]), min(hi, row["fixed_weight"])
            if lo != -float("inf") or hi != float("inf"):
                bounds[asset] = (lo, hi)
            if row.get("frozen") or row.get("tradable") == False:
                frozen.add(asset)
            if row.get("can_buy") == False:
                no_buy.add(asset)
            if row.get("can_sell") == False:
                no_sell.add(asset)
        return AssetConstraints(bounds, eligible, tuple(frozen), tuple(no_buy), tuple(no_sell))

    return for_date


def make_exposure_provider(args, exposure_bounds):
    """Legacy exposure-only path, using prior snapshots without restandardization."""
    needs_style = args.style_exposure_penalty > 0 or args.style_exposure_limit is not None or any(not name.startswith("industry:") for name in exposure_bounds)
    providers = []
    if needs_style:
        styles = load_frame(expand(args.risk_exposures))
        providers.append(ExposureMatrixProvider(styles))
    industry_names = [name for name in exposure_bounds if name.startswith("industry:")]
    if industry_names:
        industry = constraint_panel(load_frame(expand(args.industry_exposures)))
        if "industry" in industry.columns:
            industry = industry.select("date", "asset", *[
                (pl.col("industry").cast(pl.String) == name.removeprefix("industry:")).cast(pl.Float64).alias(name)
                for name in industry_names
            ])
        else:
            industry = industry.select("date", "asset", *[
                pl.col(name if name in industry.columns else name.removeprefix("industry:")).alias(name)
                for name in industry_names
            ])
        providers.append(ExposureMatrixProvider(industry, exposure_columns=industry_names, standardize=False))
    names = [name for provider in providers for name in provider.exposure_columns]

    def for_date(date, assets):
        import numpy as np
        matrices = [provider(date, assets) for provider in providers]
        if any(matrix is None for matrix in matrices):
            raise ValueError(f"Required constraint exposures unavailable on {date}")
        return np.column_stack(matrices)

    return for_date, names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized index-enhancement target weights.")
    parser.add_argument("--factor-values", required=True, help="External factor parquet/csv")
    parser.add_argument("--factor-column", help="Factor column to optimize on")
    parser.add_argument("--decision-time", help="Single decision timestamp, exchange-local time or ISO offset; enables optimize_signal.")
    parser.add_argument("--signal-time", help="Availability timestamp for a signal file without a signal_time column.")
    parser.add_argument("--trade-status", help="Pre-decision status table: observed_at, code, status or trading permissions.")
    parser.add_argument("--unknown-status-policy", choices=("reject", "freeze"), default="reject")
    parser.add_argument("--calibration-state", help="Externally computed CalibrationState JSON; never updated by this command.")
    parser.add_argument("--objective-mode", choices=("legacy", "te_cost"), help="Single-decision default: te_cost. Daily legacy workflow default: legacy.")
    parser.add_argument("--buy-cost-bps", type=float, default=8.0)
    parser.add_argument("--sell-cost-bps", type=float, default=8.0)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--benchmark", default="500", choices=sorted(BENCHMARK_WEIGHT_FILES))
    parser.add_argument("--data-dir", default="~/data")
    parser.add_argument("--benchmark-weights", help="Override benchmark weights file")
    parser.add_argument("--universe", default="~/data/universe.parquet")
    parser.add_argument("--tradability", default="~/data/tradability.parquet")
    parser.add_argument("--asset-constraints", help="Dated asset table: min_weight/max_weight/fixed_weight/frozen/can_buy/can_sell.")
    parser.add_argument("--exposure-bounds", help="TOML [exposure_bounds] table: named active exposure [lower, upper]; industries use industry:NAME.")
    parser.add_argument("--asset-returns", help="Asset return panel used to drift previous target weights before optimization.")
    parser.add_argument("--previous-weights", help="Current pre-trade holdings (date, code, weight); uses the last snapshot as initial holdings.")
    parser.add_argument("--output", default="~/data/target_weights.parquet")
    parser.add_argument("--use-risk-model", action="store_true", help="Use prior-day X F X' + D snapshots in legacy daily mode; always used for a single decision.")
    parser.add_argument("--risk-exposures", default="~/data/style_exposures.parquet")
    parser.add_argument("--industry-exposures", default="~/data/industry_exposures.parquet")
    parser.add_argument("--factor-covariance", default="~/data/factor_covariance.parquet")
    parser.add_argument("--specific-risk", default="~/data/specific_risk.parquet")
    parser.add_argument("--specific-var-floor", type=float, default=1e-8)
    parser.add_argument("--style-exposure-penalty", type=float, default=0.0, help="Soft penalty for active style exposure squared norm.")
    parser.add_argument("--style-exposure-limit", type=float, help="Hard active style exposure band, matching style_bands semantics.")
    parser.add_argument("--alpha-transform", choices=("none", "zscore", "rank_zscore", "calibrated_rank"), help="Single-decision default calibrated_rank; legacy daily default zscore.")
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--alpha-return-scale", type=float, help="Expected return per transformed score unit, applied after clipping; same horizon as covariance.")
    parser.add_argument("--alpha-clip", type=float, default=5.0)
    parser.add_argument("--allow-off-benchmark", action="store_true", help="Allow assets outside the benchmark constituent file.")
    parser.add_argument("--max-weight", type=float, default=0.03)
    parser.add_argument("--max-active-weight", type=float, default=0.01)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
    parser.add_argument("--turnover-cost-bps", type=float, default=8.0)
    parser.add_argument("--turnover-cost-alpha-fraction", type=float, default=0.0, help="Require cost <= fraction * expected return improvement; needs return-scaled alpha. Default disabled.")
    parser.add_argument("--max-daily-turnover", type=float, help="Hard cap for daily one-way turnover, e.g. 0.05 means 5%%.")
    parser.add_argument("--tracking-error", type=float, help="Annualized tracking error limit, e.g. 0.05 for 5%%.")
    parser.add_argument("--risk-annualization", type=float, default=252.0, help="Covariance periods per year; 252 for daily, 1 for already annualized covariance.")
    parser.add_argument("--max-trade-weight", type=float, help="Maximum absolute change per asset from supplied/current holdings.")
    parser.add_argument("--long-short", action="store_true", help="Allow short positions")
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser.parse_args(argv)


def load_barra(args):
    # Keep history before the requested interval; asof selection happens at decision time.
    exposures = constraint_panel(load_frame(expand(args.risk_exposures)))
    industry_path = expand(args.industry_exposures)
    if industry_path.exists():
        industry = constraint_panel(load_frame(industry_path))
        if "industry" in industry.columns:
            labels = industry["industry"].drop_nulls().unique().to_list()
            industry = industry.select("date", "asset", *[
                (pl.col("industry") == label).cast(pl.Float64).alias(f"industry:{label}") for label in labels
            ])
        extras = [c for c in industry.columns if c not in exposures.columns]
        if extras:
            exposures = exposures.join(industry.select("date", "asset", *extras), on=["date", "asset"], how="left")
    return RiskMatrixProvider(exposures, load_frame(expand(args.factor_covariance)),
        load_frame(expand(args.specific_risk)), specific_var_floor=args.specific_var_floor)


def snapshot_rows(path, now):
    """Adapt a current snapshot or choose latest dated observations available by now."""
    frame = load_frame(expand(path))
    code = next(name for name in ("asset", "code", "header") if name in frame.columns)
    frame = frame.rename({code: "asset"}).with_columns(pl.col("asset").cast(pl.String).str.slice(0, 6))
    rows = frame.to_dicts()
    time_column = "observed_at" if "observed_at" in frame.columns else "date" if "date" in frame.columns else None
    if time_column:
        available = [(decision_timestamp(str(row[time_column])), row) for row in rows]
        available = [(timestamp, row) for timestamp, row in available if timestamp <= now]
        snapshot_time = max((timestamp for timestamp, _ in available), default=None)
        rows = [row for timestamp, row in available if timestamp == snapshot_time]
    return rows


def run_signal_decision(args):
    now = decision_timestamp(args.decision_time)
    signal_frame = load_frame(expand(args.factor_values))
    code = next(name for name in ("asset", "code", "header") if name in signal_frame.columns)
    factor = args.factor_column or next(c for c in signal_frame.columns if c not in {code, "date", "signal_time", "observed_at"})
    signals = {}
    for row in signal_frame.iter_rows(named=True):
        timestamp = row.get("signal_time", args.signal_time)
        if timestamp is None:
            raise ValueError("Single-decision signals require signal_time or --signal-time")
        timestamp = decision_timestamp(timestamp)
        asset = str(row[code])[:6]
        if timestamp <= now and (asset not in signals or timestamp > signals[asset][0]):
            signals[asset] = (timestamp, float(row[factor]))
    if not signals:
        raise ValueError("No signal available at decision_time")
    if not args.previous_weights:
        raise ValueError("Single-decision optimization requires --previous-weights (actual positions)")
    current = {row["asset"]: float(row["weight"]) for row in snapshot_rows(args.previous_weights, now)}
    zero_benchmark = BENCHMARK_WEIGHT_FILES[args.benchmark] is None
    benchmark_path = args.benchmark_weights or (expand(args.data_dir) / BENCHMARK_WEIGHT_FILES[args.benchmark] if not zero_benchmark else None)
    benchmark = {row["asset"]: float(row["weight"]) for row in snapshot_rows(benchmark_path, now)} if benchmark_path else dict.fromkeys(signals, 0.0)
    date = now.strftime("%Y%m%d")
    benchmark_panel = pl.DataFrame([{"date": date, "asset": a, "weight": w} for a, w in benchmark.items()])
    universe = None
    if expand(args.universe).exists():
        universe = pl.DataFrame([{**row, "date": date} for row in snapshot_rows(args.universe, now)])
    panels = []
    if args.asset_constraints:
        panels.append(pl.DataFrame([{**row, "date": date} for row in snapshot_rows(args.asset_constraints, now)]))
    restrictions = asset_constraint_provider(benchmark_panel, universe, panels, args.allow_off_benchmark or zero_benchmark)(date)
    status_path = args.trade_status or args.tradability
    if expand(status_path).exists():
        trade_status = load_frame(expand(status_path))
        status_code = next(name for name in ("asset", "code", "header") if name in trade_status.columns)
        trade_status = trade_status.rename({status_code: "asset"}).with_columns(pl.col("asset").cast(pl.String).str.slice(0, 6))
    else:
        trade_status = {}
    state = CalibrationState.from_dict(json.loads(expand(args.calibration_state).read_text(encoding="utf-8"))) if args.calibration_state else None
    bounds = {}
    if args.exposure_bounds:
        with expand(args.exposure_bounds).open("rb") as stream:
            bounds = tomllib.load(stream)["exposure_bounds"]
    config = IndexEnhancementConfig(
        objective_mode=args.objective_mode or "te_cost", tracking_error=args.tracking_error,
        risk_annualization=args.risk_annualization, risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty, buy_cost_bps=args.buy_cost_bps, sell_cost_bps=args.sell_cost_bps,
        max_weight=args.max_weight, max_active_weight=args.max_active_weight, max_trade_weight=args.max_trade_weight,
        max_turnover=args.max_daily_turnover, exposure_bounds=bounds, active_exposure_limit=args.style_exposure_limit,
        active_exposure_penalty=args.style_exposure_penalty, zero_benchmark=zero_benchmark, long_only=not args.long_short,
        alpha_scale=args.alpha_scale, alpha_clip=args.alpha_clip, alpha_return_scale=args.alpha_return_scale,
        turnover_cost_bps=args.turnover_cost_bps,
        turnover_cost_alpha_fraction=args.turnover_cost_alpha_fraction if args.turnover_cost_alpha_fraction > 0 else None,
        solver_time_limit=args.solver_time_limit,
    )
    result = IndexEnhancementStrategy(config).optimize_signal(
        signal={"observed_at": max(time for time, _ in signals.values()).isoformat(), "values": {a: value for a, (_, value) in signals.items()}},
        decision_time=now, benchmark_weights=benchmark, current_weights=current, risk_data=load_barra(args),
        trade_status=trade_status, calibration_state=state, asset_constraints=restrictions,
        alpha_transform=args.alpha_transform or "calibrated_rank", unknown_status_policy=args.unknown_status_policy,
    )
    output = expand(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"date": date, "decision_time": now.isoformat(), "code": a, "weight": w} for a, w in result.weights.items()]).write_parquet(output)
    output.with_suffix(".diagnostics.json").write_text(json.dumps(result.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote target weights for %s to %s", now.isoformat(), output)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    if args.decision_time:
        run_signal_decision(args)
        return
    args.alpha_transform = args.alpha_transform or "zscore"

    data_dir = expand(args.data_dir)
    zero_benchmark = BENCHMARK_WEIGHT_FILES[args.benchmark] is None
    benchmark_path = (
        expand(args.benchmark_weights)
        if args.benchmark_weights
        else None if zero_benchmark else data_dir / BENCHMARK_WEIGHT_FILES[args.benchmark]
    )

    signals = normalize_panel(load_frame(expand(args.factor_values)))
    factor_col = value_column(signals, args.factor_column, "factor_values")
    signals = signals.select("date", "asset", pl.col(factor_col).cast(pl.Float64).alias("alpha"))
    if args.start:
        signals = signals.filter(pl.col("date") >= args.start.replace("-", ""))
    if args.end:
        signals = signals.filter(pl.col("date") <= args.end.replace("-", ""))

    universe_path = expand(args.universe)
    universe = constraint_panel(load_frame(universe_path)) if universe_path.exists() else None
    constraint_panels = []
    tradability_path = expand(args.tradability)
    if tradability_path.exists():
        constraint_panels.append(constraint_panel(load_frame(tradability_path)))
    if args.asset_constraints:
        constraint_panels.append(constraint_panel(load_frame(expand(args.asset_constraints))))

    asset_returns = date_filter(normalize_panel(load_frame(expand(args.asset_returns))), args.start, args.end) if args.asset_returns else None

    if zero_benchmark:
        benchmark_weights = signals.select("date", "asset").unique().with_columns(pl.lit(0.0).alias("weight"))
    else:
        assert benchmark_path is not None
        benchmark_weights = normalize_panel(load_frame(benchmark_path))
        weight_col = value_column(benchmark_weights, "weight" if "weight" in benchmark_weights.columns else None, "benchmark_weights")
        benchmark_weights = benchmark_weights.select("date", "asset", pl.col(weight_col).cast(pl.Float64).alias("weight"))
    constraints_by_date = asset_constraint_provider(
        benchmark_weights, universe, constraint_panels, args.allow_off_benchmark or zero_benchmark
    )

    exposure_provider = None
    exposure_names = None
    exposure_bounds = {}
    if args.exposure_bounds:
        with expand(args.exposure_bounds).open("rb") as stream:
            exposure_bounds = tomllib.load(stream)["exposure_bounds"]
    if not args.use_risk_model and (args.style_exposure_penalty > 0 or args.style_exposure_limit is not None or exposure_bounds):
        exposure_provider, exposure_names = make_exposure_provider(args, exposure_bounds)

    covariance_provider = None
    if args.use_risk_model:
        covariance_provider = load_barra(args)
        LOGGER.info("Using prior-day Barra snapshots for risk and exposure constraints")

    config = IndexEnhancementConfig(
        objective_mode=args.objective_mode or "legacy",
        buy_cost_bps=args.buy_cost_bps, sell_cost_bps=args.sell_cost_bps,
        solver_time_limit=args.solver_time_limit,
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        max_weight=args.max_weight,
        max_active_weight=args.max_active_weight,
        tracking_error=args.tracking_error,
        risk_annualization=args.risk_annualization,
        require_covariance=args.use_risk_model,
        long_only=not args.long_short,
        active_exposure_penalty=args.style_exposure_penalty,
        active_exposure_limit=args.style_exposure_limit,
        exposure_bounds=exposure_bounds,
        turnover_cost_bps=args.turnover_cost_bps,
        turnover_cost_alpha_fraction=(
            args.turnover_cost_alpha_fraction
            if args.turnover_cost_alpha_fraction is None or args.turnover_cost_alpha_fraction > 0
            else None
        ),
        max_turnover=args.max_daily_turnover,
        max_trade_weight=args.max_trade_weight,
        zero_benchmark=zero_benchmark,
        alpha_transform=args.alpha_transform,
        alpha_scale=args.alpha_scale,
        alpha_return_scale=args.alpha_return_scale,
        alpha_clip=args.alpha_clip,
    )
    initial_weights = None
    if args.previous_weights:
        snapshot = normalize_panel(load_frame(expand(args.previous_weights)))
        snapshot = snapshot.filter(pl.col("date") == snapshot["date"].max())
        initial_weights = dict(snapshot.select("asset", "weight").iter_rows())
    strategy = IndexEnhancementStrategy(config)
    calibration_states = None
    if args.calibration_state:
        state = CalibrationState.from_dict(json.loads(expand(args.calibration_state).read_text(encoding="utf-8")))
        calibration_states = dict.fromkeys(signals["date"].unique().to_list(), state)
    targets = strategy.build_targets(
        signals,
        benchmark_weights,
        covariance_by_date=covariance_provider,
        exposure_by_date=exposure_provider,
        asset_returns=asset_returns,
        initial_weights=initial_weights,
        asset_constraints_by_date=constraints_by_date,
        exposure_names=exposure_names,
        calibration_states_by_date=calibration_states,
    )
    output = expand(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    targets.rename({"asset": "code"}).write_parquet(output)
    diagnostics_path = output.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(strategy.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s (%s rows)", output, targets.height)
    LOGGER.info("Portfolio diagnostics: %s", diagnostics_path)
