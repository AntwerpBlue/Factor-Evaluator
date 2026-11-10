from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import polars as pl
from scipy.optimize import linprog, minimize
from scipy import sparse

from .data import PortfolioDataset, normalize_panel
from .risk import FactorCovarianceMatrix, RiskMatrixProvider, decision_timestamp
from .alpha_calibration import CalibrationState, RankReturnCalibrator, signal_ranks
from .metrics import annualized_return, annualized_volatility, safe_mean, sharpe_ratio


@dataclass(frozen=True)
class IndexEnhancementConfig:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.002
    max_weight: float = 0.03
    max_active_weight: float = 0.01
    tracking_error: float | None = None
    long_only: bool = True
    active_exposure_penalty: float = 0.0
    turnover_cost_bps: float = 8.0
    turnover_cost_alpha_fraction: float | None = None
    max_turnover: float | None = None
    active_exposure_limit: float | None = None
    max_trade_weight: float | None = None
    zero_benchmark: bool = False
    require_covariance: bool = False
    alpha_transform: Literal["none", "zscore", "rank_zscore", "calibrated_rank"] = "zscore"
    alpha_scale: float = 1.0
    alpha_clip: float | None = 5.0
    alpha_return_scale: float | None = None
    risk_annualization: float = 252.0
    exposure_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # None preserves the low-level legacy default; optimize_signal defaults to te_cost.
    objective_mode: Literal["legacy", "te_cost"] | None = None
    buy_cost_bps: float = 8.0
    sell_cost_bps: float = 8.0
    solver_time_limit: float = 30.0


@dataclass(frozen=True)
class AssetConstraints:
    """Weight restrictions at one rebalance; all bounds are intersected."""

    weight_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    allowed_assets: set[str] | None = None
    frozen: Sequence[str] = ()
    no_buy: Sequence[str] = ()
    no_sell: Sequence[str] = ()

    def assets(self) -> set[str]:
        return set(self.weight_bounds) | set(self.allowed_assets or ()) | set(self.frozen) | set(self.no_buy) | set(self.no_sell)


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    diagnostics: dict[str, Any]


def constraints_from_trade_status(trade_status, decision_time, assets, base=None, *, unknown_policy="reject") -> AssetConstraints:
    """Use only status observations available strictly before this decision."""
    cutoff = decision_timestamp(decision_time)
    base = base or AssetConstraints()
    rows = trade_status.to_dicts() if isinstance(trade_status, pl.DataFrame) else [
        {**value, "asset": asset} for asset, value in trade_status.items()
    ]
    latest = {}
    for row in rows:
        asset = str(row.get("asset", row.get("code", row.get("header"))))
        if row.get("observed_at") is None:
            continue
        timestamp = decision_timestamp(row["observed_at"])
        if timestamp < cutoff and (asset not in latest or timestamp > latest[asset][0]):
            latest[asset] = (timestamp, row)
    frozen, no_buy, no_sell = set(base.frozen), set(base.no_buy), set(base.no_sell)
    for asset in assets:
        row = latest.get(asset, (None, {}))[1]
        status = row.get("status")
        label = row.get("limit_label")
        known = (status in {"normal", "limit_up", "limit_down", "suspended"}
            or label in {-1, 0, 1} or row.get("tradable") is not None
            or (row.get("can_buy") is not None and row.get("can_sell") is not None)
            or row.get("frozen") is True)
        if not known:
            if unknown_policy == "freeze":
                frozen.add(asset)
                continue
            raise ValueError(f"Unknown pre-decision trading status for {asset}")
        if status == "suspended" or row.get("tradable") is False or row.get("frozen") is True:
            frozen.add(asset)
        if status == "limit_up" or label == 1 or row.get("can_buy") is False:
            no_buy.add(asset)
        if status == "limit_down" or label == -1 or row.get("can_sell") is False:
            no_sell.add(asset)
    return AssetConstraints(base.weight_bounds, base.allowed_assets, tuple(frozen), tuple(no_buy), tuple(no_sell))


class IndexEnhancementStrategy:
    """Small convex index-enhancement optimizer.

    optimize_signal defaults to expected active return minus estimated trading
    costs with a hard TE budget. Low-level calls retain the legacy risk penalty
    objective unless objective_mode is supplied explicitly.
    """

    def __init__(self, config: IndexEnhancementConfig | None = None):
        self.config = config or IndexEnhancementConfig()
        self.diagnostics: list[dict[str, Any]] = []

    def optimize_signal(
        self, *, signal, decision_time, benchmark_weights, current_weights,
        risk_data: RiskMatrixProvider, trade_status,
        calibration_state: CalibrationState | None = None,
        asset_constraints: AssetConstraints | None = None,
        alpha_transform="calibrated_rank", unknown_status_policy="reject",
    ) -> OptimizationResult:
        """One decision, using current holdings and prior-day Barra; never updates calibration."""
        now = decision_timestamp(decision_time)
        if "values" in signal:
            if decision_timestamp(signal["observed_at"]) > now:
                raise ValueError("Signal is not available at decision_time")
            signal = signal["values"]
        config = replace(self.config, objective_mode=self.config.objective_mode or "te_cost", alpha_transform=alpha_transform)
        calibration_id, uncertainty = None, None
        if alpha_transform == "calibrated_rank":
            if calibration_state is None:
                raise ValueError("calibrated_rank requires an explicit calibration_state")
            if calibration_state.updated_at and decision_timestamp(calibration_state.updated_at) > now:
                raise ValueError("Calibration state was not available at decision_time")
            prediction = RankReturnCalibrator().predict(signal_ranks(signal), calibration_state, return_uncertainty=True)
            mu, uncertainty = prediction["mu"], prediction["uncertainty"]
            calibration_id, units = calibration_state.state_id, "return"
        else:
            mu = IndexEnhancementStrategy(config)._prepare_alpha(list(signal), list(signal.values()))
            units = "return" if alpha_transform == "none" or config.alpha_return_scale is not None else "score"
        base = asset_constraints or AssetConstraints()
        assets = sorted(set(signal) | set(benchmark_weights) | set(current_weights) | base.assets())
        restrictions = constraints_from_trade_status(trade_status, now, assets, base, unknown_policy=unknown_status_policy)
        snapshot = risk_data.snapshot(now, assets)
        result = IndexEnhancementStrategy(config).optimize_result(
            mu, benchmark_weights, current_weights, snapshot.covariance, assets,
            snapshot.covariance.exposures, exposure_names=snapshot.factor_names,
            asset_constraints=restrictions, alpha_units=units,
        )
        result.diagnostics.update(
            decision_time=now.isoformat(), barra_asof_date=snapshot.asof_date,
            alpha_calibration_state_id=calibration_id,
            alpha_calibration_basis=calibration_state.basis if calibration_id else None,
            alpha_uncertainty=uncertainty,
            active_exposures=result.diagnostics["named_active_exposures"],
        )
        return result

    def optimize(
        self,
        alpha: Mapping[str, float],
        benchmark_weights: Mapping[str, float],
        previous_weights: Mapping[str, float] | None = None,
        covariance: np.ndarray | None = None,
        assets: Sequence[str] | None = None,
        exposure_matrix: np.ndarray | None = None,
        *,
        exposure_names: Sequence[str] | None = None,
        asset_constraints: AssetConstraints | None = None,
    ) -> dict[str, float]:
        """Optimize already prepared, per-period expected returns; return weights only.

        Alpha preprocessing options apply to build_targets, not this low-level API.
        Covariance uses the same period as alpha; tracking_error is annualized.
        """
        return self.optimize_result(
            alpha, benchmark_weights, previous_weights, covariance, assets, exposure_matrix,
            exposure_names=exposure_names, asset_constraints=asset_constraints,
        ).weights

    def optimize_result(
        self,
        alpha: Mapping[str, float],
        benchmark_weights: Mapping[str, float],
        previous_weights: Mapping[str, float] | None = None,
        covariance: np.ndarray | None = None,
        assets: Sequence[str] | None = None,
        exposure_matrix: np.ndarray | None = None,
        *,
        alpha_units: Literal["return", "score"] = "return",
        exposure_names: Sequence[str] | None = None,
        asset_constraints: AssetConstraints | None = None,
    ) -> OptimizationResult:
        mode = self.config.objective_mode or "legacy"
        if mode == "te_cost" and (self.config.tracking_error is None or covariance is None):
            raise ValueError("te_cost requires an explicit annual tracking_error budget and risk covariance")
        if mode == "te_cost" and alpha_units != "return":
            raise ValueError("te_cost requires expected returns, not uncalibrated scores")
        restrictions = asset_constraints or AssetConstraints()
        assets = list(assets) if assets is not None else sorted(set(alpha) | set(benchmark_weights) | set(previous_weights or {}) | restrictions.assets())
        missing_assets = (set(alpha) | set(benchmark_weights)) - set(assets)
        if missing_assets:
            raise ValueError("assets must include all alpha and benchmark names")
        if not assets:
            raise ValueError("alpha and benchmark_weights are empty")
        expected = np.asarray([alpha.get(asset, 0.0) for asset in assets], dtype=float)
        benchmark = np.asarray(
            [benchmark_weights.get(asset, 0.0) for asset in assets], dtype=float
        )
        if benchmark.sum() <= 0:
            benchmark = (
                np.zeros(len(assets), dtype=float)
                if self.config.zero_benchmark
                else np.full(len(assets), 1.0 / len(assets))
            )
        else:
            benchmark /= benchmark.sum()
        previous = np.asarray(
            [
                (
                    previous_weights.get(asset, 0.0)
                    if previous_weights is not None
                    else benchmark[index]
                )
                for index, asset in enumerate(assets)
            ],
            dtype=float,
        )
        if previous_weights is None and self.config.zero_benchmark:
            previous = np.full(len(assets), 1.0 / len(assets))
        risk_source = "provided" if covariance is not None else "identity_regularizer"
        if covariance is None and (self.config.require_covariance or self.config.tracking_error is not None):
            raise ValueError(
                "Risk covariance is required but missing for the optimization date"
            )
        covariance = covariance if covariance is not None else np.eye(len(assets), dtype=float)
        if covariance is not None and not hasattr(covariance, "shape"):
            covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(assets), len(assets)):
            raise ValueError("covariance shape does not match optimizer universe")
        if exposure_matrix is None and (self.config.active_exposure_penalty > 0 or self.config.active_exposure_limit is not None or self.config.exposure_bounds):
            raise ValueError("Exposure controls require an exposure matrix for the optimization date")
        if alpha_units == "score" and self.config.turnover_cost_alpha_fraction is not None:
            raise ValueError("Cost/alpha constraint requires expected returns; set alpha_return_scale or use alpha_transform='none'")
        exposure_matrix = (
            np.asarray(exposure_matrix, dtype=float)
            if exposure_matrix is not None
            else None
        )
        if exposure_matrix is not None and exposure_matrix.shape[0] != len(assets):
            raise ValueError("exposure_matrix rows must match optimizer universe")

        names = list(exposure_names) if exposure_names is not None else [str(i) for i in range(exposure_matrix.shape[1] if exposure_matrix is not None else 0)]
        exposure_lower = np.full(len(names), -np.inf)
        exposure_upper = np.full(len(names), np.inf)
        if self.config.active_exposure_limit is not None:
            exposure_lower[:] = -self.config.active_exposure_limit
            exposure_upper[:] = self.config.active_exposure_limit
        for name, (lo, hi) in self.config.exposure_bounds.items():
            index = names.index(name)
            exposure_lower[index] = max(exposure_lower[index], lo)
            exposure_upper[index] = min(exposure_upper[index], hi)
        has_exposure_bounds = bool(self.config.exposure_bounds) or self.config.active_exposure_limit is not None

        cost_rate = self.config.turnover_cost_bps / 10_000.0

        def expected_alpha_gain(weight: np.ndarray) -> float:
            return float(expected @ (weight - previous))

        def turnover_cost(weight: np.ndarray) -> float:
            turnover_delta = weight - previous
            if mode == "te_cost":
                return float(self.config.buy_cost_bps / 10_000 * np.maximum(turnover_delta, 0).sum()
                    + self.config.sell_cost_bps / 10_000 * np.maximum(-turnover_delta, 0).sum())
            return float(cost_rate * np.abs(turnover_delta).sum())

        def turnover_cost_jac(weight: np.ndarray) -> np.ndarray:
            turnover_delta = weight - previous
            return cost_rate * np.sign(turnover_delta)

        def one_way_turnover(weight: np.ndarray) -> float:
            turnover_delta = weight - previous
            return float(0.5 * np.abs(turnover_delta).sum())

        def one_way_turnover_jac(weight: np.ndarray) -> np.ndarray:
            turnover_delta = weight - previous
            return 0.5 * np.sign(turnover_delta)

        def objective(weight: np.ndarray) -> float:
            active = weight - benchmark
            risk = self._risk_quadratic(covariance, active)
            turnover_delta = weight - previous
            turnover = np.sqrt(turnover_delta**2 + 1e-12).sum()
            exposure_penalty = 0.0
            if exposure_matrix is not None:
                active_exposure = exposure_matrix.T @ active
                exposure_penalty = float(active_exposure @ active_exposure)
            return (
                -float(expected @ weight)
                + self.config.risk_aversion * risk
                + self.config.turnover_penalty * turnover
                + self.config.active_exposure_penalty * exposure_penalty
            )

        def objective_jac(weight: np.ndarray) -> np.ndarray:
            active = weight - benchmark
            turnover_delta = weight - previous
            turnover_grad = turnover_delta / np.sqrt(turnover_delta**2 + 1e-12)
            gradient = (
                -expected
                + 2.0 * self.config.risk_aversion * self._risk_matvec(covariance, active)
                + self.config.turnover_penalty * turnover_grad
            )
            if exposure_matrix is not None:
                gradient += (
                    2.0
                    * self.config.active_exposure_penalty
                    * (exposure_matrix @ (exposure_matrix.T @ active))
                )
            return gradient

        if self.config.zero_benchmark:
            lower = np.full(
                len(assets), 0.0 if self.config.long_only else -self.config.max_weight
            )
            upper = np.full(len(assets), self.config.max_weight)
        else:
            lower = np.maximum(
                0.0 if self.config.long_only else -self.config.max_weight,
                benchmark - self.config.max_active_weight,
            )
            upper = np.minimum(
                self.config.max_weight, benchmark + self.config.max_active_weight
            )
        if self.config.max_trade_weight is not None and previous_weights is not None:
            max_trade_weight = float(self.config.max_trade_weight)
            if max_trade_weight <= 0:
                raise ValueError("max_trade_weight must be positive")
            lower = np.maximum(lower, previous - max_trade_weight)
            upper = np.minimum(upper, previous + max_trade_weight)
        if (restrictions.frozen or restrictions.no_buy or restrictions.no_sell) and previous_weights is None:
            raise ValueError("Frozen/no-buy/no-sell restrictions require current holdings")
        positions = {asset: i for i, asset in enumerate(assets)}
        for asset, (lo, hi) in restrictions.weight_bounds.items():
            i = positions[asset]
            lower[i], upper[i] = max(lower[i], lo), min(upper[i], hi)
        if restrictions.allowed_assets is not None:
            for i, asset in enumerate(assets):
                if asset not in restrictions.allowed_assets:
                    # Legacy holdings outside the investment pool may be unsellable.
                    locked = asset in restrictions.frozen or asset in restrictions.no_sell
                    retained = previous[i] if locked else 0.0
                    lower[i], upper[i] = max(lower[i], retained), min(upper[i], retained)
        for asset in restrictions.frozen:
            i = positions[asset]
            lower[i], upper[i] = max(lower[i], previous[i]), min(upper[i], previous[i])
        for asset in restrictions.no_buy:
            i = positions[asset]
            upper[i] = min(upper[i], previous[i])
        for asset in restrictions.no_sell:
            i = positions[asset]
            lower[i] = max(lower[i], previous[i])
        if np.any(lower > upper) or lower.sum() > 1.0 or upper.sum() < 1.0:
            conflicts = [asset for asset, lo, hi in zip(assets, lower, upper) if lo > hi]
            raise ValueError(f"Weight bounds are infeasible; conflicting assets={conflicts}, sum(lower)={lower.sum():.6g}, sum(upper)={upper.sum():.6g}")
        constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}
        ]
        if has_exposure_bounds:
            # Keep exact neutrality as equality constraints, including zero bands.
            equality = exposure_lower == exposure_upper
            lo_mask = np.isfinite(exposure_lower) & ~equality
            hi_mask = np.isfinite(exposure_upper) & ~equality
            for mask, kind, sign, bound in (
                (equality, "eq", 1, exposure_lower),
                (lo_mask, "ineq", 1, exposure_lower),
                (hi_mask, "ineq", -1, exposure_upper),
            ):
                if np.any(mask):
                    matrix = sign * exposure_matrix[:, mask].T
                    offset = sign * bound[mask] + matrix @ benchmark
                    constraints.append({"type": kind,
                        "fun": lambda weight, m=matrix, b=offset: m @ weight - b,
                        "jac": lambda weight, m=matrix: m})
        initial = self._project_box_simplex(benchmark, lower, upper)
        max_turnover = None
        if self.config.max_turnover is not None:
            max_turnover = float(self.config.max_turnover)
            if max_turnover <= 0:
                raise ValueError("max_turnover must be positive")

        use_linear_program = (
            self.config.risk_aversion == 0
            and self.config.turnover_penalty == 0
            and self.config.active_exposure_penalty == 0
            and self.config.tracking_error is None
        )
        def finish(solution: np.ndarray, solver: str, status: str) -> OptimizationResult:
            # Only evaluate the produced portfolio; no input scans or matrix audits.
            active = solution - benchmark
            delta = solution - previous
            turnover = one_way_turnover(solution)
            cost = turnover_cost(solution)
            gain = expected_alpha_gain(solution)
            active_variance = self._risk_quadratic(covariance, active)
            annual_te = float(np.sqrt(max(active_variance, 0.0) * self.config.risk_annualization))
            exposures = exposure_matrix.T @ active if exposure_matrix is not None else np.array([])
            slack = {
                "weight_sum": 1e-6 - abs(float(solution.sum()) - 1.0),
                "lower_weight": float(np.min(solution - lower)),
                "upper_weight": float(np.min(upper - solution)),
            }
            if max_turnover is not None:
                slack["turnover"] = max_turnover - turnover
            if self.config.tracking_error is not None:
                slack["tracking_error_annual"] = self.config.tracking_error - annual_te
            for i, name in enumerate(names):
                if np.isfinite(exposure_lower[i]):
                    slack[f"exposure:{name}:lower"] = float(exposures[i] - exposure_lower[i])
                if np.isfinite(exposure_upper[i]):
                    slack[f"exposure:{name}:upper"] = float(exposure_upper[i] - exposures[i])
            if self.config.turnover_cost_alpha_fraction is not None:
                slack["cost_alpha"] = self.config.turnover_cost_alpha_fraction * gain - cost
            violations = [name for name, value in slack.items() if not np.isfinite(value) or value < -1e-7]
            if not np.isfinite(solution).all() or violations:
                raise RuntimeError(f"Optimizer produced unusable weights: {violations}; constraint slacks={slack}")
            factor_risk, specific_risk = None, None
            if isinstance(covariance, FactorCovarianceMatrix):
                factor_active = covariance.exposures.T @ active
                factor_risk = float(factor_active @ covariance.factor_covariance @ factor_active * self.config.risk_annualization)
                specific_risk = float(np.sum((covariance.specific_variance + covariance.diagonal_jitter) * active**2) * self.config.risk_annualization)
            return OptimizationResult(dict(zip(assets, map(float, solution))), {
                "solver": solver, "solver_status": status, "constraints_satisfied": True,
                "objective_mode": mode,
                "constraint_slacks": slack, "alpha_units": alpha_units,
                "expected_alpha": float(expected @ solution),
                "expected_active_alpha": float(expected @ active),
                "expected_alpha_gain": gain, "expected_cost": cost,
                "turnover": turnover, "max_trade_weight": float(np.max(np.abs(delta))),
                "weight_sum": float(solution.sum()), "gross_weight": float(np.abs(solution).sum()),
                "max_weight": float(np.max(solution)), "max_active_weight": float(np.max(np.abs(active))),
                "active_exposures": exposures.tolist(), "risk_source": risk_source,
                "named_active_exposures": dict(zip(names, exposures.tolist())),
                "predicted_returns": dict(zip(assets, map(float, expected))),
                "predicted_te": annual_te if risk_source == "provided" else None,
                "factor_risk": factor_risk, "specific_risk": specific_risk,
                "risk_annualization": self.config.risk_annualization,
                "tracking_error_annual": annual_te if risk_source == "provided" else None,
                "volatility_annual": float(np.sqrt(max(self._risk_quadratic(covariance, solution), 0.0) * self.config.risk_annualization)) if risk_source == "provided" else None,
            })

        if mode == "te_cost":
            solution, status = self._cost_budget_optimize(
                expected, benchmark, previous, lower, upper, covariance,
                exposure_matrix, exposure_lower, exposure_upper,
            )
            return finish(solution, "clarabel", status)

        if use_linear_program:
            solution = self._linear_optimize(
                expected,
                benchmark,
                previous,
                lower,
                upper,
                exposure_matrix,
                max_turnover,
                self.config.turnover_cost_alpha_fraction,
                cost_rate,
                exposure_lower,
                exposure_upper,
            )
            return finish(solution, "highs", "converged")
        use_slsqp = (
            self.config.tracking_error is not None
            or has_exposure_bounds
            or max_turnover is not None
            or self.config.turnover_cost_alpha_fraction is not None
        )
        if not use_slsqp:
            solution, status = self._projected_gradient_optimize(
                initial,
                lower,
                upper,
                objective,
                objective_jac,
                covariance,
            )
            return finish(solution, "projected_gradient", status)

        if self.config.tracking_error is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weight: float(
                        self.config.tracking_error**2
                        - self.config.risk_annualization * self._risk_quadratic(covariance, weight - benchmark)
                    ),
                    "jac": lambda weight: -2.0 * self.config.risk_annualization * self._risk_matvec(covariance, weight - benchmark),
                }
            )
        if use_slsqp and self.config.turnover_cost_alpha_fraction is not None:
            fraction = float(self.config.turnover_cost_alpha_fraction)
            if fraction <= 0:
                raise ValueError("turnover_cost_alpha_fraction must be positive")
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weight: float(
                        fraction * expected_alpha_gain(weight) - turnover_cost(weight)
                    ),
                    "jac": lambda weight: fraction * expected - turnover_cost_jac(weight),
                }
            )
        if max_turnover is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weight: float(max_turnover - one_way_turnover(weight)),
                    "jac": lambda weight: -one_way_turnover_jac(weight),
                }
            )
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=objective_jac,
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Index enhancement optimization failed: {result.message}")
        return finish(result.x, "SLSQP", str(result.message))

    def _cost_budget_optimize(self, mu, benchmark, previous, lower, upper, covariance, exposures, exposure_lower, exposure_upper):
        """Sparse SOCP over active weights, buys, sells and factor exposures.

        Factor-form input never forms an N x N covariance or quadratic objective.
        """
        import clarabel

        n = len(mu)
        if isinstance(covariance, FactorCovarianceMatrix):
            x = sparse.csc_matrix(covariance.exposures.T)
            omega = covariance.factor_covariance
            d = covariance.specific_variance + covariance.diagonal_jitter
        else:
            # Dense covariance is only used when explicitly supplied by the caller.
            x = sparse.eye(n, format="csc")
            omega = np.asarray(covariance)
            d = np.zeros(n)
        k, size = x.shape[0], 3 * n + x.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        root = sparse.csc_matrix(np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T)
        identity = sparse.eye(n, format="csc")
        znn, znk = sparse.csc_matrix((n, n)), sparse.csc_matrix((n, k))
        blocks, rhs, cones = [], [], []

        def add(matrix, values, cone):
            blocks.append(sparse.csc_matrix(matrix))
            rhs.append(np.asarray(values, dtype=float))
            cones.append(cone(len(values)))

        # a = w-b, a-buy+sell = p-b, and f = X' a.
        add(sparse.hstack([np.ones((1, n)), sparse.csc_matrix((1, 2*n+k))]), [1-benchmark.sum()], clarabel.ZeroConeT)
        add(sparse.hstack([identity, -identity, identity, znk]), previous-benchmark, clarabel.ZeroConeT)
        add(sparse.hstack([-x, sparse.csc_matrix((k, 2*n)), sparse.eye(k)]), np.zeros(k), clarabel.ZeroConeT)
        add(sparse.hstack([identity, znn, znn, znk]), upper-benchmark, clarabel.NonnegativeConeT)
        add(sparse.hstack([-identity, znn, znn, znk]), benchmark-lower, clarabel.NonnegativeConeT)
        add(sparse.hstack([znn, -identity, znn, znk]), np.zeros(n), clarabel.NonnegativeConeT)
        add(sparse.hstack([znn, znn, -identity, znk]), np.zeros(n), clarabel.NonnegativeConeT)
        if exposures is not None:
            e = sparse.hstack([sparse.csc_matrix(exposures.T), sparse.csc_matrix((exposures.shape[1], 2*n+k))], format="csc")
            equal = exposure_lower == exposure_upper
            for mask, sign, limits, cone in (
                (equal, 1, exposure_lower, clarabel.ZeroConeT),
                (np.isfinite(exposure_upper) & ~equal, 1, exposure_upper, clarabel.NonnegativeConeT),
                (np.isfinite(exposure_lower) & ~equal, -1, exposure_lower, clarabel.NonnegativeConeT),
            ):
                if mask.any():
                    add(sign*e[mask], sign*limits[mask], cone)
        buy_rate, sell_rate = self.config.buy_cost_bps/10_000, self.config.sell_cost_bps/10_000
        if self.config.max_turnover is not None:
            add(np.r_[np.zeros(n), np.full(2*n, .5), np.zeros(k)][None, :], [self.config.max_turnover], clarabel.NonnegativeConeT)
        if self.config.turnover_cost_alpha_fraction is not None:
            fraction = self.config.turnover_cost_alpha_fraction
            row = np.r_[-fraction*mu, np.full(n, buy_rate), np.full(n, sell_rate), np.zeros(k)]
            add(row[None, :], [fraction*float(mu @ (benchmark-previous))], clarabel.NonnegativeConeT)
        risk_rows = sparse.vstack([
            sparse.csc_matrix((1, size)),
            sparse.hstack([sparse.diags(-np.sqrt(d)), znn, znn, znk]),
            sparse.hstack([sparse.csc_matrix((k, 3*n)), -root]),
        ], format="csc")
        add(risk_rows, np.r_[self.config.tracking_error / np.sqrt(self.config.risk_annualization), np.zeros(n+k)], clarabel.SecondOrderConeT)
        q = np.r_[-mu, np.full(n, buy_rate), np.full(n, sell_rate), np.zeros(k)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.time_limit = self.config.solver_time_limit
        settings.tol_gap_abs = settings.tol_feas = settings.tol_gap_rel = 1e-9
        solver = clarabel.DefaultSolver(sparse.csc_matrix((size, size)), q,
            sparse.vstack(blocks, format="csc"), np.concatenate(rhs), cones, settings)
        result = solver.solve()
        status = str(result.status)
        if status not in {"Solved", "AlmostSolved"}:
            raise RuntimeError(f"Portfolio optimization failed: {status}")
        return benchmark + np.asarray(result.x[:n]), status

    @staticmethod
    def _risk_matvec(covariance: Any, value: np.ndarray) -> np.ndarray:
        if hasattr(covariance, "matvec"):
            return covariance.matvec(value)
        return np.asarray(covariance, dtype=float) @ value

    @classmethod
    def _risk_quadratic(cls, covariance: Any, value: np.ndarray) -> float:
        if hasattr(covariance, "quadratic"):
            return float(covariance.quadratic(value))
        matrix = np.asarray(covariance, dtype=float)
        return float(value @ matrix @ value)

    @staticmethod
    def _risk_spectral_bound(covariance: Any) -> float:
        if hasattr(covariance, "max_eigenvalue_bound"):
            return float(covariance.max_eigenvalue_bound())
        matrix = np.asarray(covariance, dtype=float)
        try:
            return float(np.linalg.eigvalsh(matrix).max())
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(matrix, ord=2))

    @staticmethod
    def _linear_optimize(
        expected: np.ndarray,
        benchmark: np.ndarray,
        previous: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        exposure_matrix: np.ndarray | None,
        max_turnover: float | None = None,
        cost_fraction: float | None = None,
        cost_rate: float = 0.0,
        exposure_lower: np.ndarray | None = None,
        exposure_upper: np.ndarray | None = None,
    ) -> np.ndarray:
        n = expected.size
        extended = max_turnover is not None or cost_fraction is not None
        c = -expected
        bounds = list(zip(lower, upper))
        a_eq = np.ones((1, n))
        b_eq = np.asarray([1.0])
        a_ub, b_ub = None, None
        if extended:
            # u >= abs(w - previous), with sparse blocks for large universes.
            identity = sparse.eye(n, format="csr")
            blocks = [sparse.hstack([identity, -identity]), sparse.hstack([-identity, -identity])]
            limits = [previous, -previous]
            if max_turnover is not None:
                blocks.append(sparse.csr_matrix(np.r_[np.zeros(n), np.full(n, 0.5)][None, :]))
                limits.append(np.array([max_turnover]))
            if cost_fraction is not None:
                blocks.append(sparse.csr_matrix(np.r_[-cost_fraction * expected, np.full(n, cost_rate)][None, :]))
                limits.append(np.array([-cost_fraction * float(expected @ previous)]))
            a_ub = sparse.vstack(blocks, format="csr")
            b_ub = np.concatenate(limits)
            c = np.r_[-expected, np.zeros(n)]
            a_eq = sparse.csr_matrix(np.r_[np.ones(n), np.zeros(n)][None, :])
            bounds += [(0.0, None)] * n
        if exposure_matrix is not None:
            matrix = sparse.csr_matrix(exposure_matrix.T)
            if extended:
                matrix = sparse.hstack([matrix, sparse.csr_matrix((matrix.shape[0], n))], format="csr")
            benchmark_exposure = exposure_matrix.T @ benchmark
            equality = exposure_lower == exposure_upper
            if np.any(equality):
                a_eq = sparse.vstack([sparse.csr_matrix(a_eq), matrix[equality]], format="csr")
                b_eq = np.r_[b_eq, exposure_lower[equality] + benchmark_exposure[equality]]
            lo_mask = np.isfinite(exposure_lower) & ~equality
            hi_mask = np.isfinite(exposure_upper) & ~equality
            if np.any(lo_mask) or np.any(hi_mask):
                exposure_a = sparse.vstack([-matrix[lo_mask], matrix[hi_mask]], format="csr")
                exposure_b = np.r_[-exposure_lower[lo_mask] - benchmark_exposure[lo_mask], exposure_upper[hi_mask] + benchmark_exposure[hi_mask]]
                a_ub = sparse.vstack([a_ub, exposure_a], format="csr") if a_ub is not None else exposure_a
                b_ub = np.r_[b_ub, exposure_b] if b_ub is not None else exposure_b
        result = linprog(
            c,
            A_ub=a_ub, b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Index enhancement linear optimization failed: {result.message}")
        return np.asarray(result.x[:n], dtype=float)

    @staticmethod
    def _project_box_simplex(
        value: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        if lower.sum() > 1.0 or upper.sum() < 1.0:
            raise ValueError("Weight bounds are infeasible; relax max_weight/max_active_weight")
        low = float(np.min(value - upper))
        high = float(np.max(value - lower))
        for _ in range(100):
            mid = (low + high) / 2.0
            projected = np.clip(value - mid, lower, upper)
            if projected.sum() > 1.0:
                low = mid
            else:
                high = mid
        projected = np.clip(value - high, lower, upper)
        total = projected.sum()
        if total <= 0:
            return lower.copy()
        residual = 1.0 - total
        if abs(residual) > 1e-12:
            room = upper - projected if residual > 0 else projected - lower
            room_sum = room.sum()
            if room_sum > 0:
                projected += residual * room / room_sum
        return projected

    @classmethod
    def _projected_gradient_optimize(
        cls,
        initial: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        objective,
        objective_jac,
        covariance: np.ndarray,
        *,
        max_iter: int = 500,
        tolerance: float = 1e-9,
    ) -> tuple[np.ndarray, str]:
        weight = initial.copy()
        spectral = cls._risk_spectral_bound(covariance)
        step = 1.0 / max(2.0 * spectral + 1e-3, 1e-3)
        value = objective(weight)
        for _ in range(max_iter):
            gradient = objective_jac(weight)
            accepted = False
            local_step = step
            for _ in range(30):
                candidate = cls._project_box_simplex(weight - local_step * gradient, lower, upper)
                direction = candidate - weight
                if np.linalg.norm(direction, ord=1) < tolerance:
                    return candidate, "converged"
                candidate_value = objective(candidate)
                if candidate_value <= value + 1e-4 * float(gradient @ direction):
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                return weight, "line_search_stopped"
            if abs(value - candidate_value) < tolerance:
                return candidate, "converged"
            weight = candidate
            value = candidate_value
            step = min(local_step * 1.2, 10.0)
        return weight, "iteration_limit"

    def build_targets(
        self,
        signals: pl.DataFrame,
        benchmark_weights: pl.DataFrame,
        covariance_by_date: Mapping[str, np.ndarray] | Callable[[str, list[str]], np.ndarray | None] | None = None,
        exposure_by_date: Callable[[str, list[str]], np.ndarray | None] | None = None,
        asset_returns: pl.DataFrame | None = None,
        initial_weights: Mapping[str, float] | None = None,
        asset_constraints_by_date: Mapping[str, AssetConstraints] | Callable[[str], AssetConstraints] | None = None,
        exposure_names: Sequence[str] | None = None,
        calibration_states_by_date: Mapping[str, CalibrationState] | None = None,
    ) -> pl.DataFrame:
        self.diagnostics = []
        signal_panel = normalize_panel(signals)
        benchmark_panel = normalize_panel(benchmark_weights)
        returns_panel = normalize_panel(asset_returns) if asset_returns is not None else None
        signal_columns = [c for c in signal_panel.columns if c not in {"date", "asset"}]
        weight_columns = [c for c in benchmark_panel.columns if c not in {"date", "asset"}]
        if len(signal_columns) != 1 or len(weight_columns) != 1:
            raise ValueError("signals and benchmark_weights each require one value column")
        signal_column, weight_column = signal_columns[0], weight_columns[0]
        signal_dates = {
            day["date"][0]: self._prepare_alpha(day["asset"].to_list(), day[signal_column].to_list(),
                calibration_states_by_date[day["date"][0]] if calibration_states_by_date is not None else None,
                day["date"][0])
            for day in signal_panel.partition_by("date", maintain_order=True)
        }
        benchmark_dates = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day[weight_column].to_list()))
            for day in benchmark_panel.partition_by("date", maintain_order=True)
        }
        return_dates: dict[str, dict[str, float]] = {}
        if returns_panel is not None:
            return_columns = [c for c in returns_panel.columns if c not in {"date", "asset"}]
            if not return_columns:
                raise ValueError("asset_returns requires a return value column")
            return_column = "return" if "return" in return_columns else return_columns[0]
            return_dates = {
                day["date"][0]: dict(zip(day["asset"].to_list(), day[return_column].to_list()))
                for day in returns_panel.partition_by("date", maintain_order=True)
            }
        rows = []
        previous = dict(initial_weights) if initial_weights is not None else None
        previous_date: str | None = None
        for date in sorted(set(signal_dates) & set(benchmark_dates)):
            previous_for_opt = previous
            if previous and previous_date in return_dates:
                daily_returns = return_dates[previous_date]
                assets_for_drift = set(previous) | set(daily_returns)
                holding_pnl = sum(
                    previous.get(asset, 0.0) * float(daily_returns.get(asset, 0.0) or 0.0)
                    for asset in assets_for_drift
                )
                denominator = 1.0 + holding_pnl
                previous_for_opt = {
                    asset: previous.get(asset, 0.0)
                    * (1.0 + float(daily_returns.get(asset, 0.0) or 0.0))
                    / denominator
                    for asset in assets_for_drift
                }
            restrictions = (
                asset_constraints_by_date(date) if callable(asset_constraints_by_date)
                else asset_constraints_by_date[date] if asset_constraints_by_date is not None
                else AssetConstraints()
            )
            assets = sorted(set(signal_dates[date]) | set(benchmark_dates[date]) | set(previous_for_opt or {}) | restrictions.assets())
            snapshot = None
            if isinstance(covariance_by_date, RiskMatrixProvider):
                snapshot = covariance_by_date.snapshot(date, assets)
                covariance = snapshot.covariance
            elif callable(covariance_by_date):
                covariance = covariance_by_date(date, assets)
            else:
                covariance = (covariance_by_date or {}).get(date)
            exposure_matrix = snapshot.covariance.exposures if snapshot else exposure_by_date(date, assets) if exposure_by_date is not None else None
            if covariance is None and covariance_by_date is not None:
                raise ValueError(f"Risk covariance unavailable on {date}")
            if exposure_matrix is None and exposure_by_date is not None:
                raise ValueError(f"Risk exposures unavailable on {date}")
            result = self.optimize_result(
                signal_dates[date],
                benchmark_dates[date],
                previous_for_opt,
                covariance,
                assets=assets,
                exposure_matrix=exposure_matrix,
                alpha_units="return" if self.config.alpha_transform in {"none", "calibrated_rank"} or self.config.alpha_return_scale is not None else "score",
                exposure_names=snapshot.factor_names if snapshot else exposure_names if exposure_names is not None else getattr(exposure_by_date, "exposure_columns", None),
                asset_constraints=restrictions,
            )
            weights = result.weights
            result.diagnostics["barra_asof_date"] = snapshot.asof_date if snapshot else None
            result.diagnostics["alpha_calibration_state_id"] = calibration_states_by_date[date].state_id if calibration_states_by_date else None
            self.diagnostics.append({"date": date, **result.diagnostics})
            rows.extend(
                {"date": date, "asset": asset, "weight": weight}
                for asset, weight in weights.items()
            )
            previous = weights
            previous_date = date
        if not rows:
            return pl.DataFrame(schema={"date": pl.String, "asset": pl.String, "weight": pl.Float64})
        return pl.DataFrame(rows)

    def _prepare_alpha(self, assets: list[str], values: list[float], calibration_state=None, decision_time=None) -> dict[str, float]:
        if self.config.alpha_transform == "calibrated_rank":
            if calibration_state is None:
                raise ValueError("calibrated_rank requires a previously computed calibration state")
            if decision_time and calibration_state.updated_at and decision_timestamp(calibration_state.updated_at) > decision_timestamp(decision_time):
                raise ValueError("Calibration state was not available at decision_time")
            return RankReturnCalibrator().predict(signal_ranks(dict(zip(assets, values))), calibration_state)
        raw = np.asarray(values, dtype=float)
        finite = np.isfinite(raw)
        transformed = np.zeros_like(raw, dtype=float)
        if finite.any():
            valid = raw[finite]
            if self.config.alpha_transform == "none":
                transformed[finite] = valid
            elif self.config.alpha_transform == "zscore":
                mean = float(np.nanmean(valid))
                std = float(np.nanstd(valid))
                transformed[finite] = (valid - mean) / std if std > 1e-12 else 0.0
            elif self.config.alpha_transform == "rank_zscore":
                order = np.argsort(valid, kind="mergesort")
                ranked = np.empty(valid.size, dtype=float)
                ranked[order] = np.arange(1, valid.size + 1, dtype=float)
                mean = float(ranked.mean())
                std = float(ranked.std())
                transformed[finite] = (ranked - mean) / std if std > 1e-12 else 0.0
            else:
                raise ValueError(f"Unknown alpha_transform: {self.config.alpha_transform}")
        transformed *= float(self.config.alpha_scale)
        if self.config.alpha_clip is not None:
            clip = float(self.config.alpha_clip)
            if clip <= 0:
                raise ValueError("alpha_clip must be positive")
            transformed = np.clip(transformed, -clip, clip)
        if self.config.alpha_return_scale is not None:
            transformed *= self.config.alpha_return_scale
        return dict(zip(assets, transformed))


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    annualization: int = 240
    transaction_cost_bps: float = 8.0
    slot_weights: Mapping[str, float] = field(default_factory=lambda: {"1500_1": 1.0})
    pf_by_time: Mapping[str, float] = field(default_factory=lambda: {"1500_1": 0.9})
    max_deviation: float = 0.005
    max_holding: float = 0.005
    target_volatility: float = 0.05
    charge_initial_trade: bool = False
    rebalance_frequency: int = 1

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be positive")
        if not self.slot_weights or sum(self.slot_weights.values()) <= 0:
            raise ValueError("slot_weights must contain a positive allocation")


@dataclass
class PortfolioBacktestResult:
    portfolio: pl.DataFrame
    trades: pl.DataFrame
    pnl_daily: pl.DataFrame
    trading_pnl_by_time: pl.DataFrame
    summary: dict[str, Any]
    decomposition: pl.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def export(
        self,
        output_dir: str | Path,
        *,
        product_name: str = "ZeroIndex",
        test_name: str = "factor",
    ) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"{product_name}_intraday"
        start = self.pnl_daily["__index_level_0__"].min().strftime("%Y%m%d")
        end = self.pnl_daily["__index_level_0__"].max().strftime("%Y%m%d")
        self.metadata["chart_title"] = (
            f"{test_name}/{product_name} | maxDev={self.metadata.get('max_deviation', 0):.2%}, "
            f"maxHold={self.metadata.get('max_holding', 0):.2%}, "
            f"targetVol={self.metadata.get('target_volatility', 0):.2%} "
            f"[{start} - {end}]"
        )
        outputs = {
            f"{prefix}_port.parquet": self.portfolio,
            f"{prefix}_trade.parquet": self.trades,
            f"{prefix}_pnl_daily.parquet": self.pnl_daily,
            f"{prefix}_trading_pnl_by_time.parquet": self.trading_pnl_by_time,
        }
        files = []
        for filename, frame in outputs.items():
            path = output / filename
            frame.write_parquet(path)
            files.append(path)
        summary_path = output / "intraday_summary.json"
        summary_payload: dict[str, Any] = {}
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_payload[product_name] = self.summary
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files.append(summary_path)
        config_path = output / "factor_config.json"
        config_path.write_text(
            json.dumps(
                {"test_name": test_name, "backtest_config": self.metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files.append(config_path)
        files.extend(self._write_performance_charts(output, prefix))
        if self.decomposition is not None:
            csv_path = output / f"{test_name}_{product_name}_decomp.csv"
            self.decomposition.write_csv(csv_path)
            files.append(csv_path)
            chart_path = output / f"{test_name}_{product_name}_decomp.png"
            self._write_decomposition_chart(chart_path)
            files.append(chart_path)
        done = output / "DONE.ok"
        done.write_text("ok\n", encoding="utf-8")
        files.append(done)
        return files

    def _write_performance_charts(self, output: Path, prefix: str) -> list[Path]:
        from .portfolio_report import write_performance_charts

        return write_performance_charts(self, output, prefix)

    def _write_decomposition_chart(self, path: Path) -> None:
        from .portfolio_report import write_decomposition_chart

        write_decomposition_chart(self, path)


class PortfolioBacktester:
    def __init__(self, config: PortfolioBacktestConfig | None = None):
        self.config = config or PortfolioBacktestConfig()

    def run(
        self, target_weights: pl.DataFrame, dataset: PortfolioDataset
    ) -> PortfolioBacktestResult:
        data = dataset.normalized()
        weights = normalize_panel(target_weights)
        value_columns = [c for c in weights.columns if c not in {"date", "asset"}]
        if "weight" not in weights.columns:
            if len(value_columns) != 1:
                raise ValueError("target_weights require one value column named 'weight'")
            weights = weights.rename({value_columns[0]: "weight"})
        weights = weights.with_columns(pl.col("weight").cast(pl.Float64))
        targets = self._targets_by_date(weights)
        returns_by_date = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day["return"].to_list()))
            for day in data.returns.partition_by("date", maintain_order=True)
        }
        benchmark = dict(data.benchmark.iter_rows())
        dates = sorted(set(returns_by_date) & set(benchmark))
        if targets:
            first_target_date = min(targets)
            dates = [date for date in dates if date >= first_target_date]
        if not dates:
            raise ValueError("No overlapping return, benchmark, and target dates")
        assets = sorted(
            set(data.returns["asset"].to_list())
            | {asset for target in targets.values() for asset in target}
        )
        portfolio_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        pnl_rows: list[dict[str, Any]] = []
        slot_rows: list[dict[str, Any]] = []
        previous: dict[str, float] = {}
        previous_returns: dict[str, float] = {}
        latest_target: dict[str, float] = {}
        realized_targets: dict[str, dict[str, float]] = {}
        cost_rate = self.config.transaction_cost_bps / 10_000.0
        slot_total = float(sum(self.config.slot_weights.values()))
        for index, date in enumerate(dates):
            daily_returns = returns_by_date[date]
            pretrade = previous
            if index % self.config.rebalance_frequency == 0 and date in targets:
                latest_target = targets[date]
            target = latest_target or pretrade
            trade = {
                asset: target.get(asset, 0.0) - pretrade.get(asset, 0.0)
                for asset in assets
            }
            turnover = 0.5 * sum(abs(value) for value in trade.values())
            trading_pnl = -cost_rate * sum(abs(value) for value in trade.values())
            if index == 0 and not self.config.charge_initial_trade:
                trade = {asset: 0.0 for asset in assets}
                turnover = 0.0
                trading_pnl = 0.0
            holding_pnl = sum(
                target.get(asset, 0.0) * daily_returns.get(asset, 0.0)
                for asset in assets
            )
            bmk_ret = float(benchmark[date] or 0.0)
            holding_alpha = holding_pnl - bmk_ret
            alpha = holding_alpha + trading_pnl
            timestamp = self._timestamp(date)
            realized_targets[date] = target
            for slot, allocation in self.config.slot_weights.items():
                fraction = allocation / slot_total
                portfolio_rows.append({
                    **{asset: target.get(asset, 0.0) for asset in assets},
                    "trade_ts": timestamp,
                    "slot_key": slot,
                })
                trade_rows.append({
                    **{asset: value * fraction for asset, value in trade.items()},
                    "trade_ts": timestamp,
                    "slot_key": slot,
                })
            pnl_rows.append(
                {
                    "holding_pnl": holding_pnl,
                    "bmk_ret": bmk_ret,
                    "holding_alpha": holding_alpha,
                    "trading_pnl": trading_pnl,
                    "alpha": alpha,
                    "__index_level_0__": self._date(date),
                }
            )
            slot_rows.append(
                {
                    **{
                        slot: trading_pnl * allocation / slot_total
                        for slot, allocation in self.config.slot_weights.items()
                    },
                    "__index_level_0__": self._date(date),
                }
            )
            denominator = 1.0 + holding_pnl
            previous = {
                asset: target.get(asset, 0.0)
                * (1.0 + daily_returns.get(asset, 0.0))
                / denominator
                for asset in assets
            }
            previous_returns = daily_returns
        pnl = pl.DataFrame(pnl_rows)
        portfolio_frame = pl.DataFrame(portfolio_rows)
        trade_frame = pl.DataFrame(trade_rows)
        slot_frame = pl.DataFrame(slot_rows)
        from .portfolio_report import build_portfolio_summary
        summary = build_portfolio_summary(
            pnl, trade_frame, self.config.annualization, dict(self.config.pf_by_time)
        )
        decomposition = self._decompose(
            dates, previous_targets=realized_targets, dataset=data, pnl=pnl
        )
        return PortfolioBacktestResult(
            portfolio=portfolio_frame,
            trades=trade_frame,
            pnl_daily=pnl,
            trading_pnl_by_time=slot_frame,
            summary=summary,
            decomposition=decomposition,
            metadata=asdict(self.config),
        )

    @staticmethod
    def _targets_by_date(weights: pl.DataFrame) -> dict[str, dict[str, float]]:
        result = {}
        for day in weights.partition_by("date", maintain_order=True):
            raw = np.asarray(day["weight"].to_list(), dtype=float)
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite target weight on {day['date'][0]}")
            total = raw.sum()
            if total <= 0:
                raise ValueError(f"Target weights must sum positive on {day['date'][0]}")
            normalized = raw / total
            result[day["date"][0]] = dict(zip(day["asset"].to_list(), normalized))
        return result

    @staticmethod
    def _date(date: str):
        from datetime import datetime

        return datetime.strptime(date, "%Y%m%d")

    @classmethod
    def _timestamp(cls, date: str):
        return cls._date(date).replace(hour=15)

    @staticmethod
    def _lag_targets(
        dates: list[str], targets: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        lagged: dict[str, dict[str, float]] = {}
        previous: dict[str, float] = {}
        latest: dict[str, float] = {}
        for date in dates:
            lagged[date] = previous
            if date in targets:
                latest = targets[date]
            previous = latest
        return lagged

    def _decompose(
        self,
        dates: list[str],
        previous_targets: dict[str, dict[str, float]],
        dataset: PortfolioDataset,
        pnl: pl.DataFrame,
    ) -> pl.DataFrame | None:
        if dataset.styles is None and dataset.industry_exposures is None:
            return None
        style_columns = (
            [c for c in dataset.styles.columns if c not in {"date", "asset"}]
            if dataset.styles is not None else []
        )
        styles_by_date = (
            {day["date"][0]: day for day in dataset.styles.partition_by("date", maintain_order=True)}
            if dataset.styles is not None else {}
        )
        industry_by_date = (
            {day["date"][0]: day for day in dataset.industry_exposures.partition_by("date", maintain_order=True)}
            if dataset.industry_exposures is not None else {}
        )
        returns_by_date = {
            day["date"][0]: dict(zip(day["asset"].to_list(), day["return"].to_list()))
            for day in dataset.returns.partition_by("date", maintain_order=True)
        }
        benchmark_weights = {}
        if dataset.benchmark_weights is not None:
            weight_columns = [
                c for c in dataset.benchmark_weights.columns if c not in {"date", "asset"}
            ]
            weight_column = "weight" if "weight" in weight_columns else weight_columns[0]
            benchmark_weights = {}
            for day in dataset.benchmark_weights.partition_by("date", maintain_order=True):
                raw = np.asarray(day[weight_column].to_list(), dtype=float)
                total = float(np.nansum(raw))
                weights = raw / total if total > 0 else raw
                benchmark_weights[day["date"][0]] = dict(zip(day["asset"].to_list(), weights))
        style_return_map = self._style_return_map(dataset, style_columns)
        industry_return_map = self._industry_return_map(dataset)
        fallback_style_returns, fallback_industry_returns = self._cross_sectional_return_maps(
            dataset, style_columns
        )
        style_return_map = {**fallback_style_returns, **style_return_map}
        industry_return_map = {**fallback_industry_returns, **industry_return_map}
        rows = []
        for index, date in enumerate(dates):
            pnl_row = pnl.row(index, named=True)
            style_frame = styles_by_date.get(date)
            portfolio_weights = previous_targets.get(date, {})
            benchmark_date = benchmark_weights.get(date, {})
            style_contributions = {style: 0.0 for style in style_columns}
            if style_frame is not None:
                for style in style_columns:
                    exposure = sum(
                        (portfolio_weights.get(asset, 0.0) - benchmark_date.get(asset, 0.0))
                        * float(value or 0.0)
                        for asset, value in zip(
                            style_frame["asset"].to_list(), style_frame[style].to_list()
                        )
                    )
                    style_contributions[style] = exposure * style_return_map.get(
                        (date, style), 0.0
                    )
            style_ret = sum(style_contributions.values())
            industry_ret = self._industry_contribution(
                date, industry_by_date.get(date), industry_return_map,
                portfolio_weights, benchmark_date,
            )
            pure_before = pnl_row["holding_alpha"] - style_ret - industry_ret
            rows.append(
                {
                    "TRADE_DT": self._date(date).date(),
                    "SurplusRet": pnl_row["alpha"],
                    "StyleRet": style_ret,
                    "IndustryRet": industry_ret,
                    "pureAlpha_before_fee": pure_before,
                    "pureAlpha_after_fee": pure_before + pnl_row["trading_pnl"],
                    **style_contributions,
                }
            )
        return pl.DataFrame(rows)

    @staticmethod
    def _cross_sectional_return_maps(
        dataset: PortfolioDataset,
        style_columns: list[str],
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        if dataset.styles is None and dataset.industry_exposures is None:
            return {}, {}
        joined = dataset.returns
        if dataset.styles is not None and style_columns:
            joined = joined.join(dataset.styles.select("date", "asset", *style_columns), on=["date", "asset"], how="left")
        industry_columns: list[str] = []
        categorical_industry = False
        if dataset.industry_exposures is not None:
            if "industry" in dataset.industry_exposures.columns:
                categorical_industry = True
                joined = joined.join(dataset.industry_exposures.select("date", "asset", "industry"), on=["date", "asset"], how="left")
            else:
                industry_columns = [
                    c for c in dataset.industry_exposures.columns if c not in {"date", "asset"}
                ]
                joined = joined.join(
                    dataset.industry_exposures.select("date", "asset", *industry_columns),
                    on=["date", "asset"],
                    how="left",
                )
        style_map: dict[tuple[str, str], float] = {}
        industry_map: dict[tuple[str, str], float] = {}
        for day in joined.partition_by("date", maintain_order=True):
            date = day["date"][0]
            y = day["return"].fill_null(0.0).to_numpy().astype(float)
            columns: list[tuple[str, str]] = []
            matrix_parts: list[np.ndarray] = []
            for style in style_columns:
                if style in day.columns:
                    matrix_parts.append(day[style].fill_null(0.0).to_numpy().astype(float))
                    columns.append(("style", style))
            if categorical_industry and "industry" in day.columns:
                industries = sorted(
                    str(value) for value in day["industry"].drop_nulls().unique().to_list()
                )
                for industry in industries:
                    matrix_parts.append((day["industry"].cast(pl.String) == industry).cast(pl.Float64).to_numpy())
                    columns.append(("industry", industry))
            else:
                for industry in industry_columns:
                    if industry in day.columns:
                        matrix_parts.append(day[industry].fill_null(0.0).to_numpy().astype(float))
                        columns.append(("industry", industry))
            if not matrix_parts or len(y) <= len(matrix_parts):
                continue
            x = np.column_stack(matrix_parts)
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if valid.sum() <= x.shape[1]:
                continue
            coefficients = np.linalg.lstsq(x[valid], y[valid], rcond=None)[0]
            for (kind, name), coefficient in zip(columns, coefficients):
                if kind == "style":
                    style_map[(date, name)] = float(coefficient)
                else:
                    industry_map[(date, name)] = float(coefficient)
        return style_map, industry_map

    @staticmethod
    def _industry_return_map(
        dataset: PortfolioDataset,
    ) -> dict[tuple[str, str], float]:
        frame = dataset.industry_returns
        if frame is None:
            return {}
        date_column = "date" if "date" in frame.columns else "TRADE_DT"
        if {"industry", "return"}.issubset(frame.columns):
            return {
                (str(row[date_column]).replace("-", ""), str(row["industry"])): float(row["return"] or 0.0)
                for row in frame.iter_rows(named=True)
            }
        industry_columns = [c for c in frame.columns if c != date_column]
        return {
            (str(row[date_column]).replace("-", ""), industry): float(row[industry] or 0.0)
            for row in frame.iter_rows(named=True)
            for industry in industry_columns
        }

    @staticmethod
    def _industry_contribution(
        date: str,
        frame: pl.DataFrame | None,
        return_map: dict[tuple[str, str], float],
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
    ) -> float:
        if frame is None:
            return 0.0
        if "industry" in frame.columns:
            return sum(
                (portfolio_weights.get(asset, 0.0) - benchmark_weights.get(asset, 0.0))
                * return_map.get((date, str(industry)), 0.0)
                for asset, industry in zip(
                    frame["asset"].to_list(), frame["industry"].to_list()
                )
            )
        result = 0.0
        for industry in [c for c in frame.columns if c not in {"date", "asset"}]:
            exposure = sum(
                (portfolio_weights.get(asset, 0.0) - benchmark_weights.get(asset, 0.0))
                * float(value or 0.0)
                for asset, value in zip(
                    frame["asset"].to_list(), frame[industry].to_list()
                )
            )
            result += exposure * return_map.get((date, industry), 0.0)
        return result

    @staticmethod
    def _style_return_map(
        dataset: PortfolioDataset, style_columns: list[str]
    ) -> dict[tuple[str, str], float]:
        if not style_columns:
            return {}
        if dataset.style_returns is not None:
            frame = dataset.style_returns
            date_column = "date" if "date" in frame.columns else "TRADE_DT"
            return {
                (str(row[date_column]).replace("-", ""), style): float(row[style] or 0.0)
                for row in frame.iter_rows(named=True)
                for style in style_columns
                if style in row
            }
        joined = dataset.returns.join(dataset.styles, on=["date", "asset"], how="inner")
        result: dict[tuple[str, str], float] = {}
        for day in joined.partition_by("date", maintain_order=True):
            y = day["return"].fill_null(0).to_numpy()
            available = [style for style in style_columns if style in day.columns]
            if not available:
                continue
            x = np.column_stack([day[style].fill_null(0).to_numpy() for style in available])
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            for style, coefficient in zip(available, coefficients):
                result[(day["date"][0], style)] = float(coefficient)
        return result







