from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

from .data import normalize_panel


@dataclass(frozen=True)
class RiskModelEstimate:
    factor_returns: pl.DataFrame
    specific_returns: pl.DataFrame
    factor_covariance: pl.DataFrame
    specific_risk: pl.DataFrame


@dataclass(frozen=True)
class FactorCovarianceMatrix:
    """Lazy stock covariance ``X F X' + D`` in a fixed asset order."""

    exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    diagonal_jitter: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        n_assets = self.exposures.shape[0]
        return n_assets, n_assets

    def matvec(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        diagonal = self.specific_variance + self.diagonal_jitter
        return self.exposures @ (self.factor_covariance @ (self.exposures.T @ vector)) + diagonal * vector

    def quadratic(self, value: np.ndarray) -> float:
        vector = np.asarray(value, dtype=float)
        factor_exposure = self.exposures.T @ vector
        diagonal = self.specific_variance + self.diagonal_jitter
        return float(factor_exposure @ self.factor_covariance @ factor_exposure + np.sum(diagonal * vector * vector))

    def max_eigenvalue_bound(self) -> float:
        factor_norm = float(np.linalg.norm(self.factor_covariance, ord=2)) if self.factor_covariance.size else 0.0
        exposure_norm_bound = float(np.sum(self.exposures * self.exposures))
        diagonal_max = float(np.max(self.specific_variance + self.diagonal_jitter)) if self.specific_variance.size else 0.0
        return max(factor_norm * exposure_norm_bound + diagonal_max, diagonal_max)

    def to_numpy(self) -> np.ndarray:
        matrix = self.exposures @ self.factor_covariance @ self.exposures.T
        matrix = (matrix + matrix.T) / 2.0
        matrix += np.diag(self.specific_variance + self.diagonal_jitter)
        return matrix

    def __matmul__(self, value: np.ndarray) -> np.ndarray:
        return self.matvec(value)

    def __rmatmul__(self, value: np.ndarray) -> np.ndarray:
        return self.matvec(value)

    def __array__(self, dtype=None) -> np.ndarray:
        matrix = self.to_numpy()
        return matrix.astype(dtype) if dtype is not None else matrix


def infer_factor_columns(exposures: pl.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    excluded = {"date", "asset", *exclude}
    return [column for column in exposures.columns if column not in excluded]


def estimate_factor_returns(
    exposures: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    factor_columns: list[str] | None = None,
    weights: pl.DataFrame | None = None,
    weight_column: str = "weight",
    ridge: float = 1e-8,
    min_assets: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate daily factor returns and specific returns by WLS.

    ``exposures`` and ``returns`` are keyed by the same date. For a point-in-time
    workflow, ``returns`` should already be forward returns for that date.
    """
    exposure_panel = normalize_panel(exposures)
    return_panel = normalize_panel(returns)
    if "return" not in return_panel.columns:
        value_columns = [c for c in return_panel.columns if c not in {"date", "asset"}]
        if len(value_columns) != 1:
            raise ValueError("returns must contain one value column named 'return'")
        return_panel = return_panel.rename({value_columns[0]: "return"})
    factors = factor_columns or infer_factor_columns(exposure_panel)
    if not factors:
        raise ValueError("No exposure factor columns found")
    missing = [factor for factor in factors if factor not in exposure_panel.columns]
    if missing:
        raise ValueError(f"Missing exposure columns: {missing}")

    joined = exposure_panel.select("date", "asset", *factors).join(
        return_panel.select("date", "asset", pl.col("return").cast(pl.Float64)),
        on=["date", "asset"],
        how="inner",
    )
    if weights is not None:
        weight_panel = normalize_panel(weights)
        if weight_column not in weight_panel.columns:
            raise ValueError(f"weights missing column '{weight_column}'")
        joined = joined.join(
            weight_panel.select("date", "asset", pl.col(weight_column).cast(pl.Float64).alias("_w")),
            on=["date", "asset"],
            how="left",
        )
    else:
        joined = joined.with_columns(pl.lit(1.0).alias("_w"))

    factor_rows: list[dict[str, float | str]] = []
    specific_rows: list[dict[str, float | str]] = []
    required_assets = min_assets or len(factors) + 1
    for day in joined.partition_by("date", maintain_order=True):
        clean = day.drop_nulls(["return", "_w", *factors]).filter(pl.col("_w") > 0)
        if clean.height < required_assets:
            continue
        date = clean["date"][0]
        x = np.column_stack([clean[factor].to_numpy() for factor in factors]).astype(float)
        y = clean["return"].to_numpy().astype(float)
        w = clean["_w"].to_numpy().astype(float)
        sw = np.sqrt(w / np.nanmean(w))
        xw = x * sw[:, None]
        yw = y * sw
        lhs = xw.T @ xw + ridge * np.eye(len(factors))
        rhs = xw.T @ yw
        coef = np.linalg.solve(lhs, rhs)
        residual = y - x @ coef
        factor_rows.append({"date": date, **{factor: float(value) for factor, value in zip(factors, coef)}})
        specific_rows.extend(
            {"date": date, "asset": asset, "specific_return": float(value)}
            for asset, value in zip(clean["asset"].to_list(), residual)
        )
    return pl.DataFrame(factor_rows), pl.DataFrame(specific_rows)


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(n - 1, -1, -1, dtype=float)
    return weights / weights.sum()


def ewma_factor_covariance(
    factor_returns: pl.DataFrame,
    *,
    halflife: float = 63.0,
    min_periods: int = 60,
    include_current: bool = False,
) -> pl.DataFrame:
    """Estimate a daily long-form EWMA covariance matrix of factor returns."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    frame = factor_returns.with_columns(pl.col("date").cast(pl.String)).sort("date")
    factors = [column for column in frame.columns if column != "date"]
    if not factors:
        raise ValueError("factor_returns must contain factor columns")
    rows: list[dict[str, float | str]] = []
    dates = frame["date"].to_list()
    values = frame.select(factors).to_numpy().astype(float)
    for index, date in enumerate(dates):
        end = index + 1 if include_current else index
        start = 0
        window = values[start:end]
        valid = np.isfinite(window).all(axis=1)
        window = window[valid]
        if window.shape[0] < min_periods:
            continue
        weights = _ewma_weights(window.shape[0], halflife)
        mean = weights @ window
        centered = window - mean
        cov = (centered * weights[:, None]).T @ centered
        for i, factor_i in enumerate(factors):
            for j, factor_j in enumerate(factors):
                rows.append(
                    {
                        "date": date,
                        "factor_i": factor_i,
                        "factor_j": factor_j,
                        "covariance": float(cov[i, j]),
                    }
                )
    return pl.DataFrame(rows)


def ewma_specific_risk(
    specific_returns: pl.DataFrame,
    *,
    halflife: float = 63.0,
    min_periods: int = 60,
    include_current: bool = False,
) -> pl.DataFrame:
    """Estimate daily EWMA specific variance/risk for each asset."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    frame = normalize_panel(specific_returns)
    if "specific_return" not in frame.columns:
        value_columns = [c for c in frame.columns if c not in {"date", "asset"}]
        if len(value_columns) != 1:
            raise ValueError("specific_returns must contain one specific return column")
        frame = frame.rename({value_columns[0]: "specific_return"})
    rows: list[dict[str, float | str]] = []
    for asset_frame in frame.sort(["asset", "date"]).partition_by("asset", maintain_order=True):
        asset = asset_frame["asset"][0]
        dates = asset_frame["date"].to_list()
        values = asset_frame["specific_return"].to_numpy().astype(float)
        for index, date in enumerate(dates):
            end = index + 1 if include_current else index
            window = values[:end]
            window = window[np.isfinite(window)]
            if window.shape[0] < min_periods:
                continue
            weights = _ewma_weights(window.shape[0], halflife)
            mean = float(weights @ window)
            variance = float(weights @ ((window - mean) ** 2))
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "specific_var": variance,
                    "specific_risk": float(np.sqrt(max(variance, 0.0))),
                }
            )
    return pl.DataFrame(rows)


class ExposureMatrixProvider:
    """Return exposure matrices in the optimizer's asset order."""

    def __init__(
        self,
        exposures: pl.DataFrame,
        *,
        exposure_columns: list[str] | None = None,
        standardize: bool = True,
    ) -> None:
        panel = normalize_panel(exposures)
        self.exposure_columns = exposure_columns or infer_factor_columns(panel)
        if not self.exposure_columns:
            raise ValueError("ExposureMatrixProvider requires exposure columns")
        missing = [column for column in self.exposure_columns if column not in panel.columns]
        if missing:
            raise ValueError(f"Missing exposure columns: {missing}")
        if standardize:
            panel = self._standardize_by_date(panel, self.exposure_columns)
        self.exposures_by_date = {
            day["date"][0]: day.select("asset", *self.exposure_columns)
            for day in panel.partition_by("date", maintain_order=True)
        }

    def __call__(self, date: str, assets: list[str]) -> np.ndarray | None:
        frame = self.exposures_by_date.get(date)
        if frame is None:
            return None
        ordered = pl.DataFrame({"asset": assets}).join(frame, on="asset", how="left")
        return (
            ordered.select([
                pl.col(column).cast(pl.Float64).fill_null(0.0)
                for column in self.exposure_columns
            ])
            .to_numpy()
            .astype(float)
        )

    @staticmethod
    def _standardize_by_date(panel: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        expressions = []
        for column in columns:
            mean = pl.col(column).mean().over("date")
            std = pl.col(column).std().over("date")
            expressions.append(
                pl.when(std > 0)
                .then((pl.col(column) - mean) / std)
                .otherwise(0.0)
                .alias(column)
            )
        return panel.with_columns(expressions)


class RiskMatrixProvider:
    """Build stock covariance matrices on demand from factor risk data.

    For a requested date and ordered asset list, returns ``Sigma = X F X' + D``
    in exactly that asset order. Missing exposures are filled with zero, and
    missing specific variances fall back to the date median or ``specific_var_floor``.
    """

    def __init__(
        self,
        exposures: pl.DataFrame,
        factor_covariance: pl.DataFrame,
        specific_risk: pl.DataFrame,
        *,
        factor_columns: list[str] | None = None,
        specific_var_floor: float = 1e-8,
        diagonal_jitter: float = 1e-10,
        dense: bool = False,
    ) -> None:
        self.exposures = normalize_panel(exposures)
        self.factor_columns = factor_columns or infer_factor_columns(self.exposures)
        self.specific_var_floor = specific_var_floor
        self.diagonal_jitter = diagonal_jitter
        self.dense = dense
        if not self.factor_columns:
            raise ValueError("RiskMatrixProvider requires at least one exposure column")
        missing = [column for column in self.factor_columns if column not in self.exposures.columns]
        if missing:
            raise ValueError(f"Missing exposure columns: {missing}")
        self.exposures_by_date = {
            day["date"][0]: day.select("asset", *self.factor_columns)
            for day in self.exposures.partition_by("date", maintain_order=True)
        }
        self.covariance_by_date = self._build_covariance_by_date(factor_covariance)
        self.specific_by_date = self._build_specific_by_date(specific_risk)

    def __call__(self, date: str, assets: list[str]) -> FactorCovarianceMatrix | np.ndarray | None:
        exposure_frame = self.exposures_by_date.get(date)
        covariance = self.covariance_by_date.get(date)
        if exposure_frame is None or covariance is None:
            return None
        factor_index, factor_covariance = covariance
        factors = [factor for factor in self.factor_columns if factor in factor_index]
        if not factors:
            return None
        idx = [factor_index[factor] for factor in factors]
        f_matrix = factor_covariance[np.ix_(idx, idx)]
        asset_frame = pl.DataFrame({"asset": assets})
        x = (
            asset_frame.join(exposure_frame.select("asset", *factors), on="asset", how="left")
            .select([pl.col(factor).cast(pl.Float64).fill_null(0.0) for factor in factors])
            .to_numpy()
            .astype(float)
        )
        specific = self.specific_by_date.get(date, {})
        fallback = self._specific_fallback(specific)
        diagonal = np.asarray(
            [max(float(specific.get(asset, fallback)), self.specific_var_floor) for asset in assets],
            dtype=float,
        )
        factor_matrix = FactorCovarianceMatrix(
            exposures=x,
            factor_covariance=f_matrix,
            specific_variance=diagonal,
            diagonal_jitter=self.diagonal_jitter,
        )
        return factor_matrix.to_numpy() if self.dense else factor_matrix

    def _specific_fallback(self, specific: dict[str, float]) -> float:
        values = np.asarray([value for value in specific.values() if np.isfinite(value) and value > 0], dtype=float)
        if values.size == 0:
            return self.specific_var_floor
        return max(float(np.median(values)), self.specific_var_floor)

    @staticmethod
    def _build_covariance_by_date(
        frame: pl.DataFrame,
    ) -> dict[str, tuple[dict[str, int], np.ndarray]]:
        required = {"date", "factor_i", "factor_j", "covariance"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"factor_covariance missing columns: {sorted(missing)}")
        result: dict[str, tuple[dict[str, int], np.ndarray]] = {}
        clean = frame.with_columns(
            pl.col("date").cast(pl.String),
            pl.col("factor_i").cast(pl.String),
            pl.col("factor_j").cast(pl.String),
            pl.col("covariance").cast(pl.Float64),
        )
        for day in clean.partition_by("date", maintain_order=True):
            factors = sorted(set(day["factor_i"].to_list()) | set(day["factor_j"].to_list()))
            factor_index = {factor: index for index, factor in enumerate(factors)}
            matrix = np.zeros((len(factors), len(factors)), dtype=float)
            for row in day.iter_rows(named=True):
                i = factor_index[row["factor_i"]]
                j = factor_index[row["factor_j"]]
                value = float(row["covariance"] or 0.0)
                matrix[i, j] = value
                matrix[j, i] = value
            result[day["date"][0]] = (factor_index, matrix)
        return result

    @staticmethod
    def _build_specific_by_date(frame: pl.DataFrame) -> dict[str, dict[str, float]]:
        panel = normalize_panel(frame)
        if "specific_var" in panel.columns:
            value_expr = pl.col("specific_var").cast(pl.Float64).alias("specific_var")
        elif "specific_risk" in panel.columns:
            value_expr = (pl.col("specific_risk").cast(pl.Float64) ** 2).alias("specific_var")
        else:
            value_columns = [column for column in panel.columns if column not in {"date", "asset"}]
            if len(value_columns) != 1:
                raise ValueError("specific_risk must contain specific_var or specific_risk")
            value_expr = pl.col(value_columns[0]).cast(pl.Float64).alias("specific_var")
        slim = panel.select("date", "asset", value_expr)
        return {
            day["date"][0]: dict(zip(day["asset"].to_list(), day["specific_var"].to_list()))
            for day in slim.partition_by("date", maintain_order=True)
        }


def estimate_ewma_risk_model(
    exposures: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    factor_columns: list[str] | None = None,
    weights: pl.DataFrame | None = None,
    weight_column: str = "weight",
    halflife: float = 63.0,
    min_periods: int = 60,
    include_current: bool = False,
) -> RiskModelEstimate:
    factor_returns, specific_returns = estimate_factor_returns(
        exposures,
        returns,
        factor_columns=factor_columns,
        weights=weights,
        weight_column=weight_column,
    )
    factor_covariance = ewma_factor_covariance(
        factor_returns,
        halflife=halflife,
        min_periods=min_periods,
        include_current=include_current,
    )
    specific_risk = ewma_specific_risk(
        specific_returns,
        halflife=halflife,
        min_periods=min_periods,
        include_current=include_current,
    )
    return RiskModelEstimate(
        factor_returns=factor_returns,
        specific_returns=specific_returns,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
    )
