"""Externally updated rank/return calibration; prediction only reads state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Literal

import numpy as np
from scipy.stats import rankdata

from .risk import decision_timestamp


def signal_ranks(signal: Mapping[str, float]) -> dict[str, float]:
    """Average ties and map the observed cross-section to percentile ranks."""
    assets = list(signal)
    ranks = (rankdata(list(signal.values()), method="average") - 0.5) / len(assets)
    return dict(zip(assets, map(float, ranks)))


@dataclass(frozen=True)
class CalibrationState:
    state_id: str
    model_id: str
    version: int
    basis: Literal["raw", "barra_neutral"]
    horizon: str
    updated_at: str | None
    prior_means: list[float]
    prior_std: float
    observation_std: float
    halflife_days: float
    smoothing: float
    counts: list[float]
    sums: list[float]
    means: list[float]
    uncertainty: list[float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> CalibrationState:
        return cls(**value)


class RankReturnCalibrator:
    def initial_state(
        self, *, basis: Literal["raw", "barra_neutral"] = "raw", horizon: str = "1d",
        model_id: str = "rank-return", buckets: int = 20, prior_endpoint_bps: float = 1.2,
        prior_std: float = 0.0002, observation_std: float = 0.02,
        halflife_days: float = 63.0, smoothing: float = 0.25,
    ) -> CalibrationState:
        centers = (np.arange(buckets) + 0.5) / buckets
        means = prior_endpoint_bps / 10_000 * (2 * centers - 1)
        return CalibrationState(
            f"{model_id}:0", model_id, 0, basis, horizon, None, means.tolist(),
            prior_std, observation_std, halflife_days, smoothing,
            [0.0] * buckets, [0.0] * buckets, means.tolist(), [prior_std] * buckets,
        )

    def predict(self, ranks: Mapping[str, float], state: CalibrationState, *, return_uncertainty=False) -> dict:
        n = len(state.means)
        buckets = {asset: min(n - 1, max(0, int(rank * n))) for asset, rank in ranks.items()}
        mu = {asset: state.means[bucket] for asset, bucket in buckets.items()}
        if return_uncertainty:
            return {"mu": mu, "uncertainty": {asset: state.uncertainty[bucket] for asset, bucket in buckets.items()}}
        return mu

    def update(self, state: CalibrationState, realized_samples) -> CalibrationState:
        """Return a new state from realized rows; no prices or future returns are read.

        Rows: rank in [0,1], realized_at, available_at (optional), raw_return,
        and barra_neutral_return when that basis is selected. available_at denotes
        when the caller actually obtained the realized outcome.
        """
        rows = sorted(realized_samples, key=lambda row: decision_timestamp(row.get("available_at", row["realized_at"])))
        if not rows:
            return state
        counts, sums = np.array(state.counts), np.array(state.sums)
        last = decision_timestamp(state.updated_at) if state.updated_at else None
        for row in rows:
            now = decision_timestamp(row.get("available_at", row["realized_at"]))
            if decision_timestamp(row["realized_at"]) > now or (last is not None and now < last):
                raise ValueError("Calibration update requires realized, chronologically available samples")
            if last is not None:
                decay = 0.5 ** ((now - last).total_seconds() / 86400 / state.halflife_days)
                counts *= decay
                sums *= decay
            bucket = min(len(counts) - 1, max(0, int(row["rank"] * len(counts))))
            counts[bucket] += 1
            sums[bucket] += row[f"{state.basis}_return"]
            last = now
        precision = 1 / state.prior_std**2 + counts / state.observation_std**2
        means = (np.array(state.prior_means) / state.prior_std**2 + sums / state.observation_std**2) / precision
        # Fixed local averaging of the posterior mean, not fitted smoothing.
        n, strength = len(counts), state.smoothing
        smoother = np.eye(n) * (1 - 2 * strength)
        smoother += np.eye(n, k=1) * strength + np.eye(n, k=-1) * strength
        smoother[0, 0] += strength
        smoother[-1, -1] += strength
        payload = state.to_dict()
        payload.update(
            version=state.version + 1, state_id=f"{state.model_id}:{state.version + 1}",
            updated_at=last.isoformat(), counts=counts.tolist(), sums=sums.tolist(),
            means=(smoother @ means).tolist(),
            uncertainty=np.sqrt((smoother**2) @ (1 / precision)).tolist(),
        )
        return CalibrationState.from_dict(payload)
