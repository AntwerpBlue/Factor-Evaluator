from __future__ import annotations

import numpy as np


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def safe_std(values: np.ndarray, ddof: int = 1) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=ddof)) if finite.size > ddof else float("nan")


def annualized_return(returns: np.ndarray, periods: int = 252) -> float:
    return safe_mean(returns) * periods


def annualized_volatility(returns: np.ndarray, periods: int = 252) -> float:
    return safe_std(returns) * np.sqrt(periods)


def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    volatility = annualized_volatility(returns, periods)
    return annualized_return(returns, periods) / volatility if volatility > 0 else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    clean = np.nan_to_num(returns, nan=0.0)
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peaks - 1.0)) if clean.size else float("nan")


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=float)
    ranked[order] = np.arange(1, values.size + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranked)
        ranked = sums[inverse] / counts[inverse]
    return ranked


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    x = left[mask]
    y = right[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    return correlation(rank(left[mask]), rank(right[mask]))
