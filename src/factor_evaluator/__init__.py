"""Point-in-time safe factor and portfolio backtesting."""

from .data import FactorDataset, PortfolioDataset
from .config import (
    BacktestFileConfig,
    load_config,
    load_factor_dataset,
    load_portfolio_dataset,
    load_target_weights,
)
from importlib import import_module
from .risk import (
    BarraSnapshot,
    ExposureMatrixProvider,
    FactorCovarianceMatrix,
    RiskMatrixProvider,
    RiskModelEstimate,
    estimate_factor_returns,
    estimate_ewma_risk_model,
    ewma_factor_covariance,
    ewma_specific_risk,
)
from .portfolio import (
    AssetConstraints,
    IndexEnhancementConfig,
    IndexEnhancementStrategy,
    OptimizationResult,
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    PortfolioBacktester,
)
from .alpha_calibration import CalibrationState, RankReturnCalibrator


def __getattr__(name):
    # Standalone portfolio optimization does not need plotting/report imports.
    module = {
        "FactorBacktestConfig": ".factor", "FactorBacktestResult": ".factor",
        "FactorBacktester": ".factor", "FactorEvaluationConfig": ".evaluation",
        "FactorEvaluationResult": ".evaluation", "evaluate_factors": ".evaluation",
    }.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "FactorDataset",
    "PortfolioDataset",
    "BacktestFileConfig",
    "load_config",
    "load_factor_dataset",
    "load_portfolio_dataset",
    "load_target_weights",
    "FactorBacktestConfig",
    "FactorBacktestResult",
    "FactorBacktester",
    "FactorEvaluationConfig",
    "FactorEvaluationResult",
    "evaluate_factors",
    "IndexEnhancementConfig",
    "IndexEnhancementStrategy",
    "OptimizationResult",
    "CalibrationState",
    "RankReturnCalibrator",
    "AssetConstraints",
    "PortfolioBacktestConfig",
    "PortfolioBacktestResult",
    "PortfolioBacktester",
    "ExposureMatrixProvider",
    "BarraSnapshot",
    "FactorCovarianceMatrix",
    "RiskMatrixProvider",
    "RiskModelEstimate",
    "estimate_factor_returns",
    "estimate_ewma_risk_model",
    "ewma_factor_covariance",
    "ewma_specific_risk",
]

