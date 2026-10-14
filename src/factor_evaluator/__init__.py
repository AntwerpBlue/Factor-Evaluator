"""Point-in-time safe factor and portfolio backtesting."""

from .data import FactorDataset, PortfolioDataset
from .config import (
    BacktestFileConfig,
    load_config,
    load_factor_dataset,
    load_portfolio_dataset,
    load_target_weights,
)
from .factor import FactorBacktestConfig, FactorBacktestResult, FactorBacktester
from .evaluation import (
    FactorEvaluationConfig,
    FactorEvaluationResult,
    evaluate_factors,
)
from .risk import (
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
    IndexEnhancementConfig,
    IndexEnhancementStrategy,
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    PortfolioBacktester,
)

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
    "PortfolioBacktestConfig",
    "PortfolioBacktestResult",
    "PortfolioBacktester",
    "ExposureMatrixProvider",
    "FactorCovarianceMatrix",
    "RiskMatrixProvider",
    "RiskModelEstimate",
    "estimate_factor_returns",
    "estimate_ewma_risk_model",
    "ewma_factor_covariance",
    "ewma_specific_risk",
]

