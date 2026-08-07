from __future__ import annotations

import numpy as np
import polars as pl

from factor_evaluator.risk import (
    FactorCovarianceMatrix,
    RiskMatrixProvider,
    estimate_factor_returns,
    ewma_factor_covariance,
    ewma_specific_risk,
)


def test_factor_returns_and_ewma_risk_are_point_in_time():
    dates = [f"202401{day:02d}" for day in range(1, 7)]
    assets = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    sizes = [-1.0, 0.0, 1.0, 0.0]
    betas = [0.0, -1.0, 0.0, 1.0]
    exposure_rows, return_rows = [], []
    for t, date in enumerate(dates):
        for i, asset in enumerate(assets):
            size = sizes[i]
            beta = betas[i]
            ret = 0.01 * size + 0.02 * beta
            exposure_rows.append({"date": date, "code": asset, "size": size, "beta": beta})
            return_rows.append({"date": date, "code": asset, "return": ret})
    factor_returns, specific_returns = estimate_factor_returns(
        pl.DataFrame(exposure_rows),
        pl.DataFrame(return_rows),
        ridge=0.0,
    )

    assert factor_returns.columns == ["date", "size", "beta"]
    np.testing.assert_allclose(factor_returns["size"].to_numpy(), np.full(6, 0.01), atol=1e-10)
    np.testing.assert_allclose(factor_returns["beta"].to_numpy(), np.full(6, 0.02), atol=1e-10)
    assert set(specific_returns.columns) == {"date", "asset", "specific_return"}

    cov = ewma_factor_covariance(factor_returns, halflife=2.0, min_periods=3)
    assert cov["date"].min() == "20240104"
    assert {"date", "factor_i", "factor_j", "covariance"} == set(cov.columns)
    first_cov_date = cov.filter(pl.col("date") == "20240104")
    assert first_cov_date.height == 4

    spec = ewma_specific_risk(specific_returns, halflife=2.0, min_periods=3)
    assert spec["date"].min() == "20240104"
    assert {"date", "asset", "specific_var", "specific_risk"} == set(spec.columns)



def test_risk_matrix_provider_aligns_requested_asset_order():
    exposures = pl.DataFrame(
        [
            {"date": "20240102", "code": "000001.SZ", "size": 1.0, "beta": 2.0},
            {"date": "20240102", "code": "000002.SZ", "size": 3.0, "beta": 4.0},
        ]
    )
    factor_covariance = pl.DataFrame(
        [
            {"date": "20240102", "factor_i": "size", "factor_j": "size", "covariance": 0.10},
            {"date": "20240102", "factor_i": "size", "factor_j": "beta", "covariance": 0.01},
            {"date": "20240102", "factor_i": "beta", "factor_j": "size", "covariance": 0.01},
            {"date": "20240102", "factor_i": "beta", "factor_j": "beta", "covariance": 0.20},
        ]
    )
    specific_risk = pl.DataFrame(
        [
            {"date": "20240102", "code": "000001.SZ", "specific_var": 0.001},
            {"date": "20240102", "code": "000002.SZ", "specific_var": 0.002},
        ]
    )
    provider = RiskMatrixProvider(exposures, factor_covariance, specific_risk, diagonal_jitter=0.0)
    sigma = provider("20240102", ["000002", "000001"])
    x = np.asarray([[3.0, 4.0], [1.0, 2.0]])
    f = np.asarray([[0.10, 0.01], [0.01, 0.20]])
    expected = x @ f @ x.T + np.diag([0.002, 0.001])
    np.testing.assert_allclose(sigma, expected)

def test_risk_matrix_provider_returns_lazy_factor_form_by_default():
    exposures = pl.DataFrame(
        [
            {"date": "20240102", "code": "000001.SZ", "size": 1.0, "beta": 2.0},
            {"date": "20240102", "code": "000002.SZ", "size": 3.0, "beta": 4.0},
        ]
    )
    factor_covariance = pl.DataFrame(
        [
            {"date": "20240102", "factor_i": "size", "factor_j": "size", "covariance": 0.10},
            {"date": "20240102", "factor_i": "size", "factor_j": "beta", "covariance": 0.01},
            {"date": "20240102", "factor_i": "beta", "factor_j": "size", "covariance": 0.01},
            {"date": "20240102", "factor_i": "beta", "factor_j": "beta", "covariance": 0.20},
        ]
    )
    specific_risk = pl.DataFrame(
        [
            {"date": "20240102", "code": "000001.SZ", "specific_var": 0.001},
            {"date": "20240102", "code": "000002.SZ", "specific_var": 0.002},
        ]
    )
    provider = RiskMatrixProvider(exposures, factor_covariance, specific_risk, diagonal_jitter=0.0)
    lazy = provider("20240102", ["000002", "000001"])
    dense = RiskMatrixProvider(
        exposures, factor_covariance, specific_risk, diagonal_jitter=0.0, dense=True
    )("20240102", ["000002", "000001"])
    assert isinstance(lazy, FactorCovarianceMatrix)
    vector = np.asarray([0.2, -0.1])
    np.testing.assert_allclose(lazy @ vector, dense @ vector)
    np.testing.assert_allclose(lazy.quadratic(vector), vector @ dense @ vector)
