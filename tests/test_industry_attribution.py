from __future__ import annotations

import numpy as np
import polars as pl

from factor_evaluator import PortfolioBacktester, PortfolioDataset


def test_industry_only_attribution_is_used():
    dates = ["20250102", "20250103", "20250106"]
    codes = ["000001.SZ", "000002.SZ"]
    returns = pl.DataFrame(
        [
            {"date": date, "code": code, "return": value}
            for date, pair in zip(dates, ((0.0, 0.0), (0.02, -0.01), (0.01, -0.005)))
            for code, value in zip(codes, pair)
        ]
    )
    benchmark = pl.DataFrame(
        {"date": dates, "benchmark_return": [0.0, 0.005, 0.0025]}
    )
    weights = pl.DataFrame(
        [
            {"date": date, "code": code, "weight": weight}
            for date in dates
            for code, weight in zip(codes, (0.8, 0.2))
        ]
    )
    industries = pl.DataFrame(
        [
            {"date": date, "code": code, "industry": industry}
            for date in dates
            for code, industry in zip(codes, ("Bank", "Tech"))
        ]
    )
    industry_returns = pl.DataFrame(
        {
            "date": dates,
            "Bank": [0.0, 0.015, 0.008],
            "Tech": [0.0, -0.005, -0.003],
        }
    )
    result = PortfolioBacktester().run(
        weights,
        PortfolioDataset(
            returns=returns,
            benchmark=benchmark,
            industry_exposures=industries,
            industry_returns=industry_returns,
        ),
    )
    decomposition = result.decomposition
    assert decomposition is not None
    assert np.any(np.abs(decomposition["IndustryRet"].to_numpy()) > 0)
    np.testing.assert_allclose(
        decomposition["SurplusRet"].to_numpy(),
        (
            decomposition["StyleRet"]
            + decomposition["IndustryRet"]
            + decomposition["pureAlpha_after_fee"]
        ).to_numpy(),
    )
