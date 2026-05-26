"""Portfolio optimization API router."""

from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from ..schemas import (
    PortfolioRequest, WeightsResponse, EfficientFrontierResponse,
)

router = APIRouter()
CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


def _resolve_tickers(req: PortfolioRequest) -> list[str]:
    from arbitron.data import SECTOR_UNIVERSES, DEFAULT_UNIVERSE
    if req.tickers:
        return req.tickers
    if req.universe == "SP500":
        return DEFAULT_UNIVERSE
    if req.universe in SECTOR_UNIVERSES:
        return SECTOR_UNIVERSES[req.universe]
    raise HTTPException(status_code=400, detail=f"Unknown universe: {req.universe}")


@router.post("/portfolio/optimize", response_model=WeightsResponse)
async def optimize_portfolio(req: PortfolioRequest):
    """
    Optimise a portfolio and return weights with expected performance.
    """
    import pandas as pd
    from arbitron.data import MarketData
    from arbitron.portfolio import (
        ledoit_wolf_covariance,
        min_variance_portfolio,
        max_sharpe_portfolio,
        risk_parity_portfolio,
        portfolio_performance,
        diversification_ratio,
    )

    tickers = _resolve_tickers(req)

    try:
        md      = MarketData(cache_dir=CACHE_DIR)
        prices  = md.get_prices(tickers, req.start_date, req.end_date)
        returns = prices.pct_change().dropna()

        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252

        if req.optimizer == "min_variance":
            weights = min_variance_portfolio(cov, max_weight=req.max_weight)
        elif req.optimizer == "risk_parity":
            weights = risk_parity_portfolio(cov)
        else:  # max_sharpe
            weights = max_sharpe_portfolio(mu, cov, rf=req.rf, max_weight=req.max_weight)

        perf = portfolio_performance(weights, mu, cov, rf=req.rf)
        dr   = diversification_ratio(weights, cov)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    tickers_out = list(weights.index)
    return WeightsResponse(
        tickers=tickers_out,
        weights=[round(float(w), 6) for w in weights],
        expected_return=perf["expected_return"],
        volatility=perf["volatility"],
        sharpe=perf["sharpe"],
        diversification_ratio=round(float(dr), 4),
    )


@router.post("/portfolio/frontier", response_model=EfficientFrontierResponse)
async def efficient_frontier(req: PortfolioRequest):
    """Compute the efficient frontier (20 points)."""
    from arbitron.data import MarketData
    from arbitron.portfolio import ledoit_wolf_covariance, mean_variance_frontier

    tickers = _resolve_tickers(req)

    try:
        md      = MarketData(cache_dir=CACHE_DIR)
        prices  = md.get_prices(tickers, req.start_date, req.end_date)
        returns = prices.pct_change().dropna()
        cov     = ledoit_wolf_covariance(returns)
        mu      = returns.mean() * 252
        frontier = mean_variance_frontier(mu, cov, n_points=25)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return EfficientFrontierResponse(
        returns=frontier["return"].tolist(),
        volatilities=frontier["volatility"].tolist(),
        sharpes=frontier["sharpe"].tolist(),
    )
