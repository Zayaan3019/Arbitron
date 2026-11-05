"""
Portfolio optimization API router.

Sync `def` handlers on purpose — see the note in `routers/data.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import (
    EfficientFrontierResponse,
    PortfolioRequest,
    WeightsResponse,
)
from ..serialization import safe_float

router = APIRouter()

#: Minimum usable return observations before a covariance estimate is
#: anything other than noise.
MIN_OBSERVATIONS = 60


def _returns_for(req: PortfolioRequest):
    """Prices → complete-case daily returns, with an honest sample-size gate."""
    from arbitron.data import to_returns

    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

    # Names never quoted in the window (pre-IPO for the whole period) are
    # dropped before `dropna()`, otherwise a single such column would empty
    # the entire panel.
    prices = prices.loc[:, prices.notna().sum() > MIN_OBSERVATIONS]
    if prices.empty:
        raise HTTPException(
            status_code=422,
            detail=f"No ticker has more than {MIN_OBSERVATIONS} observations in this window.",
        )

    returns = to_returns(prices).dropna()
    if len(returns) < MIN_OBSERVATIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(returns)} complete observations after aligning "
                f"{prices.shape[1]} tickers; need at least {MIN_OBSERVATIONS} "
                "for a usable covariance estimate."
            ),
        )
    return returns


@router.post("/portfolio/optimize", response_model=WeightsResponse)
def optimize_portfolio(req: PortfolioRequest) -> WeightsResponse:
    """Optimise a portfolio and return weights with expected performance."""
    from arbitron.portfolio import (
        diversification_ratio,
        ledoit_wolf_covariance,
        max_sharpe_portfolio,
        min_variance_portfolio,
        portfolio_performance,
        risk_parity_portfolio,
    )

    returns = _returns_for(req)
    cov = ledoit_wolf_covariance(returns)
    mu = returns.mean() * 252

    if req.optimizer == "min_variance":
        weights = min_variance_portfolio(
            cov, long_only=req.long_only, max_weight=req.max_weight
        )
    elif req.optimizer == "risk_parity":
        # ERC is only defined for strictly positive weights.
        weights = risk_parity_portfolio(cov)
    else:  # max_sharpe
        weights = max_sharpe_portfolio(
            mu, cov, rf=req.rf, long_only=req.long_only, max_weight=req.max_weight
        )

    perf = portfolio_performance(weights, mu, cov, rf=req.rf)
    dr = diversification_ratio(weights, cov)

    return WeightsResponse(
        tickers=[str(t) for t in weights.index],
        weights=[round(float(w), 6) for w in weights],
        weight_sum=round(float(weights.sum()), 10),
        expected_return=safe_float(perf["expected_return"]),
        volatility=safe_float(perf["volatility"]),
        sharpe=safe_float(perf["sharpe"]),
        diversification_ratio=safe_float(dr),
        n_observations=len(returns),
        data_caveats=data_caveats(req.universe, req.tickers),
    )


@router.post("/portfolio/frontier", response_model=EfficientFrontierResponse)
def efficient_frontier(req: PortfolioRequest) -> EfficientFrontierResponse:
    """Compute the mean-variance efficient frontier."""
    from arbitron.portfolio import ledoit_wolf_covariance, mean_variance_frontier

    returns = _returns_for(req)
    cov = ledoit_wolf_covariance(returns)
    mu = returns.mean() * 252
    frontier = mean_variance_frontier(mu, cov, n_points=25, long_only=req.long_only)

    if frontier.empty:
        raise HTTPException(
            status_code=422,
            detail="Frontier optimisation did not converge for any target return.",
        )

    return EfficientFrontierResponse(
        returns=[safe_float(v) for v in frontier["return"].tolist()],
        volatilities=[safe_float(v) for v in frontier["volatility"].tolist()],
        sharpes=[safe_float(v) for v in frontier["sharpe"].tolist()],
        data_caveats=data_caveats(req.universe, req.tickers),
    )
