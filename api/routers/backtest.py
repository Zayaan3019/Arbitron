"""Backtest API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from ..schemas import BacktestRequest, BacktestResponse, BacktestMetricsResponse

router = APIRouter()
CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """
    Run a walk-forward factor backtest and return cumulative returns + metrics.
    """
    from arbitron.data import MarketData, SECTOR_UNIVERSES, DEFAULT_UNIVERSE
    from arbitron.factors import compute_factor
    from arbitron.backtest import BacktestConfig, run_factor_backtest

    if req.universe == "SP500":
        tickers = DEFAULT_UNIVERSE
    elif req.universe in SECTOR_UNIVERSES:
        tickers = SECTOR_UNIVERSES[req.universe]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown universe: {req.universe}")

    try:
        md     = MarketData(cache_dir=CACHE_DIR)
        prices = md.get_prices(tickers, req.start_date, req.end_date)
        factor = compute_factor(req.factor_name, prices)

        cfg = BacktestConfig(
            n_quantiles=req.n_quantiles,
            rebal_freq=req.rebal_freq,
            transaction_cost=req.transaction_cost,
            long_only=req.long_only,
        )
        result = run_factor_backtest(factor, prices, cfg, factor_name=req.factor_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    import numpy as np
    m = result.metrics
    # total_return and ann_volatility not stored on BacktestMetrics — derive them
    cum_end   = float(result.cumulative.dropna().iloc[-1]) if not result.cumulative.dropna().empty else 1.0
    total_ret = round(cum_end - 1.0, 4)
    ann_vol   = round(float(result.returns.std() * np.sqrt(252)), 4)
    metrics = BacktestMetricsResponse(
        factor_name=req.factor_name,
        cagr=round(float(m.cagr), 4),
        sharpe=round(float(m.sharpe), 4),
        sortino=round(float(m.sortino), 4),
        calmar=round(float(m.calmar), 4),
        max_drawdown=round(float(m.max_drawdown), 4),
        win_rate=round(float(m.win_rate), 4),
        total_return=total_ret,
        ann_volatility=ann_vol,
        skewness=round(float(m.skewness), 4),
        kurtosis=round(float(m.kurtosis), 4),
        n_periods=int(m.n_days),
    )

    cum = result.cumulative.fillna(1.0)
    dd  = result.drawdown.fillna(0.0)

    return BacktestResponse(
        metrics=metrics,
        dates=[d.strftime("%Y-%m-%d") for d in cum.index],
        cumulative_returns=cum.tolist(),
        drawdown=dd.tolist(),
    )
