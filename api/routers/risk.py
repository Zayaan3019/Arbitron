"""Risk analytics API router."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import RiskRequest, RiskResponse

router = APIRouter()
CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


@router.post("/risk", response_model=RiskResponse)
async def compute_risk(req: RiskRequest):
    """
    Compute risk metrics for a portfolio (or equal-weight if no weights given).
    """
    from arbitron.data import MarketData, SECTOR_UNIVERSES, DEFAULT_UNIVERSE
    from arbitron.risk import full_risk_report
    from arbitron.models import build_regime_detector

    # Resolve tickers
    if req.tickers:
        tickers = req.tickers
    elif req.universe == "SP500":
        tickers = DEFAULT_UNIVERSE
    elif req.universe in SECTOR_UNIVERSES:
        tickers = SECTOR_UNIVERSES[req.universe]
    else:
        raise HTTPException(status_code=400, detail="Provide tickers or universe")

    try:
        md      = MarketData(cache_dir=CACHE_DIR)
        prices  = md.get_prices(tickers, req.start_date, req.end_date)
        returns = prices.pct_change().dropna()

        # Portfolio weights
        if req.weights:
            w = pd.Series(req.weights).reindex(returns.columns).fillna(0)
            w = w / w.sum()
        else:
            w = pd.Series(1 / len(returns.columns), index=returns.columns)

        port_ret = (returns * w).sum(axis=1)

        # Market proxy = equal-weight
        market_ret = returns.mean(axis=1)

        report = full_risk_report(port_ret, market_ret=market_ret)

        # Regime
        rd = build_regime_detector(use_hmm=False)
        rd.fit(market_ret)
        regime = rd.current_regime(market_ret)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return RiskResponse(
        var_95=report.get("var_95", 0.0),
        var_99=report.get("var_99", 0.0),
        cvar_95=report.get("cvar_95", 0.0),
        cvar_99=report.get("cvar_99", 0.0),
        ulcer_index=report.get("ulcer_index", 0.0),
        tail_ratio=report.get("tail_ratio", 1.0),
        max_drawdown_duration_days=report.get("max_duration_days", 0),
        avg_drawdown_duration_days=report.get("avg_duration_days", 0),
        n_drawdown_periods=report.get("n_drawdown_periods", 0),
        market_beta=report.get("market_beta"),
        regime=regime,
    )
