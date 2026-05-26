"""Factor library API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from ..schemas import (
    FactorRequest, FactorEvalRequest, ICStatsResponse, FactorListResponse,
)

router = APIRouter()
CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


@router.get("/factors", response_model=FactorListResponse)
async def list_factors():
    """List all available factor names."""
    from arbitron.factors import FACTOR_REGISTRY
    return FactorListResponse(factors=list(FACTOR_REGISTRY.keys()))


@router.post("/factors/evaluate", response_model=ICStatsResponse)
async def evaluate_factor(req: FactorEvalRequest):
    """
    Evaluate a single factor via IC statistics and Fama-MacBeth regression.
    """
    from arbitron.data import MarketData, SECTOR_UNIVERSES, DEFAULT_UNIVERSE
    from arbitron.factors import compute_factor, factor_summary

    # Resolve tickers
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
        stats  = factor_summary(req.factor_name, factor, prices, req.horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    def _f(key, default=0.0):
        v = stats.get(key, default)
        return float(v) if v is not None else default

    return ICStatsResponse(
        factor_name=req.factor_name,
        ic_mean=round(_f("ic_ic_mean"), 4),
        ic_std=round(_f("ic_ic_std"), 4),
        ic_ir=round(_f("ic_ic_ir"), 4),
        t_stat=round(_f("ic_t_stat"), 4),
        pct_positive=round(_f("ic_pct_pos", 0.5), 4),
        n_periods=int(stats.get("ic_n_periods", 0)),
        fama_macbeth_gamma=round(_f("fm_gamma"), 6),
        fama_macbeth_t_stat=round(_f("fm_t_stat"), 4),
    )
