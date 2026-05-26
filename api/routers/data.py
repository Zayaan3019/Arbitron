"""Market data API router."""

from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, HTTPException

from ..schemas import PriceRequest, PriceResponse, FactorListResponse

router = APIRouter()

CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


@router.get("/universes", response_model=dict)
async def list_universes():
    """List available stock universes."""
    from arbitron.data import SECTOR_UNIVERSES, DEFAULT_UNIVERSE
    return {
        "SP500": DEFAULT_UNIVERSE,
        **SECTOR_UNIVERSES,
    }


@router.post("/data/prices", response_model=PriceResponse)
async def get_prices(req: PriceRequest):
    """Fetch historical prices for a list of tickers or a named universe."""
    from arbitron.data import MarketData, SECTOR_UNIVERSES, DEFAULT_UNIVERSE

    tickers = req.tickers
    if not tickers:
        if req.universe == "SP500":
            tickers = DEFAULT_UNIVERSE
        elif req.universe and req.universe in SECTOR_UNIVERSES:
            tickers = SECTOR_UNIVERSES[req.universe]
        else:
            raise HTTPException(status_code=400,
                                detail="Provide tickers or a valid universe name")

    try:
        md = MarketData(cache_dir=CACHE_DIR)
        prices = md.get_prices(tickers, req.start_date, req.end_date, req.field)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return PriceResponse(
        tickers=list(prices.columns),
        dates=[d.strftime("%Y-%m-%d") for d in prices.index],
        prices={col: prices[col].tolist() for col in prices.columns},
        n_rows=len(prices),
        n_cols=len(prices.columns),
    )
