"""
Market data API router.

Handlers are plain `def`, not `async def`: they do blocking network I/O
(yfinance) and blocking CPU work.  On the event loop that stalls every other
request in the process; as a sync handler FastAPI runs them on the threadpool
where blocking is expected.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import PriceRequest, PriceResponse, UniverseListResponse

router = APIRouter()


@router.get("/universes", response_model=UniverseListResponse)
def list_universes() -> UniverseListResponse:
    """List available stock universes, with their data limitations."""
    from arbitron.data import DEFAULT_UNIVERSE, SECTOR_UNIVERSES

    return UniverseListResponse(
        universes={"SP500": list(DEFAULT_UNIVERSE), **SECTOR_UNIVERSES},
        data_caveats=data_caveats(universe=None),
    )


@router.post("/data/prices", response_model=PriceResponse)
def get_prices(req: PriceRequest) -> PriceResponse:
    """Fetch historical prices for a list of tickers or a named universe."""
    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

    # Leading NaNs are preserved on purpose (pre-IPO names); report the real
    # coverage instead of pretending the panel is rectangular.
    coverage = {str(col): int(prices[col].notna().sum()) for col in prices.columns}

    return PriceResponse(
        tickers=[str(c) for c in prices.columns],
        dates=[d.strftime("%Y-%m-%d") for d in prices.index],
        prices={str(col): prices[col].tolist() for col in prices.columns},
        n_rows=len(prices),
        n_cols=len(prices.columns),
        observations_per_ticker=coverage,
        data_caveats=data_caveats(req.universe, req.tickers),
    )
