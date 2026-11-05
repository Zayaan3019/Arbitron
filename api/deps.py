"""Shared helpers for the API routers."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException

CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")

#: Ceiling on caller-supplied ticker lists.  Each name is a yfinance download
#: and a column in every matrix downstream, so an unbounded list is a cheap
#: way to make the server do expensive work.
MAX_TICKERS = 60


def resolve_tickers(
    universe: Optional[str],
    tickers: Optional[list[str]] = None,
) -> list[str]:
    """
    Resolve a request's universe/ticker fields into a concrete ticker list.

    Raises 400 on an unknown universe or an unusable ticker list, rather than
    letting an empty frame fail obscurely three layers down.
    """
    from arbitron.data import DEFAULT_UNIVERSE, SECTOR_UNIVERSES

    if tickers:
        cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
        if not cleaned:
            raise HTTPException(status_code=400, detail="`tickers` contained no usable symbols.")
        if len(cleaned) > MAX_TICKERS:
            raise HTTPException(
                status_code=400,
                detail=f"At most {MAX_TICKERS} tickers per request (got {len(cleaned)}).",
            )
        return cleaned

    if universe in (None, "", "SP500"):
        return list(DEFAULT_UNIVERSE)
    if universe in SECTOR_UNIVERSES:
        return list(SECTOR_UNIVERSES[universe])

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown universe '{universe}'. "
            f"Valid: SP500, {', '.join(sorted(SECTOR_UNIVERSES))}."
        ),
    )


def data_caveats(universe: Optional[str], tickers: Optional[list[str]] = None) -> dict:
    """
    The data-limitation block attached to every numeric response.

    Item 16 of the audit: either ship point-in-time membership or state the
    limitation prominently in the README AND the API response.  We do the
    latter, in both places, on every response rather than only in the docs.
    """
    from arbitron.data import universe_caveat

    return universe_caveat(universe=universe, tickers=tickers)


def load_prices(tickers: list[str], start: str, end: Optional[str]):
    """
    Fetch prices with uniform, honest error mapping.

    A yfinance outage is a 502 (upstream), an empty result is a 422 (the
    request cannot be satisfied), and anything else propagates as a 500.
    """
    from arbitron.data import MarketData

    try:
        md = MarketData(cache_dir=CACHE_DIR)
        prices = md.get_prices(tickers, start, end)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}")

    if prices is None or prices.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No price data returned for the requested tickers and date range. "
                "Check the symbols and that the range covers trading days."
            ),
        )
    return prices
