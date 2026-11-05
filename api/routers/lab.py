"""
Flagship research-lab API router.

Sync `def` handler on purpose.  This endpoint runs a dozen backtests plus a
walk-forward Ridge adapter — minutes of pure CPU.  As `async def` it held the
event loop for the whole run and froze every other request in the process,
including `/health`, which is what made Render mark the service dead.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import LabRunRequest, LabRunResponse
from ..security import rate_limit_expensive

router = APIRouter()


@router.post(
    "/lab/run",
    response_model=LabRunResponse,
    dependencies=[Depends(rate_limit_expensive)],
)
def run_lab(req: LabRunRequest) -> LabRunResponse:
    """Run the autonomous quant research lab over a universe or ticker list."""
    from arbitron.lab import run_flagship_research_lab

    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

    report = run_flagship_research_lab(
        prices=prices,
        factor_names=req.factor_names,
        horizon=req.horizon,
        include_foundation=req.include_foundation,
        n_quantiles=req.n_quantiles,
        transaction_cost=req.transaction_cost,
        max_factors=req.max_factors,
    )
    report["data_caveats"] = data_caveats(req.universe, req.tickers)

    return LabRunResponse(report=report)
