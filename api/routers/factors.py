"""
Factor library API router.

Sync `def` handlers on purpose — see the note in `routers/data.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import FactorEvalRequest, FactorListResponse, ICStatsResponse
from ..serialization import safe_float

router = APIRouter()


@router.get("/factors", response_model=FactorListResponse)
def list_factors() -> FactorListResponse:
    """List all available factor names."""
    from arbitron.factors import FACTOR_REGISTRY

    return FactorListResponse(factors=sorted(FACTOR_REGISTRY.keys()))


@router.post("/factors/evaluate", response_model=ICStatsResponse)
def evaluate_factor(req: FactorEvalRequest) -> ICStatsResponse:
    """
    Evaluate a single factor via IC statistics and Fama-MacBeth regression.

    Both t-statistics are returned.  `t_stat` is Newey-West corrected for the
    overlap the `horizon`-day forward return induces; `t_stat_naive` is the
    iid figure, inflated by roughly sqrt(horizon).  The naive number is shown
    only so the size of the correction is visible — do not quote it.
    """
    from arbitron.factors import compute_factor, factor_summary

    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

    try:
        factor = compute_factor(req.factor_name, prices)
        stats = factor_summary(req.factor_name, factor, prices, req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ICStatsResponse(
        factor_name=req.factor_name,
        horizon=req.horizon,
        ic_mean=safe_float(stats.get("ic_ic_mean")),
        ic_std=safe_float(stats.get("ic_ic_std")),
        ic_ir=safe_float(stats.get("ic_ic_ir")),
        t_stat=safe_float(stats.get("ic_t_stat")),
        t_stat_naive=safe_float(stats.get("ic_t_stat_naive")),
        hac_lags=int(stats.get("ic_hac_lags") or 0),
        pct_positive=safe_float(stats.get("ic_pct_pos")),
        n_periods=int(stats.get("ic_n_periods") or 0),
        fama_macbeth_gamma=safe_float(stats.get("fm_gamma")),
        fama_macbeth_t_stat=safe_float(stats.get("fm_t_stat")),
        fama_macbeth_t_stat_naive=safe_float(stats.get("fm_t_stat_naive")),
        fama_macbeth_p_value=safe_float(stats.get("fm_p_value")),
        q5_minus_q1_ann=safe_float(stats.get("q5_minus_q1_mean")),
        data_caveats=data_caveats(req.universe, req.tickers),
    )
