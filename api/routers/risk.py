"""
Risk analytics API router.

Sync `def` handler on purpose — see the note in `routers/data.py`.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import RiskRequest, RiskResponse
from ..serialization import safe_float

router = APIRouter()

MIN_OBSERVATIONS = 30


@router.post("/risk", response_model=RiskResponse)
def compute_risk(req: RiskRequest) -> RiskResponse:
    """
    Compute risk metrics for a portfolio (or equal-weight if no weights given).

    The regime label is the **current** regime only — the single output of
    the regime module that is causal without qualification.  Historical
    regime labels are available from `arbitron.models.regime` via
    `predict_walk_forward`, and are deliberately not exposed as a
    full-sample decoded path.
    """
    from arbitron.data import to_returns
    from arbitron.models import build_regime_detector
    from arbitron.risk import full_risk_report

    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

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
                f"Only {len(returns)} complete observations; need at least "
                f"{MIN_OBSERVATIONS} for meaningful risk statistics."
            ),
        )

    if req.weights:
        w = pd.Series(req.weights, dtype=float).reindex(returns.columns).fillna(0.0)
        total = w.sum()
        if abs(total) < 1e-12:
            raise HTTPException(
                status_code=400,
                detail="Supplied weights sum to zero over the resolved tickers.",
            )
        w = w / total
    else:
        w = pd.Series(1.0 / len(returns.columns), index=returns.columns)

    port_ret = (returns * w).sum(axis=1)
    market_ret = returns.mean(axis=1)          # equal-weight market proxy

    report = full_risk_report(port_ret, market_ret=market_ret)

    detector = build_regime_detector(use_hmm=False)
    detector.fit(market_ret)
    regime_detail = detector.current_regime_detail(market_ret)

    return RiskResponse(
        var_95=safe_float(report.get("var_95")),
        var_99=safe_float(report.get("var_99")),
        cvar_95=safe_float(report.get("cvar_95")),
        cvar_99=safe_float(report.get("cvar_99")),
        ulcer_index=safe_float(report.get("ulcer_index")),
        tail_ratio=safe_float(report.get("tail_ratio")),
        max_drawdown_duration_days=int(report.get("max_duration_days", 0)),
        avg_drawdown_duration_days=int(report.get("avg_duration_days", 0)),
        n_drawdown_periods=int(report.get("n_drawdown_periods", 0)),
        market_beta=safe_float(report.get("market_beta")),
        regime=regime_detail.get("regime"),
        regime_as_of=regime_detail.get("as_of"),
        regime_is_causal=bool(regime_detail.get("causal", True)),
        n_observations=len(returns),
        data_caveats=data_caveats(req.universe, req.tickers),
    )
