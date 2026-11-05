"""
Backtest API router.

Sync `def` handler on purpose — see the note in `routers/data.py`.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from ..deps import data_caveats, load_prices, resolve_tickers
from ..schemas import BacktestMetricsResponse, BacktestRequest, BacktestResponse
from ..serialization import safe_float

router = APIRouter()


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """
    Run a walk-forward factor backtest and return cumulative returns + metrics.

    Every metric may be `null`: a run with too few observations has an
    undefined Sharpe, a run with no losing day has an undefined payoff ratio.
    Those are reported as null rather than coerced to a misleading 0.0.
    """
    from arbitron.backtest import BacktestConfig, run_factor_backtest
    from arbitron.factors import compute_factor

    tickers = resolve_tickers(req.universe, req.tickers)
    prices = load_prices(tickers, req.start_date, req.end_date)

    try:
        factor = compute_factor(req.factor_name, prices)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg = BacktestConfig(
        n_quantiles=req.n_quantiles,
        rebal_freq=req.rebal_freq,
        transaction_cost=req.transaction_cost,
        long_only=req.long_only,
    )
    result = run_factor_backtest(factor, prices, cfg, factor_name=req.factor_name)

    if result.returns.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "Backtest produced no trades. The universe needs at least "
                f"{cfg.n_quantiles * 2} names with a valid factor value on a "
                "rebalance date — widen the date range or the universe."
            ),
        )

    m = result.metrics
    cum_clean = result.cumulative.dropna()
    total_ret = safe_float(cum_clean.iloc[-1] - 1.0) if not cum_clean.empty else None
    ann_vol = safe_float(result.returns.std() * np.sqrt(cfg.ann_factor))

    metrics = BacktestMetricsResponse(
        factor_name=req.factor_name,
        cagr=safe_float(m.cagr),
        sharpe=safe_float(m.sharpe),
        sortino=safe_float(m.sortino),
        calmar=safe_float(m.calmar),
        max_drawdown=safe_float(m.max_drawdown),
        win_rate=safe_float(m.win_rate),
        total_return=total_ret,
        ann_volatility=ann_vol,
        skewness=safe_float(m.skewness),
        kurtosis=safe_float(m.kurtosis),
        n_periods=int(m.n_days),
    )

    cum = result.cumulative.fillna(1.0)
    dd = result.drawdown.fillna(0.0)

    return BacktestResponse(
        metrics=metrics,
        dates=[d.strftime("%Y-%m-%d") for d in cum.index],
        cumulative_returns=[safe_float(v) for v in cum.tolist()],
        drawdown=[safe_float(v) for v in dd.tolist()],
        n_rebalances=result.n_rebalances,
        dropped_names=result.dropped_names,
        data_caveats=data_caveats(req.universe, req.tickers),
    )
