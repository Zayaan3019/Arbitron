"""
Pydantic request/response schemas for the Arbitron API.

Two conventions worth knowing before reading further:

1. **Numeric response fields are `Optional[float]`.**  Quant metrics are
   genuinely undefined in ordinary cases — a Sharpe on two observations, a
   Calmar with no drawdown, an IC t-stat on three cross-sections.  Those
   arrive as NaN, which is not valid JSON and used to surface as an
   unhandled 500.  They are now `null`, never coerced to a plausible-looking
   0.0.

2. **Every numeric response carries `data_caveats`.**  The built-in universes
   are today's index members, so the numbers are upward-biased.  The
   limitation travels with the data rather than living only in the README.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataCaveats(BaseModel):
    """Data limitations attached to any response derived from a universe."""
    point_in_time: Optional[bool] = None
    survivorship_bias: Optional[bool] = None
    headline: str
    details: List[str] = Field(default_factory=list)
    remediation: Optional[str] = None
    universe: Optional[str] = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PriceRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"      # "SP500" | "Technology" | ...
    start_date: str = "2022-01-01"
    end_date: Optional[str] = "2024-12-31"


class PriceResponse(BaseModel):
    tickers: List[str]
    dates: List[str]
    prices: Dict[str, List[Optional[float]]]   # ticker → prices (null before IPO)
    n_rows: int
    n_cols: int
    #: Non-null observations per ticker.  Less than `n_rows` means the name
    #: was not listed (or no longer listed) for part of the window.
    observations_per_ticker: Dict[str, int] = Field(default_factory=dict)
    data_caveats: DataCaveats


class UniverseListResponse(BaseModel):
    universes: Dict[str, List[str]]
    data_caveats: DataCaveats


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

class FactorEvalRequest(BaseModel):
    factor_name: str
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    start_date: str = "2021-01-01"
    end_date: Optional[str] = "2024-12-31"
    horizon: int = Field(21, ge=2, le=252)


class ICStatsResponse(BaseModel):
    factor_name: str
    horizon: int
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    #: Newey-West HAC corrected. This is the one to quote.
    t_stat: Optional[float] = None
    #: iid t-stat, inflated ~sqrt(horizon) by overlapping returns. Reference only.
    t_stat_naive: Optional[float] = None
    hac_lags: int = 0
    pct_positive: Optional[float] = None
    n_periods: int = 0
    fama_macbeth_gamma: Optional[float] = None
    fama_macbeth_t_stat: Optional[float] = None
    fama_macbeth_t_stat_naive: Optional[float] = None
    fama_macbeth_p_value: Optional[float] = None
    q5_minus_q1_ann: Optional[float] = None
    data_caveats: DataCaveats


class FactorListResponse(BaseModel):
    factors: List[str]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    factor_name: str
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = "2024-12-31"
    n_quantiles: int = Field(5, ge=2, le=10)
    rebal_freq: str = Field("ME", pattern="^(ME|W|QE|SME)$")
    transaction_cost: float = Field(0.001, ge=0, le=0.05)
    long_only: bool = False


class BacktestMetricsResponse(BaseModel):
    factor_name: str
    cagr: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_return: Optional[float] = None
    ann_volatility: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_periods: int = 0


class BacktestResponse(BaseModel):
    metrics: BacktestMetricsResponse
    dates: List[str]
    cumulative_returns: List[Optional[float]]
    drawdown: List[Optional[float]]
    n_rebalances: int = 0
    #: ticker → holding days on which the name had no price and was dropped
    #: from the book (delisting, halt, or a gap longer than the fill limit).
    dropped_names: Dict[str, int] = Field(default_factory=dict)
    data_caveats: DataCaveats


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    start_date: str = "2022-01-01"
    end_date: Optional[str] = "2024-12-31"
    optimizer: str = Field("max_sharpe", pattern="^(max_sharpe|min_variance|risk_parity)$")
    rf: float = Field(0.02, ge=-0.1, le=0.5)
    max_weight: float = Field(0.3, gt=0.0, le=1.0)
    long_only: bool = True


class WeightsResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    #: Always 1.0 to within 1e-9. Exposed so the invariant is checkable by the
    #: caller — it used to come out at 1.4385 when shorts were clipped without
    #: renormalising.
    weight_sum: float
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe: Optional[float] = None
    diversification_ratio: Optional[float] = None
    n_observations: int = 0
    data_caveats: DataCaveats


class EfficientFrontierResponse(BaseModel):
    returns: List[Optional[float]]
    volatilities: List[Optional[float]]
    sharpes: List[Optional[float]]
    data_caveats: DataCaveats


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    weights: Optional[Dict[str, float]] = None   # if None → equal-weight
    start_date: str = "2022-01-01"
    end_date: Optional[str] = "2024-12-31"


class RiskResponse(BaseModel):
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None
    cvar_99: Optional[float] = None
    ulcer_index: Optional[float] = None
    tail_ratio: Optional[float] = None
    max_drawdown_duration_days: int = 0
    avg_drawdown_duration_days: int = 0
    n_drawdown_periods: int = 0
    market_beta: Optional[float] = None
    #: Current regime only — the causal output of the regime module.
    regime: Optional[str] = None
    regime_as_of: Optional[str] = None
    regime_is_causal: bool = True
    n_observations: int = 0
    data_caveats: DataCaveats


# ---------------------------------------------------------------------------
# Research Lab
# ---------------------------------------------------------------------------

class LabRunRequest(BaseModel):
    universe: Optional[str] = "SP500"
    tickers: Optional[List[str]] = None
    start_date: str = "2020-01-01"
    end_date: Optional[str] = "2024-12-31"
    factor_names: Optional[List[str]] = None
    horizon: int = Field(21, ge=5, le=126)
    include_foundation: bool = True
    n_quantiles: int = Field(5, ge=2, le=10)
    transaction_cost: float = Field(0.001, ge=0, le=0.05)
    max_factors: int = Field(9, ge=1, le=13)


class LabRunResponse(BaseModel):
    report: Dict[str, Any]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=10, max_length=2000)
    max_iterations: int = Field(15, ge=1, le=25)


class AgentEventPayload(BaseModel):
    event_type: str
    data: Any
    session_id: str


class AgentRunResponse(BaseModel):
    session_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
    api_key_required: bool = False
    live_sessions: int = 0
    max_sessions: int = 0
    running_tasks: int = 0
