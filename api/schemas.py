"""
Pydantic request/response schemas for the Arbitron API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PriceRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    universe: Optional[str] = None         # "SP500" | "TECH" | ...
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    field: str = "Adj Close"


class PriceResponse(BaseModel):
    tickers: List[str]
    dates: List[str]
    prices: Dict[str, List[float]]         # ticker → list of prices
    n_rows: int
    n_cols: int


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

class FactorRequest(BaseModel):
    factor_name: str
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"


class FactorEvalRequest(BaseModel):
    factor_name: str
    universe: Optional[str] = "SP500"
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    horizon: int = 21


class ICStatsResponse(BaseModel):
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    t_stat: float
    pct_positive: float
    n_periods: int
    fama_macbeth_gamma: Optional[float] = None
    fama_macbeth_t_stat: Optional[float] = None


class FactorListResponse(BaseModel):
    factors: List[str]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    factor_name: str
    universe: Optional[str] = "SP500"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    n_quantiles: int = Field(5, ge=2, le=10)
    rebal_freq: str = "ME"
    transaction_cost: float = Field(0.001, ge=0, le=0.05)
    long_only: bool = False


class BacktestMetricsResponse(BaseModel):
    factor_name: str
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    total_return: float
    ann_volatility: float
    skewness: float
    kurtosis: float
    n_periods: int


class BacktestResponse(BaseModel):
    metrics: BacktestMetricsResponse
    dates: List[str]
    cumulative_returns: List[float]
    drawdown: List[float]


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    optimizer: str = Field("max_sharpe", pattern="^(max_sharpe|min_variance|risk_parity)$")
    rf: float = 0.02
    max_weight: float = 0.3


class WeightsResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe: float
    diversification_ratio: float


class EfficientFrontierResponse(BaseModel):
    returns: List[float]
    volatilities: List[float]
    sharpes: List[float]


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "SP500"
    weights: Optional[Dict[str, float]] = None   # if None → equal-weight
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"


class RiskResponse(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    ulcer_index: float
    tail_ratio: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: int
    n_drawdown_periods: int
    market_beta: Optional[float] = None
    regime: Optional[str] = None


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
    version: str = "0.1.0"
