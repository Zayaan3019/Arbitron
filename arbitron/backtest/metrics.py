"""
Performance metrics for portfolio backtests.

Industry-standard metrics:
  Sharpe ratio      — excess return / volatility (Sharpe, 1966)
  Sortino ratio     — excess return / downside deviation
  Calmar ratio      — CAGR / max drawdown
  Information ratio — active return / tracking error vs benchmark
  Omega ratio       — probability-weighted return vs loss
  Max drawdown      — peak-to-trough decline
  VaR / CVaR        — (also in risk module, duplicated here for portability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """All performance metrics for a backtest."""
    sharpe:          float
    sortino:         float
    calmar:          float
    cagr:            float
    max_drawdown:    float
    win_rate:        float
    avg_win:         float
    avg_loss:        float
    payoff_ratio:    float
    omega_ratio:     float
    skewness:        float
    kurtosis:        float
    n_days:          int

    def to_dict(self) -> dict:
        return {
            "sharpe":        round(self.sharpe,       4),
            "sortino":       round(self.sortino,      4),
            "calmar":        round(self.calmar,       4),
            "cagr":          round(self.cagr,         4),
            "max_drawdown":  round(self.max_drawdown, 4),
            "win_rate":      round(self.win_rate,     4),
            "avg_win":       round(self.avg_win,      6),
            "avg_loss":      round(self.avg_loss,     6),
            "payoff_ratio":  round(self.payoff_ratio, 4),
            "omega_ratio":   round(self.omega_ratio,  4),
            "skewness":      round(self.skewness,     4),
            "kurtosis":      round(self.kurtosis,     4),
            "n_days":        self.n_days,
        }


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_metrics(
    returns:    pd.Series,
    ann_factor: int = 252,
    rf:         float = 0.0,
) -> BacktestMetrics:
    """
    Compute full performance metrics from a daily return series.

    Args:
        returns:    Daily return series (not percentage, e.g. 0.01 = 1%).
        ann_factor: Trading days per year (252 for equities).
        rf:         Annualised risk-free rate.

    Returns:
        BacktestMetrics dataclass.
    """
    r = returns.dropna()
    if len(r) < 2:
        return BacktestMetrics(
            sharpe=np.nan, sortino=np.nan, calmar=np.nan,
            cagr=np.nan, max_drawdown=np.nan, win_rate=np.nan,
            avg_win=np.nan, avg_loss=np.nan, payoff_ratio=np.nan,
            omega_ratio=np.nan, skewness=np.nan, kurtosis=np.nan,
            n_days=len(r),
        )

    rf_daily = (1 + rf) ** (1 / ann_factor) - 1
    excess   = r - rf_daily

    # CAGR
    total_ret = (1 + r).prod()
    n_years   = len(r) / ann_factor
    cagr      = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe
    vol    = excess.std() * np.sqrt(ann_factor)
    sharpe = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else 0.0

    # Sortino
    downside   = excess[excess < 0]
    down_dev   = downside.std() * np.sqrt(ann_factor) if len(downside) > 1 else 1e-9
    sortino    = excess.mean() * ann_factor / down_dev if down_dev > 0 else 0.0

    # Max drawdown
    max_dd = max_drawdown(r)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else (cagr if cagr >= 0 else 0.0)

    # Win / loss stats
    wins  = r[r > 0]
    losses = r[r < 0]
    win_rate     = (r > 0).mean()
    avg_win      = wins.mean()  if len(wins)   > 0 else 0.0
    avg_loss     = losses.mean() if len(losses) > 0 else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

    # Omega ratio (threshold = 0)
    up   = r[r > 0].sum()
    down = -r[r < 0].sum()
    omega = up / down if down > 0 else np.nan

    # Higher moments
    from scipy.stats import skew, kurtosis
    sk  = float(skew(r.values))
    ku  = float(kurtosis(r.values, fisher=True))

    return BacktestMetrics(
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        cagr=cagr, max_drawdown=max_dd, win_rate=win_rate,
        avg_win=avg_win, avg_loss=avg_loss, payoff_ratio=payoff_ratio,
        omega_ratio=omega, skewness=sk, kurtosis=ku,
        n_days=len(r),
    )


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    cum = (1 + returns.dropna()).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (values ≤ 0)."""
    cum = (1 + returns.dropna()).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def sharpe_ratio(
    returns:    pd.Series,
    ann_factor: int = 252,
    rf:         float = 0.0,
) -> float:
    """Annualised Sharpe ratio."""
    rf_daily = (1 + rf) ** (1 / ann_factor) - 1
    excess   = returns.dropna() - rf_daily
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(ann_factor))


def sortino_ratio(
    returns:    pd.Series,
    ann_factor: int = 252,
    rf:         float = 0.0,
) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    rf_daily = (1 + rf) ** (1 / ann_factor) - 1
    excess   = returns.dropna() - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    down_dev = downside.std() * np.sqrt(ann_factor)
    return float(excess.mean() * ann_factor / down_dev) if down_dev > 0 else np.nan


def information_ratio(
    returns:          pd.Series,
    benchmark_returns: pd.Series,
    ann_factor:       int = 252,
) -> float:
    """Information ratio: active return / tracking error."""
    active = returns.dropna() - benchmark_returns.reindex(returns.index).fillna(0)
    te     = active.std() * np.sqrt(ann_factor)
    return float(active.mean() * ann_factor / te) if te > 0 else np.nan


def calmar_ratio(returns: pd.Series, ann_factor: int = 252) -> float:
    """Calmar ratio: CAGR / |max drawdown|."""
    r     = returns.dropna()
    total = (1 + r).prod()
    n_years = len(r) / ann_factor
    cagr   = total ** (1 / n_years) - 1 if n_years > 0 else 0.0
    mdd    = max_drawdown(r)
    return float(cagr / abs(mdd)) if mdd < 0 else np.nan


def rolling_sharpe(
    returns:    pd.Series,
    window:     int = 252,
    ann_factor: int = 252,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    roll = returns.rolling(window, min_periods=window // 2)
    return (roll.mean() / roll.std() * np.sqrt(ann_factor)).rename("rolling_sharpe")


def performance_table(
    strats:     dict[str, pd.Series],
    ann_factor: int = 252,
) -> pd.DataFrame:
    """
    Build a comparison table of performance metrics for multiple strategies.

    Args:
        strats: {name: daily_returns_series}.

    Returns:
        DataFrame with strategies as rows, metrics as columns.
    """
    rows = {}
    for name, rets in strats.items():
        m = compute_metrics(rets, ann_factor)
        rows[name] = m.to_dict()
    return pd.DataFrame(rows).T
