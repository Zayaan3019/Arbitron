"""
Walk-forward backtesting engine with purged cross-validation.

Design principles (Lopez de Prado, 2018):
  1. No lookahead bias: signal at time t is computed using only data
     available strictly before t.
  2. Purging: remove train samples whose labels overlap with the test period
     (critical for overlapping returns like 21-day forward returns).
  3. Embargo: add a gap between the train and test set to prevent leakage
     from serial correlation in features.
  4. Walk-forward: re-estimate the model periodically on expanding data.

Two backtesting modes:
  a) Factor quintile: rank stocks by factor value into quintiles,
     go long top quintile, short bottom quintile (paper portfolio).
  b) ML signal: use a fitted model's predicted return ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from arbitron.factors.library import cross_section_rank
from arbitron.backtest.metrics import (
    BacktestMetrics, compute_metrics, compute_drawdown_series,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Walk-forward backtest configuration."""
    n_quantiles:    int   = 5           # Number of factor quintiles
    rebal_freq:     str   = "ME"        # Rebalance: 'ME'=month-end, 'W'=weekly
    horizon_days:   int   = 21          # Forward return horizon
    transaction_cost: float = 0.001     # One-way cost (bps × tickers)
    long_only:      bool  = False       # Long-only (top quintile only)
    ann_factor:     int   = 252         # Trading days per year
    train_window:   int   = 252         # Minimum train window for ML mode
    embargo_days:   int   = 5           # Embargo between train and test


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    returns:        pd.Series           # Daily strategy returns
    cumulative:     pd.Series           # Cumulative return (1+r).cumprod()
    drawdown:       pd.Series           # Drawdown series
    metrics:        BacktestMetrics     # Summary stats
    quintile_rets:  pd.DataFrame        # Average returns per quintile
    weights:        pd.DataFrame        # (rebal dates × tickers) weights
    factor_name:    str = ""
    config:         BacktestConfig = field(default_factory=BacktestConfig)

    def summary(self) -> dict:
        return {
            "factor":        self.factor_name,
            "sharpe":        self.metrics.sharpe,
            "sortino":       self.metrics.sortino,
            "calmar":        self.metrics.calmar,
            "cagr":          self.metrics.cagr,
            "max_drawdown":  self.metrics.max_drawdown,
            "win_rate":      self.metrics.win_rate,
            "n_days":        self.metrics.n_days,
        }


# ---------------------------------------------------------------------------
# Long-short factor portfolio backtest
# ---------------------------------------------------------------------------

def run_factor_backtest(
    factor:      pd.DataFrame,
    prices:      pd.DataFrame,
    cfg:         Optional[BacktestConfig] = None,
    factor_name: str = "factor",
) -> BacktestResult:
    """
    Quintile long-short factor backtest.

    Portfolio construction:
      - Long top quintile, short bottom quintile (equal weight within each).
      - Rebalance at the specified frequency.
      - Apply one-way transaction cost on turnover.

    Args:
        factor:  (dates × tickers) factor values (raw, not yet ranked).
        prices:  (dates × tickers) adjusted close prices.
        cfg:     Backtest configuration (defaults used if None).

    Returns:
        BacktestResult with full metrics and time series.
    """
    if cfg is None:
        cfg = BacktestConfig()

    returns = prices.pct_change()

    # Align factor and returns to common universe
    common_tickers = factor.columns.intersection(prices.columns)
    factor  = factor[common_tickers]
    returns = returns[common_tickers]

    # Get rebalancing dates
    rebal_dates = _rebalance_dates(returns.index, cfg.rebal_freq)

    strat_returns:  list[tuple[pd.Timestamp, float]] = []
    quintile_store: dict[int, list[float]] = {q: [] for q in range(1, cfg.n_quantiles + 1)}
    weights_store:  dict[pd.Timestamp, pd.Series] = {}
    prev_weights:   Optional[pd.Series] = None

    for i, rebal in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        # Factor signal as of `rebal` — rank cross-sectionally
        if rebal not in factor.index:
            continue
        f_row = factor.loc[rebal].dropna()
        if len(f_row) < cfg.n_quantiles * 2:
            continue

        # Compute quintile weights
        weights = _quintile_weights(f_row, cfg.n_quantiles, cfg.long_only)
        weights_store[rebal] = weights

        # Compute returns for the holding period [rebal, next_rebal)
        hold_dates  = returns.index[
            (returns.index > rebal) & (returns.index <= next_rebal)
        ]
        if hold_dates.empty:
            continue

        hold_rets   = returns.loc[hold_dates, weights.index]
        port_rets   = hold_rets.multiply(weights, axis=1).sum(axis=1)

        # Transaction cost on turnover
        if prev_weights is not None:
            prev_aligned = prev_weights.reindex(weights.index).fillna(0)
            turnover     = (weights - prev_aligned).abs().sum()
            tc_drag      = cfg.transaction_cost * turnover
        else:
            tc_drag = 0.0

        port_rets.iloc[0] -= tc_drag   # Apply cost on first day of period
        prev_weights = weights

        for day, r in port_rets.items():
            strat_returns.append((day, r))

        # Track per-quintile returns for diagnostics
        for q in range(1, cfg.n_quantiles + 1):
            q_tickers = f_row.index[
                pd.qcut(f_row, cfg.n_quantiles, labels=False, duplicates="drop") == (q - 1)
            ]
            if len(q_tickers):
                q_ret = returns.loc[hold_dates, q_tickers].mean(axis=1).mean()
                quintile_store[q].append(q_ret)

    if not strat_returns:
        # Return empty result
        empty = pd.Series(dtype=float)
        return BacktestResult(
            returns=empty, cumulative=empty, drawdown=empty,
            metrics=compute_metrics(empty),
            quintile_rets=pd.DataFrame(),
            weights=pd.DataFrame(),
            factor_name=factor_name,
            config=cfg,
        )

    ret_series = (
        pd.DataFrame(strat_returns, columns=["date", "return"])
        .set_index("date")["return"]
        .sort_index()
    )
    ret_series.index = pd.to_datetime(ret_series.index)

    cum    = (1 + ret_series).cumprod()
    dd     = compute_drawdown_series(ret_series)
    metrics = compute_metrics(ret_series, cfg.ann_factor)

    q_means = {
        q: np.nanmean(v) for q, v in quintile_store.items() if v
    }
    q_df = pd.DataFrame(
        list(q_means.items()), columns=["quintile", "mean_return"]
    ).set_index("quintile")

    w_df = pd.DataFrame(weights_store).T.fillna(0)

    return BacktestResult(
        returns=ret_series,
        cumulative=cum,
        drawdown=dd,
        metrics=metrics,
        quintile_rets=q_df,
        weights=w_df,
        factor_name=factor_name,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Return all period-end dates within `index`."""
    resampled = pd.Series(range(len(index)), index=index).resample(freq).last()
    # Intersect with actual trading days
    valid     = [d for d in resampled.index if d in index]
    return sorted(valid)


def _quintile_weights(
    factor_row:  pd.Series,
    n_quantiles: int,
    long_only:   bool,
) -> pd.Series:
    """
    Build a dollar-neutral long-short portfolio from a factor row.

    Returns a pd.Series of weights summing to 0 (long-short) or +1 (long-only).
    """
    try:
        ranks = pd.qcut(factor_row, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(dtype=float)

    top_q    = n_quantiles - 1  # Highest rank
    bot_q    = 0                # Lowest rank

    top_mask = ranks == top_q
    bot_mask = ranks == bot_q

    n_top = top_mask.sum()
    n_bot = bot_mask.sum()

    if n_top == 0:
        return pd.Series(dtype=float)

    weights = pd.Series(0.0, index=factor_row.index)
    weights[top_mask]  = 1.0 / n_top
    if not long_only and n_bot > 0:
        weights[bot_mask] = -1.0 / n_bot

    return weights[weights != 0]


# ---------------------------------------------------------------------------
# Walk-forward with purging (for ML models)
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """
    Walk-forward cross-validation with purging and embargo.

    Suitable for overlapping label problems (e.g. 21-day forward returns
    computed from daily prices create 21× overlap in labels).

    Each split:
      train_idx: [start, test_start - embargo_days]
      test_idx:  [test_start, test_start + test_size]
    """

    def __init__(
        self,
        n_splits:    int = 6,
        test_size:   int = 63,   # ~3 months
        embargo_days: int = 21,
    ) -> None:
        self.n_splits     = n_splits
        self.test_size    = test_size
        self.embargo_days = embargo_days

    def split(
        self,
        X: pd.DataFrame,
    ):
        """
        Yield (train_index, test_index) pairs.

        Args:
            X: DataFrame indexed by date.
        """
        n = len(X)
        test_size   = self.test_size
        embargo     = self.embargo_days
        split_size  = test_size + embargo

        for k in range(self.n_splits):
            test_end   = n - k * test_size
            test_start = test_end - test_size
            train_end  = test_start - embargo

            if train_end < test_size:
                continue

            train_idx = list(range(0, train_end))
            test_idx  = list(range(test_start, test_end))
            yield train_idx, test_idx
