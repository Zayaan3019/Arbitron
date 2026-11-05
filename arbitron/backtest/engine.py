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

from arbitron.backtest.metrics import (
    BacktestMetrics, compute_metrics, compute_drawdown_series,
)
from arbitron.trials import record_trial


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
    #: ticker → number of holding days on which the name had no price and
    #: was therefore dropped from the book and its weight redistributed.
    dropped_names:  dict = field(default_factory=dict)
    #: Number of rebalance dates the engine actually traded.
    n_rebalances:   int = 0

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
            "n_rebalances":  self.n_rebalances,
            "dropped_names": self.dropped_names,
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

    Missing prices: `MarketData` preserves genuine NaNs (pre-IPO,
    post-delisting), so a held name can stop quoting mid-period.  Such a name
    is removed from the book for the affected days and each leg is
    renormalised over the names that are still live, preserving the book's
    gross exposure.  Silently summing over NaN — the previous behaviour —
    would have let a delisted name keep its weight while contributing
    exactly zero return.  Every drop is reported in
    `BacktestResult.dropped_names`.

    Args:
        factor:  (dates × tickers) factor values (raw, not yet ranked).
        prices:  (dates × tickers) adjusted close prices.
        cfg:     Backtest configuration (defaults used if None).

    Returns:
        BacktestResult with full metrics and time series.
    """
    if cfg is None:
        cfg = BacktestConfig()

    returns = prices.pct_change(fill_method=None)

    # Align factor and returns to common universe
    common_tickers = factor.columns.intersection(prices.columns)
    factor  = factor[common_tickers]
    returns = returns[common_tickers]

    # Get rebalancing dates
    rebal_dates = _rebalance_dates(returns.index, cfg.rebal_freq)

    strat_returns:  list[tuple[pd.Timestamp, float]] = []
    quintile_store: dict[int, list[float]] = {q: [] for q in range(1, cfg.n_quantiles + 1)}
    weights_store:  dict[pd.Timestamp, pd.Series] = {}
    dropped_names:  dict[str, int] = {}
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

        hold_rets = returns.loc[hold_dates, weights.index]

        # Drop names with no price on a given day and renormalise each leg
        # over the survivors, so the book keeps its intended gross exposure.
        eff_weights, day_drops = _live_weights(weights, hold_rets)
        for ticker, n in day_drops.items():
            dropped_names[ticker] = dropped_names.get(ticker, 0) + n

        # min_count=1 → a day on which NO held name quotes is NaN, not a
        # fabricated 0.0 return.
        port_rets = (eff_weights * hold_rets).sum(axis=1, min_count=1)

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
            dropped_names=dropped_names,
            n_rebalances=0,
        )

    ret_series = (
        pd.DataFrame(strat_returns, columns=["date", "return"])
        .set_index("date")["return"]
        .sort_index()
        .dropna()
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

    # Every backtest is a trial.  Record it so the Deflated Sharpe Ratio
    # knows the true size of the search rather than only the survivors that
    # made it into a comparison table.
    record_trial(factor_name or "unnamed", metrics.sharpe)

    return BacktestResult(
        returns=ret_series,
        cumulative=cum,
        drawdown=dd,
        metrics=metrics,
        quintile_rets=q_df,
        weights=w_df,
        factor_name=factor_name,
        config=cfg,
        dropped_names=dict(sorted(dropped_names.items(), key=lambda kv: -kv[1])),
        n_rebalances=len(weights_store),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """
    Return the last TRADING day of every period in `index`.

    `resample(freq).last()` produces calendar period-end labels (31 Jan,
    28 Feb, 31 Mar, ...).  Those labels are not trading days whenever a month
    ends on a weekend or a holiday, so intersecting the resampled *index*
    with the trading index — the previous implementation — silently deleted
    those months entirely: 8 of 28 rebalances vanished on a typical run.

    The fix is to resample the positional VALUES and map them back, which
    yields the true last trading day of each period.
    """
    if len(index) == 0:
        return []
    resampled = pd.Series(range(len(index)), index=index).resample(freq).last()
    return [index[int(i)] for i in resampled.dropna().values]


def _live_weights(
    weights:   pd.Series,
    hold_rets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Expand a static weight vector over the holding period, zeroing names with
    no price and renormalising each leg over the survivors.

    Long and short legs are renormalised *separately* so that a delisting on
    one side cannot quietly turn a dollar-neutral book into a directional
    one.  Returns the (days × tickers) effective weights and a
    {ticker: n_days_dropped} diagnostic.
    """
    valid = hold_rets.notna()
    base  = np.tile(weights.values.astype(float), (len(hold_rets), 1))
    base  = np.where(valid.values, base, 0.0)

    target_long  = float(weights[weights > 0].sum())
    target_short = float(weights[weights < 0].sum())   # <= 0

    pos = np.where(base > 0, base, 0.0)
    neg = np.where(base < 0, base, 0.0)

    pos_sum = pos.sum(axis=1, keepdims=True)
    neg_sum = neg.sum(axis=1, keepdims=True)

    if target_long > 0:
        pos = np.divide(pos, pos_sum, out=np.zeros_like(pos), where=pos_sum > 0) * target_long
    if target_short < 0:
        neg = np.divide(neg, neg_sum, out=np.zeros_like(neg), where=neg_sum < 0) * target_short

    effective = pd.DataFrame(
        pos + neg, index=hold_rets.index, columns=hold_rets.columns
    ).where(valid)

    missing = (~valid).sum(axis=0)
    drops = {str(t): int(n) for t, n in missing.items() if n > 0}
    return effective, drops


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
