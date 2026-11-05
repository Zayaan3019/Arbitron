"""
Look-ahead detection.

A backtest that peeks at the future produces beautiful numbers and is worth
nothing.  These tests attack the question from three directions:

1. **Positive control** — a factor that IS the future return must score near
   a perfect IC.  If it does not, the evaluation machinery is broken and no
   other test here means anything.
2. **Shift the factor** — that same cheating factor, delayed past the point
   where it could have been known, must lose its edge entirely.  If delaying
   a signal does not change the result, the pipeline is reading ahead.
3. **Truncation invariance** — computing a factor (or running a backtest) on
   `prices[:T]` must give the same answer as computing it on the full panel
   and slicing to `T`.  Any dependence on data after `T` shows up as a
   mismatch.  This is the structural test: it catches leakage that a
   statistical test would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arbitron.backtest.engine import BacktestConfig, run_factor_backtest
from arbitron.factors.evaluation import (
    forward_returns,
    ic_stats,
    information_coefficient,
)
from arbitron.factors.library import FACTOR_REGISTRY, compute_factor

HORIZON = 21


@pytest.fixture
def panel() -> pd.DataFrame:
    """A price panel with enough history for the 252-day factors."""
    rng = np.random.default_rng(20250101)
    n_days, n_stocks = 900, 16
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    rets = rng.normal(0.0004, 0.014, size=(n_days, n_stocks))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"S{i:02d}" for i in range(n_stocks)])


# ---------------------------------------------------------------------------
# 1. Positive control — the machinery can see prediction when it exists
# ---------------------------------------------------------------------------

def test_perfect_foresight_factor_scores_a_near_perfect_ic(panel):
    """A factor equal to the forward return must have IC ~ 1.0."""
    cheat = forward_returns(panel, HORIZON)
    ic = information_coefficient(cheat, forward_returns(panel, HORIZON))
    stats = ic_stats(ic, horizon=HORIZON)

    assert stats["ic_mean"] > 0.99, (
        f"IC machinery failed its positive control: {stats['ic_mean']}"
    )


def test_perfect_foresight_factor_makes_the_backtester_rich(panel):
    """
    The backtester must be capable of exploiting a genuine edge — otherwise a
    Sharpe of zero on a real factor would be uninformative.
    """
    cheat = forward_returns(panel, HORIZON)
    result = run_factor_backtest(cheat, panel, BacktestConfig(), "cheat")

    assert not result.returns.empty
    assert result.metrics.sharpe > 3.0, (
        f"Backtester could not profit from perfect foresight (Sharpe "
        f"{result.metrics.sharpe:.2f}); it is not measuring what it claims."
    )


# ---------------------------------------------------------------------------
# 2. Shift the factor — delaying a signal must destroy it
# ---------------------------------------------------------------------------

def test_delaying_the_cheating_factor_destroys_its_ic(panel):
    """
    Shift the perfect-foresight factor forward by more than the horizon, so
    that by the time it is 'known' the return it predicted has already
    happened.  Its IC against the NEXT window must collapse to noise.
    """
    cheat = forward_returns(panel, HORIZON)
    stale = cheat.shift(2 * HORIZON)          # only knowable long after the fact

    fwd = forward_returns(panel, HORIZON)
    ic_live = ic_stats(information_coefficient(cheat, fwd), horizon=HORIZON)["ic_mean"]
    ic_stale = ic_stats(information_coefficient(stale, fwd), horizon=HORIZON)["ic_mean"]

    assert ic_live > 0.99
    assert abs(ic_stale) < 0.15, (
        f"A signal delayed past its own horizon still had IC {ic_stale:.3f}. "
        "The evaluation is reading data it should not have."
    )


def test_backtest_edge_disappears_when_the_signal_is_delayed(panel):
    """The same reversal, measured in Sharpe rather than IC."""
    cheat = forward_returns(panel, HORIZON)
    stale = cheat.shift(3 * HORIZON)

    live_sharpe = run_factor_backtest(cheat, panel, BacktestConfig(), "live").metrics.sharpe
    stale_sharpe = run_factor_backtest(stale, panel, BacktestConfig(), "stale").metrics.sharpe

    assert live_sharpe > 3.0
    assert stale_sharpe < 1.0, (
        f"A stale signal still earned Sharpe {stale_sharpe:.2f}; the backtest "
        "is not respecting signal timing."
    )
    assert stale_sharpe < live_sharpe / 3


# ---------------------------------------------------------------------------
# 3. Truncation invariance — the structural test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor_name", sorted(FACTOR_REGISTRY))
def test_factor_value_does_not_depend_on_future_prices(panel, factor_name):
    """
    For every factor in the registry: the value at date T computed from
    `prices[:T]` must equal the value computed from the full panel.

    A factor that uses any future observation — a centred rolling window, a
    full-sample z-score, a back-filled price — fails here immediately.
    """
    cutoff = 700
    truncated = panel.iloc[:cutoff]

    full_factor = compute_factor(factor_name, panel).iloc[:cutoff]
    trunc_factor = compute_factor(factor_name, truncated)

    pd.testing.assert_frame_equal(
        trunc_factor, full_factor,
        check_exact=False, rtol=1e-10, atol=1e-12,
        obj=f"factor '{factor_name}' leaked future data",
    )


def test_backtest_returns_do_not_depend_on_future_prices(panel):
    """
    Same invariance for the backtester itself: strategy returns up to date T
    must be identical whether or not the engine was shown data after T.

    Only the overlapping prefix is compared — the truncated run legitimately
    has no final partial holding period.
    """
    cutoff = 700
    factor_full = compute_factor("momentum_252_21", panel)
    factor_trunc = compute_factor("momentum_252_21", panel.iloc[:cutoff])

    full = run_factor_backtest(factor_full, panel, BacktestConfig(), "mom")
    trunc = run_factor_backtest(factor_trunc, panel.iloc[:cutoff], BacktestConfig(), "mom")

    common = full.returns.index.intersection(trunc.returns.index)
    assert len(common) > 200, "not enough overlap to make the comparison meaningful"

    pd.testing.assert_series_equal(
        trunc.returns.loc[common], full.returns.loc[common],
        check_exact=False, rtol=1e-10, atol=1e-12,
        obj="backtest returns leaked future data",
    )


def test_rebalance_weights_do_not_depend_on_future_prices(panel):
    """The positions taken on each rebalance date must also be unchanged."""
    cutoff = 700
    factor_full = compute_factor("volatility_63", panel)
    factor_trunc = compute_factor("volatility_63", panel.iloc[:cutoff])

    full = run_factor_backtest(factor_full, panel, BacktestConfig(), "vol")
    trunc = run_factor_backtest(factor_trunc, panel.iloc[:cutoff], BacktestConfig(), "vol")

    common_dates = full.weights.index.intersection(trunc.weights.index)
    common_cols = full.weights.columns.intersection(trunc.weights.columns)
    assert len(common_dates) > 20

    pd.testing.assert_frame_equal(
        trunc.weights.loc[common_dates, common_cols],
        full.weights.loc[common_dates, common_cols],
        check_exact=False, rtol=1e-10, atol=1e-12,
        obj="rebalance weights leaked future data",
    )


def test_forward_returns_are_aligned_to_the_signal_date(panel):
    """
    `forward_returns(p, h)[t]` must be the return from t to t+h — not t-h to
    t.  An off-by-one-window here silently turns every IC into a look-ahead.
    """
    fwd = forward_returns(panel, HORIZON)
    col = panel.columns[0]
    t = 300

    expected = panel[col].iloc[t + HORIZON] / panel[col].iloc[t] - 1
    assert fwd[col].iloc[t] == pytest.approx(expected, rel=1e-12)

    # The last `horizon` rows cannot be known and must be NaN.
    assert fwd.iloc[-HORIZON:].isna().all().all()
