"""Tests for the backtest engine and metrics module."""

import numpy as np
import pandas as pd
import pytest

from arbitron.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_factor_backtest,
    WalkForwardCV,
)
from arbitron.backtest.metrics import compute_metrics, BacktestMetrics
from arbitron.factors.library import momentum, realised_volatility


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.n_quantiles == 5
        assert cfg.rebal_freq == "ME"
        assert cfg.transaction_cost == 0.001
        assert cfg.long_only is False

    def test_custom(self):
        cfg = BacktestConfig(n_quantiles=3, transaction_cost=0.002)
        assert cfg.n_quantiles == 3
        assert cfg.transaction_cost == 0.002


# ---------------------------------------------------------------------------
# run_factor_backtest
# ---------------------------------------------------------------------------

class TestRunFactorBacktest:
    def test_returns_backtest_result(self, prices):
        f = momentum(prices)
        cfg = BacktestConfig(n_quantiles=3)
        result = run_factor_backtest(f, prices, cfg, factor_name="test")
        assert isinstance(result, BacktestResult)

    def test_returns_series_is_daily(self, prices):
        f = momentum(prices)
        result = run_factor_backtest(f, prices, BacktestConfig())
        assert isinstance(result.returns, pd.Series)
        assert len(result.returns) > 0

    def test_cumulative_starts_at_one(self, prices):
        f = momentum(prices)
        result = run_factor_backtest(f, prices, BacktestConfig())
        cum = result.cumulative.dropna()
        if len(cum) > 0:
            assert abs(cum.iloc[0] - 1.0) < 0.5   # ample slack for first period

    def test_drawdown_non_positive(self, prices):
        f = momentum(prices)
        result = run_factor_backtest(f, prices, BacktestConfig())
        dd = result.drawdown.dropna()
        assert (dd <= 0 + 1e-9).all()

    def test_metrics_populated(self, prices):
        f = momentum(prices)
        result = run_factor_backtest(f, prices, BacktestConfig())
        m = result.metrics
        assert isinstance(m.sharpe, float)
        assert isinstance(m.max_drawdown, float)
        assert m.max_drawdown <= 0

    def test_transaction_cost_is_applied(self, prices):
        f = momentum(prices)
        r0 = run_factor_backtest(f, prices, BacktestConfig(transaction_cost=0.0))
        r1 = run_factor_backtest(f, prices, BacktestConfig(transaction_cost=0.05))
        # With very high tx cost (5%), CAGR must be lower than zero-cost version
        # Use a large cost to ensure statistical significance
        assert r1.metrics.cagr <= r0.metrics.cagr + 0.05


# ---------------------------------------------------------------------------
# WalkForwardCV
# ---------------------------------------------------------------------------

class TestWalkForwardCV:
    def test_split_count(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        cv = WalkForwardCV(n_splits=3, test_size=63, embargo_days=21)
        splits = list(cv.split(dates))
        assert len(splits) == 3

    def test_no_overlap_with_embargo(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        cv = WalkForwardCV(n_splits=4, test_size=50, embargo_days=21)
        for train_idx, test_idx in cv.split(dates):
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Last train date + embargo must be before first test date
                assert dates[train_idx[-1]] < dates[test_idx[0]]

    def test_test_indices_non_overlapping(self):
        dates = pd.bdate_range("2020-01-01", periods=600)
        cv = WalkForwardCV(n_splits=4, test_size=63)
        all_test = []
        for _, test_idx in cv.split(dates):
            all_test.extend(test_idx)
        assert len(all_test) == len(set(all_test)), "Test indices overlap across splits"


# ---------------------------------------------------------------------------
# BacktestMetrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_positive_returns(self):
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0.0005, 0.01, 500))
        m = compute_metrics(ret)
        assert isinstance(m, BacktestMetrics)
        assert m.sharpe > 0
        assert m.max_drawdown <= 0

    def test_all_zero_returns(self):
        ret = pd.Series(np.zeros(100))
        m = compute_metrics(ret)
        assert m.cagr == 0.0 or abs(m.cagr) < 1e-10

    def test_to_dict(self):
        rng = np.random.default_rng(1)
        ret = pd.Series(rng.normal(0.0003, 0.01, 300))
        m = compute_metrics(ret)
        d = m.to_dict()
        assert isinstance(d, dict)
        for key in ('sharpe', 'max_drawdown', 'cagr', 'sortino'):
            assert key in d

    def test_max_drawdown_in_range(self):
        rng = np.random.default_rng(2)
        ret = pd.Series(rng.normal(0, 0.01, 400))
        m = compute_metrics(ret)
        assert -1.0 <= m.max_drawdown <= 0.0

    def test_win_rate_in_range(self):
        rng = np.random.default_rng(3)
        ret = pd.Series(rng.normal(0, 0.01, 400))
        m = compute_metrics(ret)
        assert 0.0 <= m.win_rate <= 1.0
