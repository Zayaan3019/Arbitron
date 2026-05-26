"""Tests for the portfolio optimization module."""

import numpy as np
import pandas as pd
import pytest

from arbitron.portfolio.optimizer import (
    sample_covariance,
    ledoit_wolf_covariance,
    constant_correlation_shrinkage,
    min_variance_portfolio,
    max_sharpe_portfolio,
    risk_parity_portfolio,
    mean_variance_frontier,
    portfolio_performance,
    diversification_ratio,
)


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

class TestCovarianceEstimators:
    def test_sample_cov_shape(self, returns):
        cov = sample_covariance(returns)
        n = len(returns.columns)
        assert cov.shape == (n, n)

    def test_sample_cov_symmetric(self, returns):
        cov = sample_covariance(returns)
        assert np.allclose(cov.values, cov.values.T)

    def test_sample_cov_psd(self, returns):
        cov = sample_covariance(returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert eigvals.min() > -1e-10

    def test_ledoit_wolf_shape(self, returns):
        cov = ledoit_wolf_covariance(returns)
        n = len(returns.columns)
        assert cov.shape == (n, n)

    def test_ledoit_wolf_psd(self, returns):
        cov = ledoit_wolf_covariance(returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert eigvals.min() > -1e-10

    def test_constant_corr_shrinkage(self, returns):
        cov = constant_correlation_shrinkage(returns, alpha=0.5)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert eigvals.min() > -1e-10


# ---------------------------------------------------------------------------
# min_variance_portfolio
# ---------------------------------------------------------------------------

class TestMinVariancePortfolio:
    def test_weights_sum_to_one(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w = min_variance_portfolio(cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_long_only(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w = min_variance_portfolio(cov, long_only=True)
        assert (w >= -1e-8).all()

    def test_minimises_variance(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w_mvp = min_variance_portfolio(cov)
        w_eq  = pd.Series(1 / len(returns.columns), index=returns.columns)
        var_mvp = float(w_mvp.values @ cov.values @ w_mvp.values)
        var_eq  = float(w_eq.values  @ cov.values @ w_eq.values)
        assert var_mvp <= var_eq + 1e-8


# ---------------------------------------------------------------------------
# max_sharpe_portfolio
# ---------------------------------------------------------------------------

class TestMaxSharpePortfolio:
    def test_weights_sum_to_one(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        w   = max_sharpe_portfolio(mu, cov)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_long_only_bounds(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        w   = max_sharpe_portfolio(mu, cov, long_only=True, max_weight=0.3)
        assert (w >= -1e-8).all()
        assert (w <= 0.3 + 1e-6).all()


# ---------------------------------------------------------------------------
# risk_parity_portfolio
# ---------------------------------------------------------------------------

class TestRiskParityPortfolio:
    def test_weights_sum_to_one(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_equal_risk_contribution(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        w_arr = w.values
        cov_arr = cov.loc[w.index, w.index].values
        port_vol = np.sqrt(w_arr @ cov_arr @ w_arr)
        mrc = cov_arr @ w_arr / port_vol
        rc  = w_arr * mrc
        rc_pct = rc / rc.sum()
        # Each asset should contribute ≈ 1/N of total risk
        target = 1 / len(w)
        assert np.allclose(rc_pct, target, atol=0.02)

    def test_all_weights_positive(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        assert (w >= 0).all()


# ---------------------------------------------------------------------------
# mean_variance_frontier
# ---------------------------------------------------------------------------

class TestMeanVarianceFrontier:
    def test_returns_dataframe(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        f   = mean_variance_frontier(mu, cov, n_points=5)
        assert isinstance(f, pd.DataFrame)
        assert 'return' in f.columns and 'volatility' in f.columns

    def test_frontier_is_monotone(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        f   = mean_variance_frontier(mu, cov, n_points=10)
        # Returns should be non-decreasing along the frontier
        assert (f['return'].diff().dropna() >= -1e-6).all()


# ---------------------------------------------------------------------------
# portfolio_performance
# ---------------------------------------------------------------------------

class TestPortfolioPerformance:
    def test_keys_present(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        w   = risk_parity_portfolio(cov)
        perf = portfolio_performance(w, mu, cov)
        assert 'expected_return' in perf
        assert 'volatility' in perf
        assert 'sharpe' in perf

    def test_volatility_non_negative(self, returns):
        cov = ledoit_wolf_covariance(returns)
        mu  = returns.mean() * 252
        w   = min_variance_portfolio(cov)
        perf = portfolio_performance(w, mu, cov)
        assert perf['volatility'] >= 0


# ---------------------------------------------------------------------------
# diversification_ratio
# ---------------------------------------------------------------------------

class TestDiversificationRatio:
    def test_dr_ge_one(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        dr  = diversification_ratio(w, cov)
        assert dr >= 1.0 - 1e-8

    def test_concentrated_portfolio_lower_dr(self, returns):
        cov = ledoit_wolf_covariance(returns)
        w_rp = risk_parity_portfolio(cov)
        # Concentrated: all in first stock
        tickers = list(cov.index)
        w_conc  = pd.Series([1.0] + [0.0] * (len(tickers) - 1), index=tickers)
        dr_rp   = diversification_ratio(w_rp, cov)
        dr_conc = diversification_ratio(w_conc, cov)
        assert dr_rp >= dr_conc - 1e-8
