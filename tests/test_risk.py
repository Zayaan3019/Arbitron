"""Tests for the risk analytics module."""

import numpy as np
import pandas as pd
import pytest

from arbitron.risk.analytics import (
    historical_var,
    historical_cvar,
    parametric_var,
    monte_carlo_var,
    var_table,
    factor_beta,
    multi_factor_exposure,
    marginal_risk_contribution,
    component_risk_contribution,
    ulcer_index,
    drawdown_duration,
    tail_ratio,
    full_risk_report,
)


@pytest.fixture
def ret_series():
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.0003, 0.012, 500))


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

class TestHistoricalVaR:
    def test_positive(self, ret_series):
        assert historical_var(ret_series, 0.95) > 0

    def test_99_gt_95(self, ret_series):
        assert historical_var(ret_series, 0.99) >= historical_var(ret_series, 0.95)

    def test_cvar_ge_var(self, ret_series):
        var  = historical_var(ret_series, 0.95)
        cvar = historical_cvar(ret_series, 0.95)
        assert cvar >= var - 1e-12


class TestParametricVaR:
    def test_positive(self, ret_series):
        assert parametric_var(ret_series, 0.95) > 0

    def test_scales_with_horizon(self, ret_series):
        v1 = parametric_var(ret_series, 0.95, horizon=1)
        v5 = parametric_var(ret_series, 0.95, horizon=5)
        assert abs(v5 - v1 * np.sqrt(5)) < 0.01


class TestMonteCarloVaR:
    def test_returns_tuple(self, ret_series):
        result = monte_carlo_var(ret_series, n_sim=1000)
        assert isinstance(result, tuple) and len(result) == 2

    def test_both_positive(self, ret_series):
        var, cvar = monte_carlo_var(ret_series, n_sim=1000)
        assert var > 0 and cvar > 0

    def test_cvar_ge_var(self, ret_series):
        var, cvar = monte_carlo_var(ret_series, n_sim=1000)
        assert cvar >= var - 1e-10


class TestVarTable:
    def test_shape(self, ret_series):
        tbl = var_table(ret_series)
        assert len(tbl) == 3
        assert 'historical_var' in tbl.columns

    def test_sorted_ascending(self, ret_series):
        tbl = var_table(ret_series)
        assert (tbl['historical_var'].diff().dropna() >= 0).all()


# ---------------------------------------------------------------------------
# Factor beta / multi-factor
# ---------------------------------------------------------------------------

class TestFactorBeta:
    def test_returns_float(self, ret_series, market_ret):
        beta = factor_beta(ret_series, market_ret.reindex(ret_series.index))
        assert isinstance(beta, float)

    def test_high_correlation_near_one(self):
        rng = np.random.default_rng(5)
        f   = pd.Series(rng.normal(0, 0.01, 300))
        r   = f + rng.normal(0, 0.001, 300)   # almost identical
        beta = factor_beta(r, f)
        assert abs(beta - 1.0) < 0.15

    def test_low_correlation_near_zero(self):
        rng = np.random.default_rng(6)
        f   = pd.Series(rng.normal(0, 0.01, 300))
        r   = pd.Series(rng.normal(0, 0.01, 300))  # independent
        beta = factor_beta(r, f)
        assert abs(beta) < 0.3


class TestMultiFactorExposure:
    def test_returns_dataframe(self, returns, market_ret):
        port_ret = returns.mean(axis=1)
        factors  = pd.DataFrame({
            "mkt": market_ret.reindex(port_ret.index),
        })
        result = multi_factor_exposure(port_ret, factors)
        assert isinstance(result, pd.DataFrame)
        assert 'alpha' in result.index
        assert 'mkt' in result.index

    def test_t_stats_finite(self, returns, market_ret):
        port_ret = returns.mean(axis=1)
        factors  = pd.DataFrame({
            "mkt": market_ret.reindex(port_ret.index),
        })
        result = multi_factor_exposure(port_ret, factors)
        assert np.isfinite(result['t_stat']).all()


# ---------------------------------------------------------------------------
# MRC / CRC
# ---------------------------------------------------------------------------

class TestMarginalRiskContribution:
    def test_shape(self, returns):
        from arbitron.portfolio import ledoit_wolf_covariance, risk_parity_portfolio
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        mrc = marginal_risk_contribution(w, cov)
        assert len(mrc) == len(w)

    def test_mrc_times_weights_sums_to_vol(self, returns):
        from arbitron.portfolio import ledoit_wolf_covariance, risk_parity_portfolio
        cov = ledoit_wolf_covariance(returns)
        w   = risk_parity_portfolio(cov)
        mrc = marginal_risk_contribution(w, cov)
        crc_sum = (w * mrc).sum()
        port_vol = np.sqrt(float(w.values @ cov.loc[w.index, w.index].values @ w.values))
        assert abs(crc_sum - port_vol) < 1e-8


class TestComponentRiskContribution:
    def test_crc_pct_sums_to_100(self, returns):
        from arbitron.portfolio import ledoit_wolf_covariance, risk_parity_portfolio
        cov  = ledoit_wolf_covariance(returns)
        w    = risk_parity_portfolio(cov)
        df   = component_risk_contribution(w, cov)
        assert abs(df['crc_pct'].sum() - 100.0) < 1.0   # rounding in display


# ---------------------------------------------------------------------------
# Drawdown / tail
# ---------------------------------------------------------------------------

class TestUlcerIndex:
    def test_positive(self, ret_series):
        assert ulcer_index(ret_series) >= 0

    def test_all_positive_returns_near_zero(self):
        ret = pd.Series([0.001] * 100)
        assert ulcer_index(ret) < 1e-6


class TestDrawdownDuration:
    def test_keys(self, ret_series):
        d = drawdown_duration(ret_series)
        assert 'max_duration_days' in d
        assert 'avg_duration_days' in d
        assert 'n_drawdown_periods' in d

    def test_max_ge_avg(self, ret_series):
        d = drawdown_duration(ret_series)
        assert d['max_duration_days'] >= d['avg_duration_days']


class TestTailRatio:
    def test_right_skewed_above_one(self):
        rng = np.random.default_rng(7)
        # Chi-squared distribution → right-skewed
        ret = pd.Series(rng.chisquare(df=5, size=500) / 500 - 0.01)
        tr  = tail_ratio(ret)
        assert tr > 1.0

    def test_left_skewed_below_one(self):
        rng = np.random.default_rng(8)
        ret = pd.Series(-(rng.chisquare(df=5, size=500) / 500 - 0.01))
        tr  = tail_ratio(ret)
        assert tr <= 1.0 + 0.1


# ---------------------------------------------------------------------------
# full_risk_report
# ---------------------------------------------------------------------------

class TestFullRiskReport:
    def test_returns_dict(self, ret_series, market_ret):
        mr = market_ret.reindex(ret_series.index)
        r  = full_risk_report(ret_series, market_ret=mr)
        assert isinstance(r, dict)

    def test_required_keys(self, ret_series):
        r = full_risk_report(ret_series)
        for k in ('var_95', 'var_99', 'cvar_95', 'ulcer_index', 'tail_ratio'):
            assert k in r

    def test_values_finite(self, ret_series):
        r = full_risk_report(ret_series)
        for k, v in r.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"{k} is not finite"
