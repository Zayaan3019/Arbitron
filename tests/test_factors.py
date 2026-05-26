"""Tests for the factor library and evaluation module."""

import numpy as np
import pandas as pd
import pytest

from arbitron.factors.library import (
    cross_section_zscore,
    cross_section_rank,
    momentum,
    short_term_reversal,
    realised_volatility,
    rsi_factor,
    FACTOR_REGISTRY,
    compute_factor,
)
from arbitron.factors.evaluation import (
    information_coefficient,
    ic_stats,
    fama_macbeth,
)


# ---------------------------------------------------------------------------
# cross_section_zscore
# ---------------------------------------------------------------------------

class TestCrossSectionZscore:
    def test_output_shape(self, prices):
        f = momentum(prices)
        z = cross_section_zscore(f)
        assert z.shape == f.shape

    def test_mean_near_zero(self, prices):
        f = momentum(prices)
        z = cross_section_zscore(f)
        row_means = z.mean(axis=1).dropna()
        assert (row_means.abs() < 1e-10).all()

    def test_winsorised_at_3sigma(self, prices):
        f = momentum(prices)
        z = cross_section_zscore(f)
        assert z.max().max() <= 3.01
        assert z.min().min() >= -3.01


# ---------------------------------------------------------------------------
# cross_section_rank
# ---------------------------------------------------------------------------

class TestCrossSectionRank:
    def test_bounds(self, prices):
        f = momentum(prices)
        r = cross_section_rank(f)
        valid = r.stack().dropna()
        assert valid.max() <= 0.5 + 1e-9
        assert valid.min() >= -0.5 - 1e-9

    def test_mean_near_zero(self, prices):
        # For N stocks, sum(rank(pct=True) - 0.5) = 0.5 always.
        # The mean approaches 0 as N grows. With 10 stocks, mean ≈ 0.05.
        f = momentum(prices)
        r = cross_section_rank(f)
        valid_rows = r.dropna(how='all')
        row_means = valid_rows.mean(axis=1).dropna()
        assert (row_means.abs() < 0.1).all()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_returns_dataframe(self, prices):
        f = momentum(prices)
        assert isinstance(f, pd.DataFrame)
        assert f.shape[1] == prices.shape[1]

    def test_skip_period(self, prices):
        # With skip=21, momentum at date t ignores last 21 days
        f_default = momentum(prices, skip=21)
        f_no_skip = momentum(prices, skip=0)
        # They should differ
        diff = (f_default - f_no_skip).abs().sum().sum()
        assert diff > 0

    def test_no_future_leak(self, prices):
        # Factor at date t should not depend on prices strictly after t
        # Use a perturbation well after the lookback period to check
        # that early factor values are unaffected
        t_perturb = 500    # perturb from row 500 onwards
        prices2 = prices.copy()
        prices2.iloc[t_perturb:] *= 1.5
        f  = momentum(prices)
        f2 = momentum(prices2)
        # Factor values well before perturbation must be identical
        safe = 280   # after warmup (252 days), well before perturbation
        diff = (f.iloc[safe:t_perturb - 5] - f2.iloc[safe:t_perturb - 5])
        valid = diff.abs().stack().dropna()
        assert len(valid) > 0, "No valid overlap to compare"
        assert valid.max() < 1e-8


# ---------------------------------------------------------------------------
# Realised volatility
# ---------------------------------------------------------------------------

class TestRealisedVolatility:
    def test_returns_negative(self, prices):
        # low-vol anomaly: we return -vol so high values → low vol
        f = realised_volatility(prices)
        # All non-NaN values should be ≤ 0
        assert (f.dropna(how='all') <= 0).all().all()

    def test_output_index_matches(self, prices):
        f = realised_volatility(prices)
        assert f.index.equals(prices.index)


# ---------------------------------------------------------------------------
# RSI factor
# ---------------------------------------------------------------------------

class TestRSIFactor:
    def test_bounds(self, prices):
        f = rsi_factor(prices)
        # RSI in [0,100] → (rsi/100 - 0.5) in [-0.5, 0.5] → negated in [-0.5, 0.5]
        valid = f.stack().dropna()
        assert (valid.abs() <= 0.5 + 1e-9).all()


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

class TestFactorRegistry:
    def test_all_factors_registered(self):
        expected = {
            'momentum_252_21', 'reversal_21', 'volatility_63', 'rsi_14', 'macd',
        }
        assert expected.issubset(set(FACTOR_REGISTRY.keys()))

    def test_compute_factor_dispatches(self, prices):
        for name in list(FACTOR_REGISTRY.keys())[:5]:
            f = compute_factor(name, prices)
            assert isinstance(f, pd.DataFrame), f"Factor {name} did not return DataFrame"

    def test_unknown_factor_raises(self, prices):
        with pytest.raises(ValueError):
            compute_factor("nonexistent_factor_xyz", prices)


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

class TestInformationCoefficient:
    def test_returns_series(self, prices):
        f = momentum(prices)
        ic = information_coefficient(f, prices.pct_change(), method='spearman')
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0

    def test_ic_in_range(self, prices):
        f = momentum(prices)
        ic = information_coefficient(f, prices.pct_change())
        assert (ic.dropna().abs() <= 1 + 1e-9).all()

    def test_ic_stats_keys(self, prices):
        f = momentum(prices)
        ic = information_coefficient(f, prices.pct_change())
        stats = ic_stats(ic)
        for key in ('ic_mean', 'ic_std', 'ic_ir', 't_stat', 'pct_pos', 'n_periods'):
            assert key in stats, f"Missing key: {key}"

    def test_pearson_vs_spearman(self, prices):
        f = momentum(prices)
        fwd = prices.pct_change()
        ic_p = information_coefficient(f, fwd, method='pearson')
        ic_s = information_coefficient(f, fwd, method='spearman')
        # They should both run without error and differ slightly
        diff = (ic_p - ic_s).abs().mean()
        assert diff >= 0   # always true; just checks no crash


# ---------------------------------------------------------------------------
# Fama-MacBeth
# ---------------------------------------------------------------------------

class TestFamaMacBeth:
    def test_returns_dict(self, prices):
        f = momentum(prices)
        result = fama_macbeth(f, prices.pct_change(), horizon=21)
        assert isinstance(result, dict)

    def test_required_keys(self, prices):
        f = momentum(prices)
        result = fama_macbeth(f, prices.pct_change(), horizon=21)
        for k in ('gamma', 't_stat', 'p_value', 'n_periods'):
            assert k in result, f"Missing key: {k}"

    def test_t_stat_finite(self, prices):
        f = momentum(prices)
        result = fama_macbeth(f, prices.pct_change(), horizon=21)
        assert np.isfinite(result['t_stat'])
