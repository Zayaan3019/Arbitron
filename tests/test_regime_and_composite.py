"""
Regime-gated entries, the walk-forward composite and the purged Ridge folds.

The causality tests share one idea: change the data AFTER some date T and
check that nothing computed for dates up to T moves. If it moves, the code
looked at the future.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from arbitron.backtest.engine import BacktestConfig, run_factor_backtest
from arbitron.factors.composite import walk_forward_composite
from arbitron.factors.library import momentum, realised_volatility
from arbitron.models.foundation import purged_date_splits
from arbitron.models.regime import hmm_entry_gate

needs_hmm = pytest.mark.skipif(
    importlib.util.find_spec("hmmlearn") is None,
    reason="optional extra: pip install arbitron[hmm]",
)


def _shock_after(prices: pd.DataFrame, cut: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """Same prices up to `cut`; an unrelated random path after it."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    after = out.index > cut
    noise = np.exp(np.cumsum(rng.normal(0.0, 0.03, size=(after.sum(), out.shape[1])), axis=0))
    out.loc[after] = out.loc[cut].values * noise
    return out


# ---------------------------------------------------------------------------
# Entry gate in the backtest engine
# ---------------------------------------------------------------------------

class TestEntryGate:
    def test_open_gate_matches_ungated(self, prices):
        factor = momentum(prices)
        cfg = BacktestConfig(transaction_cost=0.001)
        base = run_factor_backtest(factor, prices, cfg)
        gated = run_factor_backtest(
            factor, prices, cfg, entry_gate=pd.Series(True, index=prices.index),
        )
        pd.testing.assert_series_equal(base.returns, gated.returns, check_names=False)
        assert gated.n_gated_off == 0

    def test_closed_gate_is_flat(self, prices):
        factor = momentum(prices)
        gated = run_factor_backtest(
            factor, prices, entry_gate=pd.Series(False, index=prices.index),
        )
        assert gated.n_rebalances == 0
        assert gated.n_gated_off > 0
        assert (gated.returns == 0.0).all()

    def test_nan_gate_counts_as_closed(self, prices):
        factor = momentum(prices)
        gate = pd.Series(np.nan, index=prices.index, dtype=object)
        gated = run_factor_backtest(factor, prices, entry_gate=gate)
        assert gated.n_rebalances == 0

    def test_dates_before_the_gate_starts_are_closed(self, prices):
        factor = momentum(prices)
        late = prices.index[400]
        gate = pd.Series(True, index=prices.index[prices.index >= late])
        gated = run_factor_backtest(factor, prices, entry_gate=gate)
        traded = gated.returns[gated.returns != 0.0]
        assert not traded.empty
        assert traded.index.min() > late

    def test_exit_and_reentry_both_pay_costs(self, prices):
        """Close the gate for one month in the middle: the period after the
        close pays the exit, the period after the reopen pays full entry."""
        factor = momentum(prices)
        cost = 0.01
        cfg = BacktestConfig(transaction_cost=cost)
        base = run_factor_backtest(factor, prices, cfg)
        rebals = base.weights.index
        mid = len(rebals) // 2
        closed_at, reopen_at = rebals[mid], rebals[mid + 1]
        gate = pd.Series(True, index=prices.index)
        gate.loc[closed_at:reopen_at - pd.Timedelta(days=1)] = False
        gated = run_factor_backtest(factor, prices, cfg, entry_gate=gate)
        assert gated.n_gated_off == 1

        prev_w = base.weights.loc[rebals[mid - 1]].dropna()
        flat_day = gated.returns.index[gated.returns.index > closed_at][0]
        assert gated.returns.loc[flat_day] == pytest.approx(-cost * prev_w.abs().sum())

        reentry_day = gated.returns.index[gated.returns.index > reopen_at][0]
        w = base.weights.loc[reopen_at].dropna()
        # Same holdings as the ungated run on that day, minus a full build
        # instead of the ungated run's partial rebalance.
        base_prev = base.weights.loc[closed_at].reindex(w.index).fillna(0.0)
        base_cost = cost * (w - base_prev).abs().sum()
        gap = base.returns.loc[reentry_day] + base_cost - gated.returns.loc[reentry_day]
        assert gap == pytest.approx(cost * w.abs().sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# HMM gate
# ---------------------------------------------------------------------------

@needs_hmm
class TestHmmEntryGate:
    def test_shape_and_types(self, prices):
        gate = hmm_entry_gate(prices, min_train=200, refit_freq=63)
        assert list(gate.columns) == ["p_risk_off", "allow"]
        assert gate.index.min() > prices.index[199]
        assert gate["p_risk_off"].between(0.0, 1.0).all()
        assert gate["allow"].dtype == bool

    def test_gate_is_causal(self, prices):
        cut = prices.index[450]
        a = hmm_entry_gate(prices, min_train=200, refit_freq=63)
        b = hmm_entry_gate(_shock_after(prices, cut), min_train=200, refit_freq=63)
        up_to = a.index[a.index <= cut]
        pd.testing.assert_series_equal(
            a.loc[up_to, "p_risk_off"], b.loc[up_to, "p_risk_off"], atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Walk-forward composite
# ---------------------------------------------------------------------------

class TestWalkForwardComposite:
    @pytest.fixture
    def factors(self, prices):
        return {"mom": momentum(prices), "vol": realised_volatility(prices)}

    def test_no_signal_before_enough_history(self, prices, factors):
        res = walk_forward_composite(factors, prices, horizon=21, embargo_days=5, min_history=126)
        assert not res.signal.empty
        first = res.signal.index[0]
        # The first weights need min_history IC rows that are already
        # observable: at least min_history + horizon + embargo trading days in.
        assert prices.index.get_loc(first) >= 126 + 21 + 5 - 1

    def test_weights_are_normalised(self, prices, factors):
        res = walk_forward_composite(factors, prices)
        sums = res.weights.abs().sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_composite_is_causal(self, prices):
        cut = prices.index[420]
        shocked = _shock_after(prices, cut)
        a = walk_forward_composite(
            {"mom": momentum(prices), "vol": realised_volatility(prices)}, prices,
        )
        b = walk_forward_composite(
            {"mom": momentum(shocked), "vol": realised_volatility(shocked)}, shocked,
        )
        up_to = a.signal.index[a.signal.index <= cut]
        assert len(up_to) > 50
        pd.testing.assert_frame_equal(a.signal.loc[up_to], b.signal.loc[up_to], atol=1e-12)
        w_up_to = a.weights.index[a.weights.index <= cut]
        pd.testing.assert_frame_equal(a.weights.loc[w_up_to], b.weights.loc[w_up_to], atol=1e-12)

    def test_full_sample_ic_would_differ(self, prices, factors):
        """Sanity check that the causality test has teeth: the purge is what
        keeps the weights fixed, so removing it must change them."""
        cut = prices.index[420]
        shocked = _shock_after(prices, cut)
        fs = {"mom": momentum(shocked), "vol": realised_volatility(shocked)}
        leaky_a = walk_forward_composite(factors, prices, horizon=21, embargo_days=-21)
        leaky_b = walk_forward_composite(fs, shocked, horizon=21, embargo_days=-21)
        idx = leaky_a.weights.index[
            (leaky_a.weights.index <= cut) & (leaky_a.weights.index > prices.index[400])
        ]
        assert len(idx)
        assert not np.allclose(leaky_a.weights.loc[idx], leaky_b.weights.loc[idx])

    def test_needs_a_factor(self, prices):
        with pytest.raises(ValueError):
            walk_forward_composite({}, prices)


# ---------------------------------------------------------------------------
# Purged date folds for the Ridge penalty
# ---------------------------------------------------------------------------

class TestPurgedDateSplits:
    @pytest.fixture
    def panel_dates(self):
        days = pd.bdate_range("2021-01-01", periods=300)
        return np.repeat(days.values, 8)   # 8 tickers per date

    def test_validation_follows_training_with_a_gap(self, panel_dates):
        uniq = np.unique(panel_dates)
        folds = purged_date_splits(panel_dates, n_splits=4, purge=21)
        assert len(folds) == 4
        for train, valid in folds:
            t_last = np.searchsorted(uniq, panel_dates[train].max())
            v_first = np.searchsorted(uniq, panel_dates[valid].min())
            assert v_first - t_last > 21

    def test_dates_never_straddle(self, panel_dates):
        for train, valid in purged_date_splits(panel_dates, n_splits=4, purge=21):
            assert not set(panel_dates[train]) & set(panel_dates[valid])
            # A validation date carries all of its tickers.
            counts = pd.Series(panel_dates[valid]).value_counts()
            assert (counts == 8).all()

    def test_too_little_data_falls_back(self):
        dates = np.repeat(pd.bdate_range("2021-01-01", periods=6).values, 3)
        assert purged_date_splits(dates, n_splits=4, purge=21) == []


# ---------------------------------------------------------------------------
# Research lab wiring
# ---------------------------------------------------------------------------

@needs_hmm
class TestLabVariants:
    def test_lab_adds_gated_and_composite_candidates(self, prices):
        from arbitron.lab.research_lab import run_flagship_research_lab

        out = run_flagship_research_lab(
            prices,
            factor_names=["momentum_126_21", "volatility_63", "reversal_21"],
            include_foundation=False,
        )
        kinds = {c["kind"] for c in out["candidates"] if "error" not in c}
        assert "factor_regime_gated" in kinds
        assert "walk_forward_composite" in kinds
        assert out["regime_model"]["enabled"] is True
        # Every variant counts as a trial for the deflated Sharpe.
        names = [c["name"] for c in out["candidates"] if "error" not in c]
        assert "momentum_126_21+hmm_gate" in names
