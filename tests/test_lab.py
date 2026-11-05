"""Tests for flagship lab defenses and market foundation signal."""

import numpy as np
import pandas as pd

from arbitron.lab.statistical_defense import (
    audit_strategy,
    estimate_backtest_overfitting,
    probabilistic_sharpe_ratio,
)
from arbitron.models.foundation import MarketFoundationConfig, build_foundation_signal


def test_probabilistic_sharpe_ratio_in_range():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0007, 0.01, 500))
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= psr <= 1.0


def test_pbo_estimator_returns_trials():
    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2020-01-01", periods=260)
    matrix = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(idx), 4)),
        index=idx,
        columns=["a", "b", "c", "d"],
    )
    result = estimate_backtest_overfitting(matrix, n_splits=6)
    assert result["n_trials"] == 4
    assert result["pbo"] is None or 0.0 <= result["pbo"] <= 1.0


def test_strategy_audit_has_gates():
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2020-01-01", periods=320)
    strategy = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    peers = pd.DataFrame({
        "strategy": strategy,
        "peer": rng.normal(0.0001, 0.011, len(idx)),
    }, index=idx)
    report = audit_strategy(strategy, peers, "strategy")
    assert report["strategy_name"] == "strategy"
    assert report["verdict"] in {"deployable", "watchlist", "reject"}
    assert len(report["gates"]) >= 4


def test_foundation_signal_produces_panel(prices):
    cfg = MarketFoundationConfig(
        horizon=21,
        train_window=180,
        refit_freq=63,
        embargo_days=10,
        n_components=3,
        min_train_samples=80,
    )
    result = build_foundation_signal(prices.iloc[:420, :6], cfg)
    assert isinstance(result.signal, pd.DataFrame)
    assert result.diagnostics["asset_features"]
    assert "state_explained_variance" in result.diagnostics
