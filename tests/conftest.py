"""
Shared pytest fixtures for Arbitron tests.
"""

import numpy as np
import pandas as pd
import pytest


def _make_prices(n_stocks: int = 5, n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic log-normal price series."""
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(n_stocks)]
    daily_ret = rng.normal(0.0003, 0.015, size=(n_days, n_stocks))
    prices = 100 * np.exp(np.cumsum(daily_ret, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=cols)


@pytest.fixture
def prices() -> pd.DataFrame:
    return _make_prices(n_stocks=10, n_days=600)


@pytest.fixture
def small_prices() -> pd.DataFrame:
    return _make_prices(n_stocks=5, n_days=300)


@pytest.fixture
def returns(prices) -> pd.DataFrame:
    return prices.pct_change().dropna()


@pytest.fixture
def market_ret(returns) -> pd.Series:
    return returns.mean(axis=1).rename("market")
