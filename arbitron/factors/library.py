"""
Cross-sectional factor library.

All factors return a (dates × tickers) DataFrame of raw factor values.
Use `cross_section_zscore` or `cross_section_rank` to normalise before use.

Factor naming convention:  {type}_{lookback}
  e.g. momentum_252_21 = 252-day minus 21-day price return (12-1 momentum)
       reversal_21     = 21-day reversal
       volatility_63   = 63-day realised volatility

Every factor in this module is reachable through `FACTOR_REGISTRY`; there is
no unreachable factor code here.  Factors take a price panel that may contain
genuine NaNs (pre-IPO, post-delisting) and must propagate them rather than
inventing a value — `MarketData.get_prices` no longer back-fills.

References:
  - Asness, Moskowitz & Pedersen (2013) — momentum
  - RSI: Wilder (1978)
  - MACD: Appel (1979)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_change(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each row (date) to zero mean, unit variance across tickers.
    Robust to outliers: winsorise at ±3σ before z-scoring.
    """
    def _row_zscore(row: pd.Series) -> pd.Series:
        mu, sigma = row.mean(), row.std()
        if sigma == 0 or np.isnan(sigma):
            return row * 0
        z = (row - mu) / sigma
        return z.clip(-3, 3)
    return df.apply(_row_zscore, axis=1)


def cross_section_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-normalise each row to [-0.5, +0.5].
    This is distribution-free and more robust than z-scoring for fat tails.
    """
    def _row_rank(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return row
        ranks = valid.rank(pct=True) - 0.5  # [-0.5, +0.5]
        return ranks.reindex(row.index)      # NaN for tickers not in valid
    return df.apply(_row_rank, axis=1)


# ---------------------------------------------------------------------------
# Momentum factors
# ---------------------------------------------------------------------------

def momentum(
    prices:   pd.DataFrame,
    lookback: int = 252,
    skip:     int = 21,
) -> pd.DataFrame:
    """
    Price momentum: return over [t-lookback, t-skip].

    The `skip` parameter excludes the most recent month to avoid the
    short-term reversal effect that contaminates pure momentum.

    Args:
        lookback: Formation period in trading days.
        skip:     Days to skip (typically ~21 = 1 month).

    Returns:
        (dates × tickers) factor DataFrame.
    """
    ret_full  = prices.shift(skip) / prices.shift(lookback) - 1
    return ret_full


def short_term_reversal(
    prices:   pd.DataFrame,
    lookback: int = 21,
) -> pd.DataFrame:
    """
    Short-term price reversal: negative of 1-month return.

    Stocks that have recently risen tend to underperform (and vice versa)
    at the 1-month horizon.
    """
    return -(prices / prices.shift(lookback) - 1)


# ---------------------------------------------------------------------------
# Volatility / risk factors
# ---------------------------------------------------------------------------

def realised_volatility(
    prices:   pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Annualised realised volatility over a rolling window.

    Low-vol factor: assign negative sign so low vol → high factor value.
    """
    ret   = _pct_change(prices)
    vol   = ret.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(252)
    return -vol   # low-vol anomaly: low vol stocks outperform


# ---------------------------------------------------------------------------
# Technical factors
# ---------------------------------------------------------------------------

def rsi_factor(
    prices:   pd.DataFrame,
    period:   int = 14,
) -> pd.DataFrame:
    """
    Relative Strength Index, normalised to [-0.5, +0.5].

    Low RSI = oversold → positive factor (contrarian signal).
    High RSI = overbought → negative factor.
    Returns negative RSI so the factor is positive for oversold stocks.
    """
    delta   = prices.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return -(rsi / 100 - 0.5)   # centre around 0, negate for contrarian


def macd_signal(
    prices:    pd.DataFrame,
    fast:      int = 12,
    slow:      int = 26,
    signal:    int = 9,
) -> pd.DataFrame:
    """
    MACD histogram as a factor.

    Positive histogram (MACD above signal line) → positive factor.
    """
    ema_fast   = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow   = prices.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line   # histogram


def bollinger_position(
    prices:   pd.DataFrame,
    lookback: int = 20,
    n_std:    float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Band position: (price - lower band) / bandwidth.

    Returns a value in [0, 1] where low = near lower band (oversold).
    Negated so oversold → positive factor.
    """
    mid    = prices.rolling(lookback, min_periods=5).mean()
    std    = prices.rolling(lookback, min_periods=5).std()
    upper  = mid + n_std * std
    lower  = mid - n_std * std
    bw     = upper - lower
    pos    = (prices - lower) / bw.replace(0, np.nan)
    return -(pos - 0.5)   # centre and negate (contrarian)


def price_to_high(
    prices: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    52-week price-to-high ratio: price / rolling 252-day max.

    Low price-to-high → stock is beaten-down (potential reversal or value).
    Returns negated ratio so beaten-down stocks have positive factor.
    """
    rolling_max = prices.rolling(lookback, min_periods=lookback // 2).max()
    return -(prices / rolling_max)


# ---------------------------------------------------------------------------
# Value / fundamental-proxy factors (price-based only — no fundamental data)
# ---------------------------------------------------------------------------

def price_acceleration(
    prices:   pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Second derivative of price: recent momentum minus past momentum.

    Captures acceleration in price trend.
    """
    recent = momentum(prices, lookback=lookback,     skip=5)
    past   = momentum(prices, lookback=lookback * 2, skip=lookback)
    return recent - past


# ---------------------------------------------------------------------------
# Factor registry (name → callable)
# ---------------------------------------------------------------------------

FACTOR_REGISTRY: dict[str, callable] = {
    "momentum_252_21":   lambda p, **kw: momentum(p, 252, 21),
    "momentum_126_21":   lambda p, **kw: momentum(p, 126, 21),
    "momentum_63_5":     lambda p, **kw: momentum(p, 63,  5),
    "reversal_21":       lambda p, **kw: short_term_reversal(p, 21),
    "reversal_5":        lambda p, **kw: short_term_reversal(p, 5),
    "volatility_63":     lambda p, **kw: realised_volatility(p, 63),
    "volatility_21":     lambda p, **kw: realised_volatility(p, 21),
    "rsi_14":            lambda p, **kw: rsi_factor(p, 14),
    "rsi_7":             lambda p, **kw: rsi_factor(p, 7),
    "macd":              lambda p, **kw: macd_signal(p),
    "bollinger":         lambda p, **kw: bollinger_position(p),
    "price_to_high_252": lambda p, **kw: price_to_high(p, 252),
    "acceleration_63":   lambda p, **kw: price_acceleration(p, 63),
}


def compute_factor(
    factor_name: str,
    prices:      pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Compute a named factor from the registry."""
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError(
            f"Unknown factor '{factor_name}'. "
            f"Available: {sorted(FACTOR_REGISTRY)}"
        )
    return FACTOR_REGISTRY[factor_name](prices, **kwargs)


def compute_all_factors(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute every factor in the registry."""
    return {name: fn(prices) for name, fn in FACTOR_REGISTRY.items()}
