"""
Market data fetching and caching.

Downloads OHLCV price data from Yahoo Finance (free, no API key required).
Results are cached as parquet files to avoid repeated downloads.

Usage:
    from arbitron.data.market import MarketData
    md = MarketData(cache_dir="data/cache")
    prices = md.get_prices(["AAPL", "MSFT", "GOOGL"], start="2018-01-01")
    returns = md.get_returns(["AAPL", "MSFT"], start="2018-01-01")
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# S&P 500 sector mini-universes (20 liquid names per sector)
# ---------------------------------------------------------------------------

SECTOR_UNIVERSES: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO",
        "AMD", "INTC", "TXN", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
        "ADI", "NXPI", "ON",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "ABT", "TMO", "DHR", "BMY",
        "AMGN", "PFE", "ISRG", "SYK", "MDT", "ELV", "HCA", "CVS", "GILD",
        "VRTX", "REGN",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB", "PNC",
        "TFC", "COF", "SCHW", "MCO", "ICE", "CME", "SPGI", "FIS", "PYPL",
        "V",
    ],
    "Industrials": [
        "GE", "HON", "UNP", "CAT", "RTX", "DE", "LMT", "NOC", "GD", "BA",
        "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "CARR", "OTIS", "FDX",
        "UPS",
    ],
    "ConsumerDisc": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "CMG",
        "BKNG", "MAR", "HLT", "YUM", "ROST", "TJX", "DRI", "DHI", "LEN",
        "PHM", "GM",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
        "EOG", "DVN", "HAL", "BKR", "FANG", "APA", "MRO", "HES", "OKE",
        "WMB", "KMI",
    ],
}

# Compact default universe — diverse, liquid, goes back to 2010
#
# WARNING — this list is NOT point-in-time.  See UNIVERSE_CAVEAT below.
DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "UNH", "V", "XOM", "PG", "HD", "MA", "BAC", "ABBV",
    "KO", "MRK", "PEP", "LLY", "AVGO", "COST", "MCD", "TMO", "WMT",
    "CSCO", "CVX", "ACN",
]


# ---------------------------------------------------------------------------
# Data limitations — surfaced in the README AND in every API response that
# is derived from these universes.  Do not delete without shipping a
# point-in-time membership table first.
# ---------------------------------------------------------------------------

UNIVERSE_CAVEAT: dict[str, object] = {
    "point_in_time": False,
    "survivorship_bias": True,
    "headline": (
        "Universes are TODAY's index members, not point-in-time membership. "
        "Every backtest run on them is upward-biased."
    ),
    "details": [
        "Survivorship bias: names that were delisted, acquired or bankrupted "
        "during the backtest window are absent, so the realised loss from "
        "holding them is never charged to the strategy.",
        "Ex-post selection bias: DEFAULT_UNIVERSE is a hand-picked list of "
        "mega-caps that are large TODAY. Membership itself encodes the "
        "forward return being measured.",
        "Sector universes carry the same two biases within their sector.",
        "Consequence: treat absolute Sharpe/CAGR as an upper bound. "
        "Cross-factor COMPARISONS on the same universe are far less affected "
        "because every candidate inherits the same bias.",
    ],
    "remediation": (
        "Supply an explicit point-in-time ticker list via the `tickers` field "
        "to bypass the built-in universes entirely."
    ),
}


def universe_caveat(universe: Optional[str] = None, tickers: Optional[list[str]] = None) -> dict:
    """
    Return the data-limitation block that belongs on any response derived
    from a built-in universe.

    When the caller supplied their own `tickers`, the built-in survivorship
    caveat does not apply to the SELECTION (though it may still apply to the
    caller's own list, which we cannot verify).
    """
    if tickers:
        return {
            "point_in_time": None,
            "survivorship_bias": None,
            "headline": (
                "Caller-supplied ticker list. Arbitron cannot verify whether it "
                "is point-in-time; if it was chosen with hindsight the results "
                "carry the same selection bias as the built-in universes."
            ),
            "details": [],
            "remediation": UNIVERSE_CAVEAT["remediation"],
        }
    caveat = dict(UNIVERSE_CAVEAT)
    caveat["universe"] = universe or "SP500"
    return caveat


# ---------------------------------------------------------------------------
# Market data manager
# ---------------------------------------------------------------------------

class MarketData:
    """
    Thin wrapper around yfinance with file-based caching.

    All data is stored as Parquet files in `cache_dir`.  Cache is keyed
    by (tickers, start, end) — stale entries are refreshed automatically
    when the requested end date is in the past and the cached file is
    older than `refresh_days`.
    """

    #: Maximum number of consecutive trading days a stale price may be carried
    #: forward.  Sized for exchange holidays and short feed outages only.
    FFILL_LIMIT: int = 5

    def __init__(
        self,
        cache_dir: str = "data/cache",
        refresh_days: int = 1,
        ffill_limit: int = FFILL_LIMIT,
    ) -> None:
        self.cache_dir    = Path(cache_dir)
        self.refresh_days = refresh_days
        self.ffill_limit  = ffill_limit
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: list[str],
        start:   str = "2015-01-01",
        end:     Optional[str] = None,
        field:   str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Return a (dates × tickers) DataFrame of adjusted closing prices.

        Gap handling — deliberately conservative, and load-bearing:

        * Interior gaps are forward-filled for at most ``ffill_limit`` (5)
          trading days.  That covers exchange holidays and short feed
          outages and nothing else.
        * **Leading NaNs are preserved.**  A ticker that had not yet listed
          must stay NaN so every downstream layer excludes it.  Back-filling
          (the previous behaviour) stamped the first post-IPO price across
          all pre-IPO dates, manufacturing years of *exactly zero* returns —
          which the low-vol factor then ranks as the single most attractive
          name in the universe.
        * **Trailing NaNs are preserved** past the fill limit.  Unbounded
          forward-fill carries a delisted name's last quote forward forever,
          so a company going to zero shows no loss at all.

        Callers must therefore expect NaNs.  `run_factor_backtest`
        renormalises over the names that are actually live on each date and
        reports the rest via `BacktestResult.dropped_names`.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        raw = self._fetch(tickers, start, end, field)
        return raw.ffill(limit=self.ffill_limit)

    def get_returns(
        self,
        tickers:   list[str],
        start:     str = "2015-01-01",
        end:       Optional[str] = None,
        log:       bool = False,
        freq:      str = "D",
    ) -> pd.DataFrame:
        """
        Return (dates × tickers) simple or log returns, optionally resampled.

        Missing prices stay missing: returns spanning a NaN price are NaN
        rather than being silently bridged across the gap.

        Args:
            log:  If True return log returns; else simple returns.
            freq: Resample frequency — 'D' daily, 'W' weekly, 'ME' month-end.
        """
        prices = self.get_prices(tickers, start, end)
        if freq != "D":
            prices = prices.resample(freq).last()
        if log:
            return np.log(prices / prices.shift(1)).dropna(how="all")
        return to_returns(prices).dropna(how="all")

    def get_volume(
        self,
        tickers: list[str],
        start:   str = "2015-01-01",
        end:     Optional[str] = None,
    ) -> pd.DataFrame:
        """Return daily trading volume."""
        end = end or datetime.today().strftime("%Y-%m-%d")
        return self._fetch(tickers, start, end, "Volume")

    def get_universe(self, sector: Optional[str] = None, n: int = 20) -> list[str]:
        """Return a list of liquid tickers for a given sector or the default universe."""
        if sector is not None:
            candidates = SECTOR_UNIVERSES.get(sector, DEFAULT_UNIVERSE)
        else:
            candidates = DEFAULT_UNIVERSE
        return candidates[:n]

    # ------------------------------------------------------------------
    # Internal caching
    # ------------------------------------------------------------------

    def _cache_path(self, tickers: list[str], start: str, end: str, field: str) -> Path:
        key = f"{'_'.join(sorted(tickers))}_{start}_{end}_{field}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{digest}.parquet"

    def _is_stale(self, path: Path, end: str) -> bool:
        if not path.exists():
            return True
        try:
            if pd.read_parquet(path).empty:
                return True   # empty cache (e.g. from a failed download)
        except Exception:
            return True
        age_days = (datetime.today() - datetime.fromtimestamp(path.stat().st_mtime)).days
        return age_days >= self.refresh_days

    def _fetch(
        self,
        tickers: list[str],
        start:   str,
        end:     str,
        field:   str,
    ) -> pd.DataFrame:
        path = self._cache_path(tickers, start, end, field)

        if not self._is_stale(path, end):
            return pd.read_parquet(path)

        try:
            import time
            import yfinance as yf  # lazy import — not in base env

            raw = None
            for attempt in range(3):
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                )
                if not raw.empty:
                    break
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))  # 10s, 20s backoff

            # yfinance returns MultiIndex when len(tickers) > 1
            if isinstance(raw.columns, pd.MultiIndex):
                data = raw[field]
            else:
                data = raw[[field]].rename(columns={field: tickers[0]})

            # Ensure consistent column ordering
            available = [t for t in tickers if t in data.columns]
            data = data[available]

            data.to_parquet(path)
            return data

        except ImportError:
            raise RuntimeError(
                "yfinance is required for market data: pip install yfinance"
            )
        except Exception as e:
            if path.exists():
                return pd.read_parquet(path)
            raise RuntimeError(f"Failed to fetch data: {e}") from e


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def to_returns(prices: pd.DataFrame | pd.Series, periods: int = 1):
    """
    Simple returns with **no implicit gap filling**.

    `pandas.pct_change` historically pad-filled before differencing, which
    silently bridges a missing price and reports a multi-day move as a
    one-day move.  Since `MarketData.get_prices` now preserves genuine NaNs
    (pre-IPO, post-delisting), every return computation in the repo must opt
    out of that padding explicitly.  Use this helper instead of calling
    `.pct_change()` directly.
    """
    return prices.pct_change(periods, fill_method=None)


def align_dates(
    *frames: pd.DataFrame,
    method: str = "inner",
) -> tuple[pd.DataFrame, ...]:
    """
    Align multiple DataFrames to a common date index.

    Args:
        method: 'inner' (intersection) or 'outer' (union with NaN fill).
    """
    if not frames:
        return frames
    idx = frames[0].index
    for df in frames[1:]:
        if method == "inner":
            idx = idx.intersection(df.index)
        else:
            idx = idx.union(df.index)
    return tuple(df.reindex(idx) for df in frames)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score normalisation."""
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std().replace(0, np.nan)


def market_returns(
    tickers: list[str],
    returns: pd.DataFrame,
) -> pd.Series:
    """Equal-weight market return from `tickers`."""
    cols = [t for t in tickers if t in returns.columns]
    return returns[cols].mean(axis=1).rename("market")
