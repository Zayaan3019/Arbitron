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
DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "UNH", "V", "XOM", "PG", "HD", "MA", "BAC", "ABBV",
    "KO", "MRK", "PEP", "LLY", "AVGO", "COST", "MCD", "TMO", "WMT",
    "CSCO", "CVX", "ACN",
]


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

    def __init__(
        self,
        cache_dir: str = "data/cache",
        refresh_days: int = 1,
    ) -> None:
        self.cache_dir    = Path(cache_dir)
        self.refresh_days = refresh_days
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

        NaN values are forward-filled then backward-filled (handles
        trading-day gaps for non-US-listed tickers).
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        raw = self._fetch(tickers, start, end, field)
        return raw.ffill().bfill()

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

        Args:
            log:  If True return log returns; else simple returns.
            freq: Resample frequency — 'D' daily, 'W' weekly, 'ME' month-end.
        """
        prices = self.get_prices(tickers, start, end)
        if freq != "D":
            prices = prices.resample(freq).last()
        if log:
            return np.log(prices / prices.shift(1)).dropna(how="all")
        return prices.pct_change().dropna(how="all")

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
