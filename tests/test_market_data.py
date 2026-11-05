"""
Market-data gap handling.

The single most damaging bug this repo had: `get_prices` ended in
`.ffill().bfill()`.

* `bfill()` stamped a ticker's first post-IPO price across every pre-IPO
  date, producing years of *exactly zero* returns.  The low-volatility factor
  ranks a zero-variance series as the most attractive name in the universe,
  so the strategy loaded up on companies that did not yet exist.
* Unbounded `ffill()` carried a delisted name's final quote forward forever,
  so a company going to zero showed no loss at all.

These tests pin the fixed behaviour: leading NaNs survive, fills are bounded,
and the downstream layers exclude what they cannot price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arbitron.data.market import (
    DEFAULT_UNIVERSE,
    SECTOR_UNIVERSES,
    UNIVERSE_CAVEAT,
    MarketData,
    to_returns,
    universe_caveat,
)


@pytest.fixture
def raw_panel() -> pd.DataFrame:
    """A panel with a pre-IPO name, a delisted name, and a holiday gap."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    df = pd.DataFrame(100.0, index=idx, columns=["OLD", "IPO", "DEAD", "GAPPY"])
    df["OLD"] = np.linspace(100, 130, 60)
    df.loc[df.index[:30], "IPO"] = np.nan          # not yet listed
    df.loc[df.index[40:], "DEAD"] = np.nan         # delisted at day 40
    df.loc[df.index[10:13], "GAPPY"] = np.nan      # 3-day feed outage
    return df


@pytest.fixture
def md(tmp_path, monkeypatch, raw_panel) -> MarketData:
    """A MarketData whose network fetch is replaced by the fixture panel."""
    m = MarketData(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(m, "_fetch", lambda *a, **k: raw_panel.copy())
    return m


class TestGapHandling:

    def test_leading_nans_survive(self, md, raw_panel):
        """The regression test named in the audit: leading NaNs must survive."""
        prices = md.get_prices(["OLD", "IPO", "DEAD", "GAPPY"], "2020-01-01", "2020-03-24")
        assert prices["IPO"].iloc[:30].isna().all(), (
            "Pre-IPO dates were back-filled — the low-vol factor will rank this "
            "name most attractive on a run of manufactured zero returns."
        )
        assert prices["IPO"].iloc[30:].notna().all()

    def test_pre_ipo_returns_are_nan_not_a_run_of_zeros(self, md):
        """
        The exact mechanism of the old bug: back-filling made every pre-IPO
        return exactly 0.0, giving the name zero realised volatility.
        """
        prices = md.get_prices(["IPO"], "2020-01-01", "2020-03-24")
        rets = to_returns(prices)["IPO"]
        pre_ipo = rets.iloc[:30]

        assert pre_ipo.isna().all()
        assert pre_ipo.notna().sum() == 0
        # A zero-variance window is what the low-vol factor loved; there must
        # not be enough non-NaN observations to compute one.
        assert pd.isna(pre_ipo.std())

    def test_delisted_name_is_not_carried_forward_forever(self, md):
        prices = md.get_prices(["DEAD"], "2020-01-01", "2020-03-24")
        tail = prices["DEAD"].iloc[40:]

        # At most `FFILL_LIMIT` stale quotes, then honest NaN.
        assert tail.notna().sum() == MarketData.FFILL_LIMIT
        assert pd.isna(tail.iloc[-1])
        assert tail.iloc[MarketData.FFILL_LIMIT:].isna().all()

    def test_short_interior_gaps_are_filled(self, md):
        """Genuine holiday gaps are what the fill limit exists for."""
        prices = md.get_prices(["GAPPY"], "2020-01-01", "2020-03-24")
        assert prices["GAPPY"].iloc[10:13].notna().all()

    def test_fill_limit_is_configurable_and_bounded(self, tmp_path, monkeypatch, raw_panel):
        m = MarketData(cache_dir=str(tmp_path / "c2"), ffill_limit=1)
        monkeypatch.setattr(m, "_fetch", lambda *a, **k: raw_panel.copy())
        prices = m.get_prices(["GAPPY", "DEAD"], "2020-01-01", "2020-03-24")

        # With a 1-day limit only the first day of the 3-day gap is bridged.
        assert prices["GAPPY"].notna().iloc[10]
        assert prices["GAPPY"].isna().iloc[11]
        assert prices["GAPPY"].isna().iloc[12]
        assert prices["DEAD"].iloc[40:].notna().sum() == 1


class TestReturnsHelper:

    def test_does_not_bridge_across_a_missing_price(self):
        s = pd.Series([100.0, np.nan, 110.0], index=pd.bdate_range("2021-01-04", periods=3))
        r = to_returns(s)
        assert pd.isna(r.iloc[1])
        # The 10% move must NOT be reported as a one-day return off the stale
        # price — which is what pct_change's legacy pad-fill would have done.
        assert pd.isna(r.iloc[2])

    def test_matches_pct_change_when_there_are_no_gaps(self):
        s = pd.Series([100.0, 105.0, 102.0], index=pd.bdate_range("2021-01-04", periods=3))
        pd.testing.assert_series_equal(to_returns(s), s.pct_change(fill_method=None))


class TestUniverseCaveats:
    """Audit item 16: state the limitation in the README AND the API response."""

    def test_caveat_declares_survivorship_and_non_point_in_time(self):
        assert UNIVERSE_CAVEAT["point_in_time"] is False
        assert UNIVERSE_CAVEAT["survivorship_bias"] is True
        assert "point-in-time" in UNIVERSE_CAVEAT["headline"]
        assert len(UNIVERSE_CAVEAT["details"]) >= 3

    def test_builtin_universe_gets_the_full_caveat(self):
        c = universe_caveat(universe="Technology")
        assert c["survivorship_bias"] is True
        assert c["universe"] == "Technology"

    def test_caller_supplied_tickers_get_an_honest_unknown(self):
        c = universe_caveat(universe=None, tickers=["AAPL", "MSFT"])
        # We cannot verify someone else's list, and must not claim we can.
        assert c["point_in_time"] is None
        assert c["survivorship_bias"] is None
        assert "cannot verify" in c["headline"]

    def test_universes_are_non_empty_and_deduplicated(self):
        assert len(DEFAULT_UNIVERSE) == len(set(DEFAULT_UNIVERSE))
        for name, tickers in SECTOR_UNIVERSES.items():
            assert len(tickers) == len(set(tickers)), f"{name} has duplicate tickers"
            assert len(tickers) >= 10


class TestCacheSafety:
    """The md5-hashed cache path is a security control — keep it that way."""

    def test_llm_supplied_tickers_cannot_traverse_the_filesystem(self, tmp_path):
        m = MarketData(cache_dir=str(tmp_path / "cache"))
        evil = ["../../../../etc/passwd", "..\\..\\windows\\system32"]
        path = m._cache_path(evil, "2020-01-01", "2020-12-31", "Adj Close")
        assert path.parent == m.cache_dir
        assert path.suffix == ".parquet"
        assert ".." not in str(path.name)

    def test_cache_key_depends_on_every_input(self, tmp_path):
        m = MarketData(cache_dir=str(tmp_path / "cache"))
        base = m._cache_path(["A"], "2020-01-01", "2020-12-31", "Adj Close")
        assert base != m._cache_path(["B"], "2020-01-01", "2020-12-31", "Adj Close")
        assert base != m._cache_path(["A"], "2021-01-01", "2020-12-31", "Adj Close")
        assert base != m._cache_path(["A"], "2020-01-01", "2021-12-31", "Adj Close")
        assert base != m._cache_path(["A"], "2020-01-01", "2020-12-31", "Volume")
