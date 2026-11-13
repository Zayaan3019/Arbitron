"""
Walk-forward multi-factor composite with purged, embargoed training windows.

A single factor backtest fits nothing: it ranks stocks by a fixed formula.
Combining factors does fit something -- how much weight each factor gets --
and that is where look-ahead creeps in. Weighting by full-sample IC uses
labels from the future of every trade.

Here the weights are re-estimated walk-forward. At a refit on trading day `t`
a factor's weight is its mean cross-sectional rank IC over the dates `s`
whose label is already known and clear of the embargo:

    s + horizon + embargo <= t          (positions in the trading calendar)

The label at `s` is the forward return from `s` to `s + horizon`, so it is
only observable at `s + horizon` (the purge); the embargo adds a further gap
against serial correlation (Lopez de Prado, 2018, ch. 7). The weights are
held until the next refit and applied to that day's cross-sectional z-scores,
so `composite[t]` depends on prices up to `t` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from arbitron.factors.evaluation import forward_returns, information_coefficient
from arbitron.factors.library import cross_section_zscore


@dataclass
class CompositeResult:
    """Output of `walk_forward_composite`."""
    signal:      pd.DataFrame                 # (dates × tickers) composite score
    weights:     pd.DataFrame                 # (refit dates × factors) weights used
    diagnostics: dict = field(default_factory=dict)


def walk_forward_composite(
    factors:      Mapping[str, pd.DataFrame],
    prices:       pd.DataFrame,
    horizon:      int = 21,
    embargo_days: int = 5,
    min_history:  int = 126,
    refit_every:  int = 21,
) -> CompositeResult:
    """
    IC-weighted composite of several factors, fitted walk-forward with a purge
    of `horizon` days and an embargo of `embargo_days` (see module docstring).

    Args:
        factors:      name -> (dates × tickers) raw factor values.
        prices:       (dates × tickers) prices; defines the trading calendar.
        horizon:      label horizon in trading days (the purge length).
        embargo_days: extra gap after the purge.
        min_history:  usable IC observations a factor needs before it gets weight.
        refit_every:  trading days between weight re-estimates.

    Returns:
        CompositeResult. Dates before the first refit with enough history have
        no signal (NaN), rather than a signal built from too little data.
    """
    if not factors:
        raise ValueError("walk_forward_composite needs at least one factor")
    dates = prices.sort_index().index
    names = list(factors)

    fwd = forward_returns(prices, horizon)
    ic = pd.DataFrame(
        {name: information_coefficient(factors[name], fwd) for name in names}
    ).reindex(dates)
    zs = {name: cross_section_zscore(factors[name].reindex(dates)) for name in names}

    lag = horizon + embargo_days
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    current: pd.Series | None = None
    signal = pd.DataFrame(np.nan, index=dates, columns=prices.columns)

    for pos, date in enumerate(dates):
        if pos % max(1, refit_every) == 0:
            usable_end = pos - lag + 1          # rows [0, usable_end) have s + lag <= pos
            if usable_end > 0:
                past = ic.iloc[:usable_end]
                counts = past.notna().sum()
                means = past.mean().where(counts >= min_history)
                if means.notna().any() and means.abs().sum() > 0:
                    current = (means / means.abs().sum()).fillna(0.0)
                    weight_rows[date] = current
        if current is None:
            continue
        num = None
        den = None
        for name in names:
            w = float(current.get(name, 0.0))
            if w == 0.0:
                continue
            row = zs[name].loc[date]
            contrib = row * w
            present = row.notna().astype(float) * abs(w)
            num = contrib.fillna(0.0) if num is None else num + contrib.fillna(0.0)
            den = present if den is None else den + present
        if num is not None:
            signal.loc[date] = (num / den.replace(0.0, np.nan)).reindex(prices.columns)

    weights = pd.DataFrame(weight_rows).T.reindex(columns=names)
    return CompositeResult(
        signal=signal.dropna(how="all"),
        weights=weights,
        diagnostics={
            "factors": names,
            "horizon": horizon,
            "embargo_days": embargo_days,
            "purge_plus_embargo_days": lag,
            "refit_every": refit_every,
            "min_history": min_history,
            "n_refits": int(len(weights)),
            "first_signal_date": (
                str(signal.dropna(how="all").index[0].date())
                if signal.notna().any().any() else None
            ),
        },
    )
