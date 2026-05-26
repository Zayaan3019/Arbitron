"""
Factor evaluation: Information Coefficient, ICIR, Fama-MacBeth regression.

The key question in factor research is: does this factor predict future returns?
We answer it two ways:

  1. Information Coefficient (IC): Spearman rank correlation between
     factor values at time t and returns over [t, t+h].  High IC = predictive.

  2. Fama-MacBeth (1973): Cross-sectional OLS of returns on factor values,
     run each period.  Time-series mean of coefficients = factor risk premium.
     Newey-West correction for serial correlation in t-statistics.

References:
  - Grinold & Kahn (1999) "Active Portfolio Management"
  - Fama & MacBeth (1973) Journal of Political Economy
  - Newey & West (1987) Econometrica
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Forward returns helper
# ---------------------------------------------------------------------------

def forward_returns(
    prices:   pd.DataFrame,
    horizon:  int = 21,
) -> pd.DataFrame:
    """
    Compute h-day ahead simple returns for use as IC targets.

    Returns at date t = price[t+h] / price[t] - 1
    (shifted backwards so they align with factor values at date t).
    """
    fwd = prices.shift(-horizon) / prices - 1
    return fwd


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(
    factor:      pd.DataFrame,
    fwd_ret:     pd.DataFrame,
    method:      str = "spearman",
) -> pd.Series:
    """
    Cross-sectional IC for each date.

    IC[t] = correlation(factor[t, :], fwd_ret[t, :])

    A well-known factor should have |IC| > 0.02 consistently.

    Args:
        factor:  (dates × tickers) factor values.
        fwd_ret: (dates × tickers) forward returns (same shape).
        method:  'spearman' (rank) or 'pearson' (linear).

    Returns:
        (dates,) IC time series.
    """
    # Align
    common_dates = factor.index.intersection(fwd_ret.index)
    common_ticks = factor.columns.intersection(fwd_ret.columns)
    f = factor.loc[common_dates, common_ticks]
    r = fwd_ret.loc[common_dates, common_ticks]

    ic_vals = []
    for date in common_dates:
        fi = f.loc[date].dropna()
        ri = r.loc[date].dropna()
        common = fi.index.intersection(ri.index)
        if len(common) < 5:
            ic_vals.append(np.nan)
            continue
        fi, ri = fi.loc[common].values, ri.loc[common].values
        if method == "spearman":
            rho, _ = stats.spearmanr(fi, ri)
        else:
            rho, _ = stats.pearsonr(fi, ri)
        ic_vals.append(rho)

    return pd.Series(ic_vals, index=common_dates, name="IC")


def ic_stats(ic: pd.Series) -> dict[str, float]:
    """
    Summary statistics for an IC time series.

    Returns:
        ic_mean:    Mean IC.
        ic_std:     Std of IC.
        ic_ir:      IC / std(IC) * sqrt(252/rebalance_days) — annualised.
        t_stat:     t-statistic testing H0: IC_mean = 0.
        pct_pos:    Fraction of periods with IC > 0.
        ann_ic_ir:  Annualised ICIR assuming monthly rebalancing.
    """
    ic = ic.dropna()
    if len(ic) < 3:
        return {k: np.nan for k in
                ("ic_mean", "ic_std", "ic_ir", "t_stat", "pct_pos")}

    ic_mean = ic.mean()
    ic_std  = ic.std()
    ic_ir   = ic_mean / ic_std if ic_std > 0 else np.nan
    t_stat  = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan
    pct_pos = (ic > 0).mean()

    return {
        "ic_mean":    round(ic_mean, 4),
        "ic_std":     round(ic_std,  4),
        "ic_ir":      round(ic_ir,   4),
        "t_stat":     round(t_stat,  4),
        "pct_pos":    round(pct_pos, 4),
        "n_periods":  len(ic),
    }


# ---------------------------------------------------------------------------
# Fama-MacBeth regression
# ---------------------------------------------------------------------------

def fama_macbeth(
    factor:       pd.DataFrame,
    returns:      pd.DataFrame,
    horizon:      int = 21,
    newey_west_lags: int = 4,
) -> dict[str, float]:
    """
    Fama-MacBeth (1973) cross-sectional regression.

    Each period t: run OLS  r_{i,t+h} = α_t + β_t * f_{i,t} + ε_{i,t}
    Collect β_t series; test whether mean(β_t) ≠ 0.

    Newey-West correction with `newey_west_lags` lags accounts for
    serial correlation in estimated betas (from overlapping returns).

    Returns:
        gamma:         Mean cross-sectional slope (factor risk premium).
        t_stat:        Newey-West corrected t-statistic.
        t_stat_naive:  Naive t-stat (ignores serial correlation).
        p_value:       Two-sided p-value from NW t-stat.
        n_periods:     Number of cross-sections used.
        r_squared_mean: Mean cross-sectional R².
    """
    fwd_ret = forward_returns(returns, horizon)

    common_dates = factor.index.intersection(fwd_ret.index)
    common_ticks = factor.columns.intersection(fwd_ret.columns)
    f = factor.loc[common_dates, common_ticks]
    r = fwd_ret.loc[common_dates, common_ticks]

    betas: list[float] = []
    r2s:   list[float] = []

    for date in common_dates:
        fi = f.loc[date].dropna()
        ri = r.loc[date].dropna()
        common = fi.index.intersection(ri.index)
        if len(common) < 5:
            continue
        fi_v = fi.loc[common].values
        ri_v = ri.loc[common].values

        # Add intercept
        X = np.column_stack([np.ones(len(fi_v)), fi_v])
        try:
            coef, res, *_ = np.linalg.lstsq(X, ri_v, rcond=None)
            betas.append(coef[1])
            if len(ri_v) > 2:
                ss_tot = np.var(ri_v) * len(ri_v)
                ss_res = (res[0] if len(res) > 0 else np.sum((ri_v - X @ coef) ** 2))
                r2s.append(max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
        except np.linalg.LinAlgError:
            continue

    if len(betas) < 3:
        return {k: np.nan for k in
                ("gamma", "t_stat", "t_stat_naive", "p_value", "n_periods", "r_squared_mean")}

    beta_series = pd.Series(betas)
    gamma       = beta_series.mean()
    n           = len(beta_series)

    # Naive t-stat
    se_naive = beta_series.std() / np.sqrt(n)
    t_naive  = gamma / se_naive if se_naive > 0 else np.nan

    # Newey-West HAC variance
    lags    = min(newey_west_lags, n - 1)
    demeaned = (beta_series - gamma).values
    nw_var  = np.dot(demeaned, demeaned) / n  # lag-0
    for lag in range(1, lags + 1):
        w    = 1.0 - lag / (lags + 1)   # Bartlett kernel
        cov  = np.dot(demeaned[lag:], demeaned[:-lag]) / n
        nw_var += 2 * w * cov
    se_nw = np.sqrt(max(nw_var, 0) / n)
    t_nw  = gamma / se_nw if se_nw > 0 else np.nan

    p_value = 2 * (1 - stats.t.cdf(abs(t_nw), df=n - 1)) if not np.isnan(t_nw) else np.nan

    return {
        "gamma":           round(gamma, 6),
        "t_stat":          round(t_nw, 4),
        "t_stat_naive":    round(t_naive, 4),
        "p_value":         round(p_value, 4),
        "n_periods":       n,
        "r_squared_mean":  round(np.mean(r2s), 4) if r2s else np.nan,
    }


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def quintile_returns(
    factor:   pd.DataFrame,
    prices:   pd.DataFrame,
    horizon:  int = 21,
    n_quant:  int = 5,
) -> pd.DataFrame:
    """
    Compute average forward return for each factor quintile.

    Returns:
        (dates × n_quant) DataFrame.  Column 5 = top quintile.
    """
    fwd_ret  = forward_returns(prices, horizon)
    q_labels = list(range(1, n_quant + 1))

    result = pd.DataFrame(index=factor.index, columns=q_labels, dtype=float)

    common_dates = factor.index.intersection(fwd_ret.index)
    common_ticks = factor.columns.intersection(fwd_ret.columns)

    for date in common_dates:
        fi = factor.loc[date, common_ticks].dropna()
        ri = fwd_ret.loc[date, common_ticks].dropna()
        common = fi.index.intersection(ri.index)
        if len(common) < n_quant * 2:
            continue
        fi, ri = fi.loc[common], ri.loc[common]
        try:
            bins = pd.qcut(fi, n_quant, labels=q_labels, duplicates="drop")
        except Exception:
            continue
        for q in q_labels:
            mask = bins == q
            if mask.sum() > 0:
                result.loc[date, q] = ri[mask].mean()

    return result.dropna(how="all")


def factor_summary(
    factor_name: str,
    factor:      pd.DataFrame,
    prices:      pd.DataFrame,
    horizon:     int = 21,
) -> dict:
    """
    Comprehensive factor evaluation summary.

    Returns IC stats + Fama-MacBeth + quintile spread.
    """
    fwd_ret = forward_returns(prices, horizon)

    ic     = information_coefficient(factor, fwd_ret)
    ic_s   = ic_stats(ic)
    fm     = fama_macbeth(factor, prices, horizon)
    qrets  = quintile_returns(factor, prices, horizon)

    spread = np.nan
    if not qrets.empty and 5 in qrets.columns and 1 in qrets.columns:
        spread = (qrets[5] - qrets[1]).mean()

    return {
        "factor":        factor_name,
        "horizon_days":  horizon,
        **{f"ic_{k}": v for k, v in ic_s.items()},
        **{f"fm_{k}": v for k, v in fm.items()},
        "q5_minus_q1_mean": round(spread * 252 / horizon, 4) if not np.isnan(spread) else np.nan,
    }
