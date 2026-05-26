"""
Portfolio risk analytics.

Value-at-Risk (VaR) and Expected Shortfall (CVaR/ES):
  - Historical simulation (non-parametric, no distributional assumption)
  - Parametric (Gaussian, fast)
  - Monte Carlo (with fat-tail control)

Factor risk decomposition:
  - Market beta
  - Factor loadings (via OLS)
  - Marginal Risk Contributions (MRC)
  - Component Risk Contributions (CRC)

Tail-risk measures:
  - Maximum drawdown duration
  - Tail Ratio (avg 5th percentile / avg 95th percentile)
  - Ulcer Index (root mean squared drawdown)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

def historical_var(
    returns:    pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical simulation VaR at `confidence` level.

    Returns:
        VaR as a positive number (loss threshold).
        E.g. VaR = 0.02 means 2% daily loss will not be exceeded
        with probability `confidence`.
    """
    r = returns.dropna().values
    return float(-np.percentile(r, (1 - confidence) * 100))


def historical_cvar(
    returns:    pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical Conditional VaR (Expected Shortfall) at `confidence`.

    CVaR = mean of returns below the VaR threshold.
    More coherent risk measure than VaR (Artzner et al., 1999).
    """
    r   = returns.dropna().values
    var = historical_var(returns, confidence)
    tail = r[r <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def parametric_var(
    returns:    pd.Series,
    confidence: float = 0.95,
    horizon:    int   = 1,
) -> float:
    """
    Parametric (Gaussian) VaR.  Scales with sqrt(horizon).

    Args:
        horizon: Holding period in trading days.
    """
    r   = returns.dropna()
    mu  = r.mean()
    sig = r.std()
    z   = stats.norm.ppf(1 - confidence)
    var_1day = -(mu + z * sig)
    return float(var_1day * np.sqrt(horizon))


def monte_carlo_var(
    returns:    pd.Series,
    confidence: float = 0.95,
    n_sim:      int   = 10_000,
    horizon:    int   = 1,
) -> tuple[float, float]:
    """
    Monte Carlo VaR using fitted Student-t distribution (heavier tails).

    Returns:
        (var, cvar) tuple.
    """
    r = returns.dropna().values
    # Fit Student-t (free loc parameter)
    df, loc, scale = stats.t.fit(r)
    simulated = stats.t.rvs(df=df, loc=loc, scale=scale,
                              size=(n_sim, horizon)).sum(axis=1)
    var  = float(-np.percentile(simulated, (1 - confidence) * 100))
    tail = simulated[simulated <= -var]
    cvar = float(-tail.mean()) if len(tail) > 0 else var
    return var, cvar


def var_table(
    returns:     pd.Series,
    confidences: list[float] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Return a table of VaR and CVaR at multiple confidence levels."""
    rows = []
    for c in confidences:
        rows.append({
            "confidence":      f"{int(c*100)}%",
            "historical_var":  round(historical_var(returns,  c), 4),
            "historical_cvar": round(historical_cvar(returns, c), 4),
            "parametric_var":  round(parametric_var(returns,  c), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Factor risk decomposition
# ---------------------------------------------------------------------------

def factor_beta(
    portfolio_returns: pd.Series,
    factor_returns:    pd.Series,
) -> float:
    """
    OLS beta of portfolio returns on a single factor.

    β = Cov(r_p, r_f) / Var(r_f)
    """
    r = portfolio_returns.dropna()
    f = factor_returns.reindex(r.index).dropna()
    common = r.index.intersection(f.index)
    if len(common) < 5:
        return np.nan
    r_v, f_v = r.loc[common].values, f.loc[common].values
    if f_v.var() < 1e-12:
        return np.nan
    return float(np.cov(r_v, f_v)[0, 1] / np.var(f_v))


def multi_factor_exposure(
    portfolio_returns: pd.Series,
    factor_returns:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Multi-factor OLS regression: r_p = α + Σ β_k f_k + ε

    Args:
        portfolio_returns: (dates,) portfolio return series.
        factor_returns:    (dates × factors) factor return DataFrame.

    Returns:
        DataFrame with columns [beta, t_stat, p_value] for each factor + alpha.
    """
    r = portfolio_returns.dropna()
    f = factor_returns.reindex(r.index).dropna(how="any")
    common = r.index.intersection(f.index)
    if len(common) < len(f.columns) + 5:
        return pd.DataFrame()

    r_v  = r.loc[common].values
    F    = f.loc[common].values
    X    = np.column_stack([np.ones(len(common)), F])
    n, k = X.shape

    try:
        beta, resid, rank, sv = np.linalg.lstsq(X, r_v, rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    ss_res = np.sum((r_v - X @ beta) ** 2)
    s2     = ss_res / (n - k)
    cov_b  = s2 * np.linalg.pinv(X.T @ X)
    se     = np.sqrt(np.diag(cov_b))
    t_stat = beta / se
    p_val  = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

    names = ["alpha"] + list(factor_returns.columns)
    return pd.DataFrame({
        "beta":    beta,
        "t_stat":  t_stat,
        "p_value": p_val,
        "se":      se,
    }, index=names)


# ---------------------------------------------------------------------------
# Marginal and Component Risk Contribution
# ---------------------------------------------------------------------------

def marginal_risk_contribution(
    weights:    pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Marginal Risk Contribution (MRC) = ∂σ_p / ∂w_i = (Σw)_i / σ_p

    The sensitivity of portfolio volatility to a small change in each weight.
    """
    tickers = weights.index.intersection(cov_matrix.index)
    w   = weights.loc[tickers].values
    cov = cov_matrix.loc[tickers, tickers].values
    port_vol = np.sqrt(max(w @ cov @ w, 1e-12))
    mrc = cov @ w / port_vol
    return pd.Series(mrc, index=tickers)


def component_risk_contribution(
    weights:    pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Component Risk Contribution: CRC_i = w_i * MRC_i

    Sum of CRC_i = portfolio volatility.

    Returns:
        DataFrame with [weight, crc, crc_pct] columns.
    """
    mrc  = marginal_risk_contribution(weights, cov_matrix)
    tickers = mrc.index
    w    = weights.loc[tickers]
    crc  = w * mrc
    total_risk = crc.sum()

    return pd.DataFrame({
        "weight":  w.round(4),
        "mrc":     mrc.round(6),
        "crc":     crc.round(6),
        "crc_pct": (crc / total_risk * 100).round(2) if total_risk > 0 else 0.0,
    })


# ---------------------------------------------------------------------------
# Tail risk / drawdown analytics
# ---------------------------------------------------------------------------

def ulcer_index(returns: pd.Series) -> float:
    """
    Ulcer Index: root mean squared drawdown.

    Lower is better. Unlike max drawdown, captures both depth and duration.
    """
    cum = (1 + returns.dropna()).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100   # in percent
    return float(np.sqrt((dd ** 2).mean()))


def drawdown_duration(returns: pd.Series) -> dict[str, int]:
    """
    Maximum and average drawdown duration in trading days.
    """
    cum  = (1 + returns.dropna()).cumprod()
    peak = cum.cummax()
    in_dd = cum < peak

    # Identify drawdown periods
    durations = []
    current   = 0
    for v in in_dd:
        if v:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)

    return {
        "max_duration_days": max(durations) if durations else 0,
        "avg_duration_days": int(np.mean(durations)) if durations else 0,
        "n_drawdown_periods": len(durations),
    }


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """
    Tail ratio: avg gain in top `percentile` / avg loss in bottom `percentile`.

    A ratio > 1 means right tail is larger than left tail (good skew).
    """
    r    = returns.dropna().values
    top  = r[r >= np.percentile(r, 100 - percentile)]
    bot  = r[r <= np.percentile(r, percentile)]
    return float(top.mean() / abs(bot.mean())) if len(bot) > 0 and bot.mean() != 0 else np.nan


def full_risk_report(
    returns:    pd.Series,
    market_ret: Optional[pd.Series] = None,
) -> dict:
    """
    Comprehensive risk report for a return series.
    """
    r = returns.dropna()

    report = {
        "var_95":     round(historical_var(r,  0.95), 4),
        "var_99":     round(historical_var(r,  0.99), 4),
        "cvar_95":    round(historical_cvar(r, 0.95), 4),
        "cvar_99":    round(historical_cvar(r, 0.99), 4),
        "ulcer_index": round(ulcer_index(r), 4),
        "tail_ratio":  round(tail_ratio(r), 4),
        **drawdown_duration(r),
    }

    if market_ret is not None:
        beta_val = factor_beta(r, market_ret)
        report["market_beta"] = round(beta_val, 4)

    return report
