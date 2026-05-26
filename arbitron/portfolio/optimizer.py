"""
Portfolio optimization: Mean-Variance, Max Sharpe, Min Variance, Risk Parity.

All optimizers return a pd.Series of portfolio weights (summing to 1).
Long-only constraint is enforced by default.

Covariance estimation:
  - Sample covariance (noisy for small T/N ratios)
  - Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) — strongly preferred
  - Constant-correlation shrinkage

References:
  - Markowitz (1952) "Portfolio Selection"
  - Ledoit & Wolf (2004) Journal of Multivariate Analysis
  - Maillard, Roncalli & Teïletche (2010) — Risk Parity
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame, ann_factor: int = 252) -> pd.DataFrame:
    """Annualised sample covariance matrix."""
    return returns.dropna().cov() * ann_factor


def ledoit_wolf_covariance(returns: pd.DataFrame, ann_factor: int = 252) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Shrinks the sample covariance towards a scaled identity matrix
    with optimal shrinkage intensity estimated analytically.
    This dramatically reduces estimation error for N > 20 assets.
    """
    from sklearn.covariance import LedoitWolf
    r = returns.dropna().values
    lw = LedoitWolf()
    lw.fit(r)
    cov = lw.covariance_ * ann_factor
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def constant_correlation_shrinkage(
    returns: pd.DataFrame,
    ann_factor: int = 252,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Constant-correlation shrinkage (Ledoit & Wolf, 2004 target).

    F = D^{1/2} * R_bar * D^{1/2}
    where D = diag of sample variances, R_bar = average off-diagonal correlation.
    """
    S = returns.dropna().cov() * ann_factor
    vols = np.sqrt(np.diag(S.values))
    C    = S.values / np.outer(vols, vols)   # correlation matrix
    n    = len(C)
    off  = (C.sum() - np.trace(C)) / (n * (n - 1))
    C_target = np.full_like(C, off)
    np.fill_diagonal(C_target, 1.0)
    F = np.outer(vols, vols) * C_target
    shrunk = alpha * F + (1 - alpha) * S.values
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align and validate inputs."""
    tickers = expected_returns.index.intersection(cov_matrix.index)
    mu  = expected_returns.loc[tickers].values.astype(float)
    cov = cov_matrix.loc[tickers, tickers].values.astype(float)
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (-eigvals.min() + 1e-8) * np.eye(len(cov))
    return mu, cov, list(tickers)


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(_portfolio_variance(w, cov), 0)))


def _long_only_constraints(n: int) -> list[dict]:
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]


def _long_only_bounds(n: int) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n


# ---------------------------------------------------------------------------
# Portfolio optimizers
# ---------------------------------------------------------------------------

def min_variance_portfolio(
    cov_matrix: pd.DataFrame,
    long_only:  bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Global minimum variance portfolio.

    Minimises w'Σw subject to Σw = 1 (and w ≥ 0 if long_only).

    This is the least sensitive to return estimate errors — useful when
    returns are unreliable but risk estimates are reasonable.
    """
    tickers = list(cov_matrix.index)
    n   = len(tickers)
    cov = cov_matrix.values.astype(float)

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (-eigvals.min() + 1e-8) * np.eye(n)

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, max_weight)] * n if long_only else [(-1.0, 1.0)] * n

    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        # Fallback: equal weight
        return pd.Series(1 / n, index=tickers)

    return pd.Series(result.x, index=tickers)


def max_sharpe_portfolio(
    expected_returns: pd.Series,
    cov_matrix:       pd.DataFrame,
    rf:               float = 0.02,
    long_only:        bool  = True,
    max_weight:       float = 0.3,
) -> pd.Series:
    """
    Maximum Sharpe ratio (tangency) portfolio.

    Maximises  (μ - rf) / σ_p  subject to Σw = 1 (and w ≥ 0 if long_only).

    Note: maximising Sharpe is equivalent to minimising -Sharpe, which is
    non-convex. We solve it as a QP using the "two-fund separation" trick:
      z = Σ^{-1} (μ - rf)
      w* = z / sum(z)
    This is exact under the standard assumptions.  For robustness we also
    run SLSQP from that starting point.
    """
    mu, cov, tickers = _validate_inputs(expected_returns, cov_matrix)
    n       = len(tickers)
    rf_vec  = rf * np.ones(n)
    excess  = mu - rf_vec

    # Analytical starting point (valid when long-short is allowed)
    try:
        z = np.linalg.solve(cov, excess)
        if z.sum() != 0:
            w_analytic = z / z.sum()
        else:
            w_analytic = np.ones(n) / n
    except np.linalg.LinAlgError:
        w_analytic = np.ones(n) / n

    if long_only:
        w0 = np.clip(w_analytic, 0, None)
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n
    else:
        w0 = w_analytic

    def neg_sharpe(w):
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -(port_ret - rf) / port_vol

    bounds = [(0.0, max_weight)] * n if long_only else [(-0.2, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    result = minimize(
        fun=neg_sharpe,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    w_final = result.x if result.success else w0
    return pd.Series(np.clip(w_final, 0, None), index=tickers)


def risk_parity_portfolio(
    cov_matrix:  pd.DataFrame,
    target_risk: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Equal Risk Contribution (ERC) / Risk Parity portfolio.

    Each asset contributes equally to total portfolio variance:
      RC_i = w_i * (Σw)_i / w'Σw = 1/N  for all i

    This is more robust than MVO when expected returns are unreliable.
    """
    tickers = list(cov_matrix.index)
    n   = len(tickers)
    cov = cov_matrix.values.astype(float)

    if target_risk is None:
        target_risk = np.ones(n) / n   # equal risk contribution

    def erc_objective(w):
        w = np.maximum(w, 1e-10)
        portfolio_vol = np.sqrt(w @ cov @ w)
        mrc = cov @ w / portfolio_vol   # Marginal Risk Contribution
        rc  = w * mrc                   # Risk Contribution
        rc_pct = rc / rc.sum()
        return float(np.sum((rc_pct - target_risk) ** 2))

    def erc_gradient(w):
        w = np.maximum(w, 1e-10)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(port_var)
        mrc      = cov @ w / port_vol
        rc       = w * mrc
        rc_sum   = rc.sum()
        rc_pct   = rc / rc_sum

        # Gradient of sum((rc_pct - target)**2) w.r.t. w
        d_obj = 2 * (rc_pct - target_risk)
        # Chain rule (simplified first-order approximation)
        grad = np.zeros(n)
        for i in range(n):
            grad[i] = d_obj[i] * (mrc[i] * rc_sum - rc[i] * mrc.sum()) / (rc_sum ** 2)
        return grad

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-4, 1.0)] * n

    result = minimize(
        fun=erc_objective,
        x0=w0,
        jac=erc_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    w = result.x if result.success else w0
    w = np.maximum(w, 0)
    w /= w.sum()
    return pd.Series(w, index=tickers)


def mean_variance_frontier(
    expected_returns: pd.Series,
    cov_matrix:       pd.DataFrame,
    n_points:         int = 20,
    long_only:        bool = True,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by sweeping target returns.

    Returns:
        DataFrame with columns [return, volatility, sharpe] for each point.
    """
    mu, cov, tickers = _validate_inputs(expected_returns, cov_matrix)
    n = len(tickers)

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, n_points)

    points = []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n

        result = minimize(
            fun=lambda w: float(w @ cov @ w),
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )
        if result.success:
            w = result.x
            vol    = np.sqrt(w @ cov @ w)
            sharpe = (target - 0.02) / vol if vol > 0 else 0.0
            points.append({
                "return":     round(float(target),   4),
                "volatility": round(float(vol),       4),
                "sharpe":     round(float(sharpe),    4),
            })

    return pd.DataFrame(points)


# ---------------------------------------------------------------------------
# Portfolio analytics
# ---------------------------------------------------------------------------

def portfolio_performance(
    weights:          pd.Series,
    expected_returns: pd.Series,
    cov_matrix:       pd.DataFrame,
    rf:               float = 0.02,
) -> dict[str, float]:
    """Compute return, volatility, and Sharpe for a given weight vector."""
    tickers = weights.index.intersection(expected_returns.index).intersection(cov_matrix.index)
    w   = weights.loc[tickers].values
    mu  = expected_returns.loc[tickers].values
    cov = cov_matrix.loc[tickers, tickers].values

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 0)))
    sharpe   = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

    return {
        "expected_return": round(port_ret, 4),
        "volatility":      round(port_vol, 4),
        "sharpe":          round(sharpe,   4),
    }


def diversification_ratio(
    weights:    pd.Series,
    cov_matrix: pd.DataFrame,
) -> float:
    """
    Diversification ratio (Choueifaty & Coignard, 2008).

    DR = (weighted avg individual vol) / (portfolio vol)
    DR >= 1; higher = more diversification benefit.
    """
    tickers = weights.index.intersection(cov_matrix.index)
    w   = weights.loc[tickers].values
    cov = cov_matrix.loc[tickers, tickers].values
    vol_individual = np.sqrt(np.diag(cov))
    weighted_sum   = w @ vol_individual
    port_vol       = np.sqrt(max(w @ cov @ w, 1e-12))
    return float(weighted_sum / port_vol)
