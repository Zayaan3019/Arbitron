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


def ledoit_wolf_covariance(
    returns:    pd.DataFrame,
    ann_factor: int = 252,
    target:     str = "constant_correlation",
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance with the optimal shrinkage intensity.

    target="constant_correlation" (default): Ledoit & Wolf (2004), "Honey, I
        Shrunk the Sample Covariance Matrix". Shrinks towards a matrix with
        the sample variances on the diagonal and one average correlation off
        it. Stocks really are positively correlated, so this target keeps the
        market factor that a scaled identity throws away.
    target="identity": Ledoit & Wolf (2004, JMVA) towards a scaled identity
        (sklearn's LedoitWolf), kept for comparison.

    The intensity is estimated from the data in both cases, not chosen.
    """
    if target == "identity":
        from sklearn.covariance import LedoitWolf
        r = returns.dropna().values
        cov = LedoitWolf().fit(r).covariance_ * ann_factor
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    if target != "constant_correlation":
        raise ValueError(f"unknown shrinkage target {target!r}")
    cov, _ = _constant_correlation_lw(returns.dropna().values)
    return pd.DataFrame(cov * ann_factor, index=returns.columns, columns=returns.columns)


def constant_correlation_shrinkage(
    returns:    pd.DataFrame,
    ann_factor: int = 252,
    alpha:      Optional[float] = None,
) -> pd.DataFrame:
    """
    Constant-correlation shrinkage (Ledoit & Wolf, 2004 target).

    F = D^{1/2} * R_bar * D^{1/2}
    where D = diag of sample variances, R_bar = average off-diagonal correlation.

    alpha=None (default) uses the optimal intensity from the paper; a number
    in [0, 1] forces that intensity instead.
    """
    x = returns.dropna().values
    sample, shrink_opt = _constant_correlation_lw(x, return_parts=True)
    if alpha is None:
        cov = sample
    else:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        S, F = shrink_opt
        cov = alpha * F + (1 - alpha) * S
    return pd.DataFrame(cov * ann_factor, index=returns.columns, columns=returns.columns)


def lw_shrinkage_intensity(returns: pd.DataFrame) -> float:
    """The optimal constant-correlation shrinkage intensity (diagnostic)."""
    return float(_lw_cc_parts(returns.dropna().values)[2])


def _lw_cc_parts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Sample covariance S, constant-correlation target F and the optimal
    intensity delta, following Ledoit & Wolf (2004), section 2 and appendix
    B (the authors' covCor.m):

        delta = max(0, min(1, (pi - rho) / gamma / T))

    pi    = sum of asymptotic variances of the sample covariances,
    rho   = sum of asymptotic covariances between target and sample entries,
    gamma = squared Frobenius distance between target and sample.
    """
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least two observations")
    y = x - x.mean(axis=0)
    S = y.T @ y / t
    var = np.diag(S).copy()
    if np.any(var <= 0):
        raise ValueError("an asset has zero variance")
    sd = np.sqrt(var)
    if n == 1:
        return S, S.copy(), 0.0
    corr = S / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    y2 = y ** 2
    pi_mat = y2.T @ y2 / t - S ** 2
    pi_hat = pi_mat.sum()

    gamma_hat = np.sum((F - S) ** 2)

    rho_diag = np.trace(pi_mat)
    theta = (y ** 3).T @ y / t - var[:, None] * S
    np.fill_diagonal(theta, 0.0)
    rho_off = r_bar * np.sum(np.outer(1.0 / sd, sd) * theta)
    rho_hat = rho_diag + rho_off

    if gamma_hat <= 0:
        return S, F, 1.0          # sample already equals the target
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(max(0.0, min(1.0, kappa / t)))
    return S, F, delta


def _constant_correlation_lw(x: np.ndarray, return_parts: bool = False):
    S, F, delta = _lw_cc_parts(x)
    shrunk = delta * F + (1.0 - delta) * S
    if return_parts:
        return shrunk, (S, F)
    return shrunk, delta


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


def _finalise_weights(
    w:         np.ndarray,
    tickers:   list[str],
    long_only: bool,
) -> pd.Series:
    """
    Clean up a raw optimiser solution into a valid weight vector.

    Two rules, both of which the previous code broke:

    1. **Clip only when long_only.**  `np.clip(w, 0, None)` used to run
       unconditionally, so a long-short solution had its entire short leg
       zeroed — turning a dollar-neutral book into a levered long one.
    2. **Always renormalise.**  Clipping (or any numerical drift in SLSQP)
       breaks the budget constraint.  Without renormalising, weights summed
       to 1.4385 on a real long-short run and every downstream return,
       volatility and Sharpe was scaled by that number.

    Post-condition: ``abs(w.sum() - 1) < 1e-9``.
    """
    w = np.asarray(w, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0

    if long_only:
        w = np.clip(w, 0.0, None)

    total = w.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        # Degenerate solution (e.g. a long-short book that netted to zero
        # after clipping).  Equal weight is the honest fallback.
        w = np.ones(len(w)) / max(len(w), 1)
    else:
        w = w / total

    return pd.Series(w, index=tickers)


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

    w_final = result.x if result.success else np.ones(n) / n
    return _finalise_weights(w_final, tickers, long_only)


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
    return _finalise_weights(w_final, tickers, long_only)


def risk_parity_portfolio(
    cov_matrix:  pd.DataFrame,
    target_risk: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Equal Risk Contribution (ERC) / Risk Parity portfolio.

    Each asset contributes equally to total portfolio variance:
      RC_i = w_i * (Σw)_i / w'Σw = 1/N  for all i

    This is more robust than MVO when expected returns are unreliable.

    Note on the solver: an analytic Jacobian used to be supplied here, but it
    was a self-described "simplified approximation" and was simply wrong.
    SLSQP trusts the Jacobian it is given, so it converged to a non-ERC point
    and reported ``success = True`` while doing it.  We now let SLSQP
    finite-difference the true gradient, then polish with the standard ERC
    fixed-point iteration, which converges monotonically for a PSD covariance.
    ERC is only defined for strictly positive weights, so this optimiser is
    long-only by construction.

    Post-condition: weights sum to 1 and every component risk contribution is
    within 1e-4 of `target_risk`.
    """
    tickers = list(cov_matrix.index)
    n   = len(tickers)
    cov = cov_matrix.values.astype(float)

    # Ensure PSD — a negative eigenvalue makes the risk contributions
    # meaningless and the fixed point non-convergent.
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov = cov + (-eigvals.min() + 1e-12) * np.eye(n)

    if target_risk is None:
        target_risk = np.ones(n) / n   # equal risk contribution
    target_risk = np.asarray(target_risk, dtype=float)
    target_risk = target_risk / target_risk.sum()

    def erc_objective(w):
        w = np.maximum(w, 1e-10)
        portfolio_vol = np.sqrt(max(w @ cov @ w, 1e-300))
        mrc = cov @ w / portfolio_vol   # Marginal Risk Contribution
        rc  = w * mrc                   # Risk Contribution
        rc_pct = rc / rc.sum()
        return float(np.sum((rc_pct - target_risk) ** 2))

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n

    result = minimize(
        fun=erc_objective,
        x0=w0,
        method="SLSQP",             # no jac= — finite differences are correct
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    w = np.asarray(result.x if result.success else w0, dtype=float)
    w = _erc_polish(w, cov, target_risk, erc_objective)
    return _finalise_weights(w, tickers, long_only=True)


def _erc_polish(
    w:           np.ndarray,
    cov:         np.ndarray,
    target_risk: np.ndarray,
    objective,
    max_iter:    int = 1000,
    tol:         float = 1e-14,
) -> np.ndarray:
    """
    Polish an ERC solution with the damped multiplicative fixed point.

    Solves ``w_i * (Sigma w)_i = target_i * w'Sigma w`` via the geometric
    update ``w_i <- sqrt(w_i * target_i * sigma^2 / (Sigma w)_i)``.  The
    square root damps the raw Spinu/Maillard update, which can cycle on
    near-identity covariances.

    The result is accepted **only if it lowers the objective**.  A polish
    step that makes the answer worse is exactly the failure mode the deleted
    analytic Jacobian had — silently returning a non-ERC point while
    reporting success — so it is guarded rather than trusted.
    """
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    w = w / w.sum()
    best, best_obj = w, objective(w)

    for _ in range(max_iter):
        sigma_w = cov @ w
        if not np.all(np.isfinite(sigma_w)) or np.any(sigma_w <= 0):
            break
        scale = target_risk * float(w @ sigma_w) / (w * sigma_w)
        w_new = w * np.sqrt(np.maximum(scale, 1e-300))
        w_new = np.maximum(w_new, 1e-12)
        w_new = w_new / w_new.sum()

        obj_new = objective(w_new)
        if obj_new < best_obj:
            best, best_obj = w_new, obj_new
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return best


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
