"""
Ledoit-Wolf (2004) constant-correlation shrinkage.

The vectorised estimator is checked against a plain-loop transcription of the
paper's formulas, so a slip in the matrix algebra cannot hide behind a test
that reuses the same algebra.
"""

import numpy as np
import pandas as pd
import pytest

from arbitron.portfolio.optimizer import (
    _lw_cc_parts,
    constant_correlation_shrinkage,
    ledoit_wolf_covariance,
    lw_shrinkage_intensity,
)


def _reference_delta(x: np.ndarray) -> float:
    """Ledoit & Wolf (2004), Section 2.3 and Appendix B, written as loops."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = sum(y[k, i] * y[k, j] for k in range(t)) / t
    sd = np.sqrt(np.diag(s))
    r_bar = sum(
        s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j
    ) / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else r_bar * sd[i] * sd[j] for j in range(n)]
                  for i in range(n)])

    pi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi[i, j] = sum((y[k, i] * y[k, j] - s[i, j]) ** 2 for k in range(t)) / t
    pi_hat = pi.sum()

    def theta(a, i, j):   # asy cov of sqrt(T) s_aa and sqrt(T) s_ij
        return sum((y[k, a] ** 2 - s[a, a]) * (y[k, i] * y[k, j] - s[i, j]) for k in range(t)) / t

    rho = sum(pi[i, i] for i in range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rho += (r_bar / 2) * (
                np.sqrt(s[j, j] / s[i, i]) * theta(i, i, j)
                + np.sqrt(s[i, i] / s[j, j]) * theta(j, i, j)
            )
    gamma = ((f - s) ** 2).sum()
    kappa = (pi_hat - rho) / gamma
    return max(0.0, min(1.0, kappa / t))


@pytest.fixture
def correlated_returns():
    # A market factor plus a sector factor on half the names: correlations
    # differ a lot between pairs, so the constant-correlation target is
    # wrong enough that the optimal intensity lands inside (0, 1).
    rng = np.random.default_rng(3)
    t, n = 120, 6
    market = rng.normal(0, 0.01, t)
    sector = rng.normal(0, 0.01, (t, 2))
    betas = np.array([1.2, 1.1, 1.0, 0.3, 0.2, 0.1])
    in_sector = np.array([1, 1, 1, 0, 0, 0])
    x = (market[:, None] * betas + np.repeat(sector, 3, axis=1) * in_sector
         + rng.normal(0, 0.008, (t, n)))
    return pd.DataFrame(x, columns=[f"S{i}" for i in range(n)])


def test_intensity_matches_the_paper(correlated_returns):
    x = correlated_returns.values
    _, _, delta = _lw_cc_parts(x)
    assert delta == pytest.approx(_reference_delta(x), rel=1e-10, abs=1e-12)
    assert 0.0 < delta < 1.0


def test_covariance_is_the_blend(correlated_returns):
    x = correlated_returns.values
    s, f, delta = _lw_cc_parts(x)
    cov = ledoit_wolf_covariance(correlated_returns, ann_factor=1)
    np.testing.assert_allclose(cov.values, delta * f + (1 - delta) * s)
    # Variances are untouched; only the correlations are pulled together.
    np.testing.assert_allclose(np.diag(cov.values), np.diag(s))


def test_result_is_symmetric_positive_definite(correlated_returns):
    cov = ledoit_wolf_covariance(correlated_returns)
    np.testing.assert_allclose(cov.values, cov.values.T)
    assert np.linalg.eigvalsh(cov.values).min() > 0


def test_more_data_means_less_shrinkage():
    rng = np.random.default_rng(5)
    n = 8
    a = rng.normal(size=(n, n))
    true_cov = (a @ a.T / n + np.eye(n)) * 1e-4
    chol = np.linalg.cholesky(true_cov)
    short = pd.DataFrame(rng.normal(size=(60, n)) @ chol.T)
    long_ = pd.DataFrame(rng.normal(size=(6000, n)) @ chol.T)
    assert lw_shrinkage_intensity(long_) < lw_shrinkage_intensity(short)


def test_fixes_a_singular_sample_covariance():
    """More assets than observations: the sample covariance is singular, the
    shrunk one is not (the reason shrinkage exists)."""
    rng = np.random.default_rng(9)
    x = pd.DataFrame(rng.normal(0, 0.01, (15, 25)) + rng.normal(0, 0.01, (15, 1)))
    assert np.linalg.matrix_rank(x.cov().values) < 25
    assert np.linalg.eigvalsh(ledoit_wolf_covariance(x).values).min() > 0


def test_identity_target_still_available(correlated_returns):
    from sklearn.covariance import LedoitWolf
    cov = ledoit_wolf_covariance(correlated_returns, ann_factor=1, target="identity")
    np.testing.assert_allclose(cov.values, LedoitWolf().fit(correlated_returns.values).covariance_)
    with pytest.raises(ValueError):
        ledoit_wolf_covariance(correlated_returns, target="nonsense")


def test_constant_correlation_shrinkage_alpha(correlated_returns):
    opt = constant_correlation_shrinkage(correlated_returns, ann_factor=1)
    np.testing.assert_allclose(opt.values, ledoit_wolf_covariance(correlated_returns, ann_factor=1).values)
    s, f, _ = _lw_cc_parts(correlated_returns.values)
    forced = constant_correlation_shrinkage(correlated_returns, ann_factor=1, alpha=1.0)
    np.testing.assert_allclose(forced.values, f)
    with pytest.raises(ValueError):
        constant_correlation_shrinkage(correlated_returns, alpha=1.5)
