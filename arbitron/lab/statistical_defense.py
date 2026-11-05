""" 
Statistical defenses for quant research.

This module turns a backtest from "nice equity curve" into an auditable
research artifact.  It includes:
  - Probabilistic Sharpe Ratio (PSR)
  - Deflated Sharpe Ratio (DSR) against multiple tested strategies
  - Probability of Backtest Overfitting (PBO) via CSCV-style splits
  - Deployment gates and a haircut Sharpe used for ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, log, sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from arbitron.backtest.metrics import compute_metrics


@dataclass(frozen=True)
class StatisticalDefenseConfig:
    """Deployment gates for strategy audit."""

    ann_factor: int = 252
    benchmark_sharpe: float = 0.0
    min_probabilistic_sharpe: float = 0.95
    min_deflated_sharpe: float = 0.80
    max_pbo: float = 0.30
    max_drawdown_loss: float = 0.35
    min_oos_days: int = 252


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    ann_factor: int = 252,
) -> float:
    """
    Probability that the true Sharpe exceeds `benchmark_sharpe`.

    The formula adjusts for sample size, skew, and non-Gaussian kurtosis.
    Inputs and outputs use annualised Sharpe for user-facing consistency,
    while the statistic is computed on the native daily return scale.
    """
    r = _clean_returns(returns)
    # A tolerance rather than `<= 0`: a constant series can produce a std of
    # ~1e-18 instead of exactly zero, which then makes the moment calculations
    # numerically meaningless (scipy warns about catastrophic cancellation)
    # and yields a "confidence" that is pure floating-point noise.
    std = float(r.std())
    if len(r) < 3 or not np.isfinite(std) or std <= 1e-15:
        return float("nan")

    sr = r.mean() / std
    sr_benchmark = benchmark_sharpe / sqrt(ann_factor)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    denom = sqrt(max(denom_sq, 1e-12))
    z = (sr - sr_benchmark) * sqrt(len(r) - 1) / denom
    return float(stats.norm.cdf(z))


EULER_MASCHERONI = 0.5772156649015329


def expected_maximum_sharpe(
    trial_sharpes: list[float] | pd.Series,
    n_trials: Optional[int] = None,
) -> float:
    """
    Expected maximum Sharpe **under the null hypothesis of no skill**.

    Bailey & Lopez de Prado (2014), eq. 8::

        E[max SR_N] ~= sigma_SR * [ (1 - gamma) * Phi^-1(1 - 1/N)
                                    + gamma * Phi^-1(1 - 1/(N*e)) ]

    where `sigma_SR` is the cross-sectional dispersion of the trial Sharpes,
    `gamma` is the Euler-Mascheroni constant and `N` is the number of trials.

    The location term is **zero by construction**: under the null every
    strategy has true Sharpe 0, so only the dispersion of the estimates
    drives the selection hurdle.  The previous implementation added the
    *sample mean of realised trial Sharpes*, which made the hurdle track the
    strategy being tested — with one trial the hurdle came out exactly equal
    to that strategy's own Sharpe, z collapsed to 0 and DSR was a constant
    0.5.  It never deflated anything.

    Args:
        trial_sharpes: Annualised Sharpe of each trial, used ONLY to
            estimate `sigma_SR`.
        n_trials: Number of trials `N`.  Defaults to `len(trial_sharpes)`,
            but callers should pass the true search count (see
            `arbitron.trials.TRIALS`) because the strategies that survived
            to a comparison table are a small subset of the ones tried.

    Returns:
        The hurdle in annualised Sharpe units, or **NaN when the statistic
        is not evaluable** — N < 2 (no multiple-testing burden to estimate)
        or zero dispersion (no scale to estimate it from).  NaN is
        deliberate: a number here would look like a result.
    """
    s = pd.Series(trial_sharpes, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(n_trials) if n_trials is not None else len(s)

    if n < 2 or len(s) < 2:
        return float("nan")

    sigma = float(s.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return float("nan")

    term_a = stats.norm.ppf(1.0 - 1.0 / n)
    term_b = stats.norm.ppf(1.0 - 1.0 / (n * e))
    hurdle = sigma * ((1.0 - EULER_MASCHERONI) * term_a + EULER_MASCHERONI * term_b)
    return float(hurdle) if np.isfinite(hurdle) else float("nan")


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: list[float] | pd.Series,
    ann_factor: int = 252,
    n_trials: Optional[int] = None,
) -> float:
    """
    Probability that the true Sharpe clears the multiple-testing hurdle.

    Returns NaN when the hurdle is not evaluable (see
    `expected_maximum_sharpe`).  It never silently degrades to the
    un-deflated PSR — that would report a *weaker* test under a stronger
    name.
    """
    hurdle = expected_maximum_sharpe(trial_sharpes, n_trials=n_trials)
    if not np.isfinite(hurdle):
        return float("nan")
    return probabilistic_sharpe_ratio(returns, hurdle, ann_factor)


def estimate_backtest_overfitting(
    returns_by_strategy: pd.DataFrame,
    n_splits: int = 8,
    max_combinations: int = 128,
    ann_factor: int = 252,
) -> dict[str, Any]:
    """
    Estimate Probability of Backtest Overfitting using CSCV-style splits.

    Rows are ordered return observations; columns are strategy variants.  For
    each train/test split, select the best in-sample Sharpe and measure where
    that same strategy ranks out of sample.  PBO is the fraction of splits where
    the selected strategy falls below the out-of-sample median.
    """
    matrix = returns_by_strategy.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    matrix = matrix.fillna(0.0)
    if matrix.shape[1] < 2 or len(matrix) < max(20, n_splits * 4):
        return {
            "pbo": None,
            "median_logit": None,
            "loss_probability": None,
            "n_trials": int(matrix.shape[1]),
            "n_splits": 0,
            "reason": "Need at least two strategies and enough observations for PBO.",
        }

    n_splits = max(4, min(n_splits if n_splits % 2 == 0 else n_splits - 1, len(matrix) // 4))
    if n_splits < 4:
        n_splits = 4
    partitions = np.array_split(np.arange(len(matrix)), n_splits)
    split_ids = range(n_splits)
    train_size = n_splits // 2
    combos = list(combinations(split_ids, train_size))

    if len(combos) > max_combinations:
        step = max(1, len(combos) // max_combinations)
        combos = combos[::step][:max_combinations]

    logits: list[float] = []
    oos_sharpes: list[float] = []
    selected: list[str] = []

    for combo in combos:
        train_idx = np.concatenate([partitions[i] for i in combo])
        test_idx = np.concatenate([partitions[i] for i in split_ids if i not in combo])
        train = matrix.iloc[train_idx]
        test = matrix.iloc[test_idx]

        train_sr = train.apply(_annualized_sharpe, ann_factor=ann_factor)
        test_sr = test.apply(_annualized_sharpe, ann_factor=ann_factor)
        if train_sr.dropna().empty or test_sr.dropna().empty:
            continue

        winner = str(train_sr.idxmax())
        ranks = test_sr.rank(method="average", ascending=True)
        omega = float(ranks[winner] / (len(test_sr) + 1.0))
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(log(omega / (1.0 - omega)))
        oos_sharpes.append(float(test_sr[winner]))
        selected.append(winner)

    if not logits:
        return {
            "pbo": None,
            "median_logit": None,
            "loss_probability": None,
            "n_trials": int(matrix.shape[1]),
            "n_splits": 0,
            "reason": "No valid CSCV splits.",
        }

    return {
        "pbo": round(float(np.mean(np.array(logits) <= 0.0)), 4),
        "median_logit": round(float(np.median(logits)), 4),
        "loss_probability": round(float(np.mean(np.array(oos_sharpes) < 0.0)), 4),
        "n_trials": int(matrix.shape[1]),
        "n_splits": len(logits),
        "most_selected": _most_common(selected),
    }


def audit_strategy(
    returns: pd.Series,
    peer_returns: Optional[pd.DataFrame] = None,
    strategy_name: str = "strategy",
    cfg: Optional[StatisticalDefenseConfig] = None,
    n_trials: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run statistical deployment gates for one strategy.

    Args:
        peer_returns: Every candidate tested in the same research session, so
            DSR dispersion and PBO reflect the real multiple-testing burden.
        n_trials: Total number of backtests run during the search.  Defaults
            to the number of peer columns, which UNDER-counts (peers are the
            survivors).  The lab passes `arbitron.trials.TRIALS.count()`.

    The DSR gate reports `pass: null` and a `not_evaluable` reason whenever
    the hurdle cannot be estimated (fewer than two trials, or zero Sharpe
    dispersion).  It is never reported as a number in that case.
    """
    cfg = cfg or StatisticalDefenseConfig()
    r = _clean_returns(returns)
    metrics = compute_metrics(r, ann_factor=cfg.ann_factor)
    sharpe = float(metrics.sharpe)

    if peer_returns is None or peer_returns.empty:
        peers = pd.DataFrame({strategy_name: r})
    else:
        peers = peer_returns.copy()
        if strategy_name not in peers.columns:
            peers[strategy_name] = r

    trial_sharpes = [
        _annualized_sharpe(peers[col], ann_factor=cfg.ann_factor)
        for col in peers.columns
    ]
    effective_trials = int(n_trials) if n_trials is not None else len(trial_sharpes)

    psr = probabilistic_sharpe_ratio(r, cfg.benchmark_sharpe, cfg.ann_factor)
    hurdle = expected_maximum_sharpe(trial_sharpes, n_trials=effective_trials)
    dsr = deflated_sharpe_ratio(
        r, trial_sharpes, cfg.ann_factor, n_trials=effective_trials
    )
    pbo_stats = estimate_backtest_overfitting(peers, ann_factor=cfg.ann_factor)
    pbo = pbo_stats.get("pbo")
    pbo_for_score = 0.5 if pbo is None else float(pbo)

    max_dd_loss = abs(float(metrics.max_drawdown)) if not np.isnan(metrics.max_drawdown) else np.inf

    # Haircut: only multipliers that were actually EVALUATED shrink the
    # Sharpe.  A non-evaluable DSR must not silently act as a free pass, so
    # it is recorded in `haircut_components` for the reader to see.
    haircut_sharpe = sharpe
    haircut_components: dict[str, Any] = {}
    for label, multiplier in (
        ("probabilistic_sharpe", psr),
        ("deflated_sharpe", dsr),
        ("one_minus_pbo", 1.0 - pbo_for_score),
    ):
        if np.isfinite(multiplier):
            factor = max(0.0, min(1.0, float(multiplier)))
            haircut_sharpe *= factor
            haircut_components[label] = round(factor, 6)
        else:
            haircut_components[label] = None

    dsr_reason = None if np.isfinite(dsr) else _dsr_not_evaluable_reason(
        trial_sharpes, effective_trials
    )

    gates = [
        _gate("Probabilistic Sharpe", psr, cfg.min_probabilistic_sharpe, ">="),
        _gate("Deflated Sharpe", dsr, cfg.min_deflated_sharpe, ">=", reason=dsr_reason),
        _gate("Backtest Overfitting", pbo, cfg.max_pbo, "<=",
              reason=pbo_stats.get("reason")),
        _gate("Drawdown Loss", max_dd_loss, cfg.max_drawdown_loss, "<="),
        _gate("OOS Days", len(r), cfg.min_oos_days, ">="),
    ]

    failed = [g["name"] for g in gates if g["pass"] is False]
    soft_failed = [g["name"] for g in gates if g["pass"] is None]
    if not failed and not soft_failed:
        verdict = "deployable"
    elif len(failed) <= 2:
        verdict = "watchlist"
    else:
        verdict = "reject"

    warnings = _warnings(metrics, psr, dsr, pbo, len(r), cfg)
    return _json_clean({
        "strategy_name": strategy_name,
        "verdict": verdict,
        "sharpe": sharpe,
        "haircut_sharpe": haircut_sharpe,
        "haircut_components": haircut_components,
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "deflated_sharpe_evaluable": bool(np.isfinite(dsr)),
        "deflated_sharpe_reason": dsr_reason,
        "expected_max_sharpe_hurdle": hurdle,
        "n_trials": effective_trials,
        "n_peer_strategies": len(trial_sharpes),
        "pbo": pbo,
        "pbo_stats": pbo_stats,
        "max_drawdown_loss": max_dd_loss,
        "n_observations": len(r),
        "gates": gates,
        "warnings": warnings,
    })


def _dsr_not_evaluable_reason(trial_sharpes: list[float], n_trials: int) -> str:
    s = pd.Series(trial_sharpes, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if n_trials < 2 or len(s) < 2:
        return (
            "Not evaluable: DSR needs at least two trials to estimate the "
            f"selection hurdle (n_trials={n_trials}, usable Sharpes={len(s)}). "
            "A single backtest carries no multiple-testing burden to deflate."
        )
    return (
        "Not evaluable: the tested strategies have effectively zero Sharpe "
        "dispersion, so the selection hurdle has no scale to be estimated from."
    )


def _annualized_sharpe(returns: pd.Series, ann_factor: int = 252) -> float:
    r = _clean_returns(returns)
    if len(r) < 2 or r.std() <= 0:
        return 0.0
    return float(r.mean() / r.std() * sqrt(ann_factor))


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _gate(
    name: str,
    value: Any,
    threshold: float,
    op: str,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build one deployment gate.

    `pass: None` means **not evaluable** — the statistic could not be
    computed.  It is never conflated with a pass, and `audit_strategy`
    downgrades any strategy carrying one out of "deployable".
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        passed = None
    elif op == ">=":
        passed = float(value) >= threshold
    else:
        passed = float(value) <= threshold
    return {
        "name": name,
        "value": None if value is None else round(float(value), 4),
        "threshold": threshold,
        "operator": op,
        "pass": passed,
        "evaluable": passed is not None,
        "reason": reason if passed is None else None,
    }


def _warnings(
    metrics: Any,
    psr: float,
    dsr: float,
    pbo: Optional[float],
    n_obs: int,
    cfg: StatisticalDefenseConfig,
) -> list[str]:
    warnings: list[str] = []
    if n_obs < cfg.min_oos_days:
        warnings.append("Short out-of-sample record; treat the estimate as preliminary.")
    if np.isfinite(psr) and psr < cfg.min_probabilistic_sharpe:
        warnings.append("Sharpe is not statistically convincing against the benchmark hurdle.")
    if np.isfinite(dsr) and dsr < cfg.min_deflated_sharpe:
        warnings.append("Multiple-testing adjustment materially weakens the strategy.")
    if not np.isfinite(dsr):
        warnings.append(
            "Deflated Sharpe is NOT EVALUABLE for this run — the multiple-testing "
            "burden is unmeasured, so the Sharpe shown is un-deflated."
        )
    if pbo is not None and pbo > cfg.max_pbo:
        warnings.append("Candidate selection appears unstable under CSCV overfitting tests.")
    if abs(float(metrics.max_drawdown)) > cfg.max_drawdown_loss:
        warnings.append("Drawdown exceeds the research-lab deployment threshold.")
    if float(metrics.skewness) < -1.0:
        warnings.append("Return distribution has severe negative skew.")
    return warnings


def _most_common(values: list[str]) -> Optional[str]:
    if not values:
        return None
    counts = pd.Series(values).value_counts()
    return str(counts.index[0])


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 6)
    return value
