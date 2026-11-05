"""Flagship autonomous quant research lab orchestration."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from arbitron.backtest.engine import BacktestConfig, run_factor_backtest
from arbitron.backtest.metrics import compute_metrics
from arbitron.factors.evaluation import (
    factor_summary,
    forward_returns,
    ic_stats,
    information_coefficient,
)
from arbitron.factors.library import FACTOR_REGISTRY, compute_factor
from arbitron.lab.statistical_defense import StatisticalDefenseConfig, audit_strategy
from arbitron.models.foundation import MarketFoundationConfig, build_foundation_signal
from arbitron.models.regime import SimpleRegimeDetector
from arbitron.trials import TRIALS


DEFAULT_LAB_FACTORS = [
    "momentum_252_21",
    "momentum_126_21",
    "momentum_63_5",
    "reversal_21",
    "volatility_63",
    "rsi_14",
    "macd",
    "price_to_high_252",
    "acceleration_63",
]


def run_flagship_research_lab(
    prices: pd.DataFrame,
    factor_names: Optional[Iterable[str]] = None,
    horizon: int = 21,
    include_foundation: bool = True,
    n_quantiles: int = 5,
    transaction_cost: float = 0.001,
    max_factors: int = 9,
) -> dict[str, Any]:
    """
    Run factor discovery, statistical defenses, and foundation-signal research.

    The return value is JSON-friendly and designed for both the FastAPI router
    and the autonomous agent tool.
    """
    prices = prices.sort_index().dropna(how="all")
    factor_names = list(factor_names or DEFAULT_LAB_FACTORS)[:max_factors]
    factor_names = [name for name in factor_names if name in FACTOR_REGISTRY]
    cfg = BacktestConfig(
        n_quantiles=n_quantiles,
        horizon_days=horizon,
        transaction_cost=transaction_cost,
    )

    candidates: list[dict[str, Any]] = []
    returns_by_strategy: dict[str, pd.Series] = {}
    equity_by_strategy: dict[str, pd.Series] = {}
    foundation_payload: dict[str, Any] = {"enabled": False}

    for name in factor_names:
        try:
            factor = compute_factor(name, prices)
            result = run_factor_backtest(factor, prices, cfg, factor_name=name)
            if result.returns.empty:
                continue
            eval_stats = factor_summary(name, factor, prices, horizon)
            returns_by_strategy[name] = result.returns.rename(name)
            equity_by_strategy[name] = result.cumulative.rename(name)
            candidates.append({
                "name": name,
                "kind": "factor",
                "evaluation": _select_evaluation_stats(eval_stats),
                "metrics": _metrics_payload(result.returns),
            })
        except Exception as exc:
            candidates.append({
                "name": name,
                "kind": "factor",
                "error": str(exc),
            })

    if include_foundation:
        try:
            foundation = build_foundation_signal(
                prices,
                MarketFoundationConfig(horizon=horizon),
            )
            if not foundation.signal.empty:
                result = run_factor_backtest(
                    foundation.signal,
                    prices,
                    cfg,
                    factor_name="foundation_signal",
                )
                if not result.returns.empty:
                    returns_by_strategy["foundation_signal"] = result.returns.rename("foundation_signal")
                    equity_by_strategy["foundation_signal"] = result.cumulative.rename("foundation_signal")
                    fwd = forward_returns(prices, horizon)
                    ic = information_coefficient(foundation.signal, fwd)
                    candidates.append({
                        "name": "foundation_signal",
                        "kind": "market_foundation",
                        "evaluation": {
                            **{f"ic_{k}": v for k, v in ic_stats(ic).items()},
                            "fm_t_stat": None,
                            "q5_minus_q1_mean": None,
                        },
                        "metrics": _metrics_payload(result.returns),
                    })

            latest_embedding = {}
            if not foundation.embeddings.empty:
                latest_embedding = {
                    str(k): round(float(v), 4)
                    for k, v in foundation.embeddings.iloc[-1].items()
                }
            foundation_payload = {
                "enabled": True,
                "diagnostics": foundation.diagnostics,
                "latest_embedding": latest_embedding,
                "latest_reconstruction_error": (
                    round(float(foundation.reconstruction_error.dropna().iloc[-1]), 6)
                    if not foundation.reconstruction_error.dropna().empty else None
                ),
                "state_embedding_dates": [
                    d.strftime("%Y-%m-%d") for d in foundation.embeddings.tail(80).index
                ],
                "state_z1": [
                    round(float(v), 4) for v in foundation.embeddings.tail(80).iloc[:, 0].values
                ] if not foundation.embeddings.empty else [],
            }
        except Exception as exc:
            foundation_payload = {
                "enabled": False,
                "error": str(exc),
            }

    returns_matrix = pd.DataFrame(returns_by_strategy).sort_index()
    audited: list[dict[str, Any]] = []
    defense_cfg = StatisticalDefenseConfig()

    # The true multiple-testing burden is every backtest this process has run,
    # not just the candidates that survived into this table.  `run_factor_backtest`
    # records into the registry; we read the count once so all candidates in a
    # single lab run are deflated against the same N.
    n_trials = max(TRIALS.count(), int(returns_matrix.shape[1]))

    for candidate in candidates:
        name = candidate["name"]
        if "error" in candidate or name not in returns_matrix:
            audited.append(candidate)
            continue
        audit = audit_strategy(
            returns_matrix[name],
            peer_returns=returns_matrix,
            strategy_name=name,
            cfg=defense_cfg,
            n_trials=n_trials,
        )
        candidate["audit"] = audit
        candidate["regime_robustness"] = _regime_robustness(returns_matrix[name], prices)
        candidate["research_score"] = _research_score(candidate)
        audited.append(_json_clean(candidate))

    ranked = sorted(
        [c for c in audited if "audit" in c],
        key=lambda x: x.get("research_score", -999),
        reverse=True,
    )
    champion = ranked[0] if ranked else None
    pass_count = sum(1 for c in ranked if c.get("audit", {}).get("verdict") == "deployable")
    watch_count = sum(1 for c in ranked if c.get("audit", {}).get("verdict") == "watchlist")

    lab_verdict = "research"
    if champion and champion.get("audit", {}).get("verdict") == "deployable":
        lab_verdict = "candidate_ready"
    elif champion and champion.get("audit", {}).get("verdict") == "reject":
        lab_verdict = "reject"

    equity_curve = _equity_payload(equity_by_strategy.get(champion["name"]) if champion else None)
    memo = _research_memo(champion, ranked, foundation_payload, pass_count, watch_count)

    return _json_clean({
        "lab_verdict": lab_verdict,
        "date_range": {
            "start": prices.index[0].strftime("%Y-%m-%d") if len(prices) else None,
            "end": prices.index[-1].strftime("%Y-%m-%d") if len(prices) else None,
            "n_days": int(len(prices)),
            "n_assets": int(prices.shape[1]),
        },
        "settings": {
            "horizon": horizon,
            "n_quantiles": n_quantiles,
            "transaction_cost": transaction_cost,
            "include_foundation": include_foundation,
        },
        "champion": champion,
        "candidates": ranked,
        "candidate_errors": [c for c in audited if "error" in c],
        "defense_summary": {
            "tested_strategies": int(returns_matrix.shape[1]),
            #: Every backtest this process has run, which is what the DSR is
            #: deflated against.  It is >= tested_strategies by construction.
            "n_trials_deflated_against": n_trials,
            "deployable": pass_count,
            "watchlist": watch_count,
            "rejected": max(0, len(ranked) - pass_count - watch_count),
            "global_pbo": (
                ranked[0].get("audit", {}).get("pbo_stats", {}) if ranked else {}
            ),
        },
        "foundation": foundation_payload,
        "champion_equity_curve": equity_curve,
        "research_memo": memo,
    })


def _select_evaluation_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "ic_ic_mean",
        "ic_ic_ir",
        "ic_t_stat",
        "ic_pct_pos",
        "ic_n_periods",
        "fm_t_stat",
        "fm_p_value",
        "q5_minus_q1_mean",
    ]
    return {k: stats.get(k) for k in keys}


def _metrics_payload(returns: pd.Series) -> dict[str, Any]:
    metrics = compute_metrics(returns)
    total_return = float((1.0 + returns.dropna()).prod() - 1.0) if len(returns.dropna()) else np.nan
    return {
        **metrics.to_dict(),
        "total_return": round(total_return, 4) if np.isfinite(total_return) else None,
    }


def _regime_robustness(strategy_returns: pd.Series, prices: pd.DataFrame) -> dict[str, Any]:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    market_ret = returns.mean(axis=1)
    detector = SimpleRegimeDetector()
    labels = detector.fit(market_ret).predict(market_ret)
    aligned = pd.DataFrame({
        "strategy": strategy_returns,
        "regime": labels.reindex(strategy_returns.index),
    }).dropna()

    rows: dict[str, Any] = {}
    for regime, group in aligned.groupby("regime"):
        metrics = compute_metrics(group["strategy"])
        rows[str(regime)] = {
            "sharpe": round(float(metrics.sharpe), 4),
            "mean_return_ann": round(float(group["strategy"].mean() * 252), 4),
            "hit_rate": round(float((group["strategy"] > 0).mean()), 4),
            "n_days": int(len(group)),
        }
    return rows


def _research_score(candidate: dict[str, Any]) -> float:
    """
    Defense-aware ranking score on a ~100-point scale.

    Every term is a property of the candidate's measured performance.  There
    is deliberately no term for what KIND of candidate it is: a previous
    version added a flat +5 to the author's own foundation model inside
    champion selection, in the one module whose entire purpose is guarding
    against exactly that bias.  Nothing may be scored for its provenance.

    A non-evaluable DSR scores 0 for that term rather than being skipped —
    an unmeasured multiple-testing burden is not evidence of safety, and a
    candidate whose burden could not be measured should not outrank one that
    was measured and passed.
    """
    audit = candidate.get("audit", {})
    metrics = candidate.get("metrics", {})
    psr = _bounded(audit.get("probabilistic_sharpe"))
    dsr = _bounded(audit.get("deflated_sharpe"))
    pbo = audit.get("pbo")
    pbo_score = 0.5 if pbo is None else 1.0 - _bounded(pbo)
    sharpe_score = max(-1.0, min(1.0, float(metrics.get("sharpe") or 0.0) / 2.0))
    dd = abs(float(metrics.get("max_drawdown") or 0.0))
    drawdown_score = max(0.0, 1.0 - dd / 0.35)
    score = (
        30.0 * sharpe_score
        + 25.0 * dsr
        + 20.0 * psr
        + 15.0 * pbo_score
        + 10.0 * drawdown_score
    )
    return round(float(score), 4)


def _bounded(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _equity_payload(equity: Optional[pd.Series]) -> list[dict[str, Any]]:
    if equity is None or equity.empty:
        return []
    series = equity.dropna()
    if len(series) > 240:
        step = max(1, len(series) // 240)
        series = series.iloc[::step]
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
        for d, v in series.items()
    ]


def _research_memo(
    champion: Optional[dict[str, Any]],
    ranked: list[dict[str, Any]],
    foundation_payload: dict[str, Any],
    pass_count: int,
    watch_count: int,
) -> list[str]:
    if not champion:
        return [
            "No candidate produced enough valid walk-forward returns for a deployable audit.",
            "Extend the date range or reduce the universe if data coverage is sparse.",
        ]

    audit = champion.get("audit", {})
    metrics = champion.get("metrics", {})
    dsr = audit.get("deflated_sharpe")
    dsr_text = (
        f"DSR={dsr} (deflated against {audit.get('n_trials')} trials)"
        if dsr is not None
        else "DSR=NOT EVALUABLE — the multiple-testing burden is unmeasured"
    )
    memo = [
        (
            f"Champion: {champion['name']} ({champion['kind']}) with "
            f"{metrics.get('sharpe')} Sharpe and {metrics.get('max_drawdown')} max drawdown."
        ),
        (
            f"Statistical defense verdict: {audit.get('verdict')} "
            f"(PSR={audit.get('probabilistic_sharpe')}, {dsr_text}, "
            f"PBO={audit.get('pbo')})."
        ),
        f"Research queue contains {pass_count} deployable and {watch_count} watchlist candidates.",
        (
            "Universe caveat: results are computed on a non-point-in-time "
            "universe, so absolute performance is an upper bound. Relative "
            "ranking across candidates is far less affected."
        ),
    ]
    if foundation_payload.get("enabled"):
        diagnostics = foundation_payload.get("diagnostics", {})
        memo.append(
            "Market foundation adapter ran "
            f"{diagnostics.get('refits', 0)} refits with "
            f"{diagnostics.get('coverage_ratio', 0)} signal coverage."
        )
    if ranked and ranked[0].get("audit", {}).get("warnings"):
        memo.append("Top warning: " + ranked[0]["audit"]["warnings"][0])
    return memo


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
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
