"""
Tool definitions for the Arbitron autonomous research agent.

Tools are defined in OpenAI function-calling format (compatible with Groq).
Each tool has:
  - type:        "function"
  - function.name:        unique identifier
  - function.description: tells the LLM when and how to use it
  - function.parameters:  JSON Schema for required + optional parameters

The tool executor (tools.py:execute_tool) maps tool names to Python
functions and returns structured JSON results.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible function-calling format, as Groq expects)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_universe",
            "description": (
                "Fetch a list of liquid stock tickers for a given sector or "
                "return the default S&P 500 mini-universe.  Call this first to "
                "get the tickers you'll use in subsequent tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name. One of: Technology, Healthcare, Financials, Industrials, ConsumerDisc, Energy. Omit this parameter entirely for the default S&P 500 universe.",
                        "enum": ["Technology", "Healthcare", "Financials", "Industrials", "ConsumerDisc", "Energy"],
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of tickers to return (max 30, default 15).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_factor",
            "description": (
                "Compute a named quantitative factor for a list of tickers over "
                "a date range.  Returns cross-sectional factor statistics and "
                "a sample of factor values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_name": {
                        "type": "string",
                        "description": "Factor to compute.  Options: momentum_252_21, momentum_126_21, momentum_63_5, reversal_21, reversal_5, volatility_63, volatility_21, rsi_14, rsi_7, macd, bollinger, price_to_high_252, acceleration_63.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g. ['AAPL', 'MSFT']).",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format.",
                        "default": "2018-01-01",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (default: today).",
                    },
                },
                "required": ["factor_name", "tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_factor",
            "description": (
                "Evaluate a factor's predictive power via IC (Information Coefficient) "
                "and Fama-MacBeth regression.  Returns IC mean, IC IR, t-statistic, "
                "p-value, and quintile spread."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_name": {"type": "string"},
                    "tickers":     {"type": "array", "items": {"type": "string"}},
                    "horizon_days": {
                        "type": "integer",
                        "description": "Forward return horizon in days (default 21 = 1 month).",
                        "default": 21,
                    },
                    "start_date": {"type": "string", "default": "2018-01-01"},
                    "end_date":   {"type": "string"},
                },
                "required": ["factor_name", "tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "Run a walk-forward quintile backtest of a factor.  Longs top quintile, "
                "shorts bottom quintile, rebalances monthly with transaction costs. "
                "Returns Sharpe, Sortino, CAGR, max drawdown, and daily equity curve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_name": {"type": "string"},
                    "tickers":     {"type": "array", "items": {"type": "string"}},
                    "start_date":  {"type": "string", "default": "2018-01-01"},
                    "end_date":    {"type": "string"},
                    "long_only":   {
                        "type": "boolean",
                        "description": "If true, only hold the top quintile (long-only).",
                        "default": False,
                    },
                    "transaction_cost_bps": {
                        "type": "number",
                        "description": "One-way transaction cost in basis points (default 10 bps).",
                        "default": 10,
                    },
                },
                "required": ["factor_name", "tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_portfolio",
            "description": (
                "Optimize portfolio weights for a list of tickers using the chosen method. "
                "Returns weights, expected return, volatility, and Sharpe ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "method": {
                        "type": "string",
                        "description": "Optimization method: max_sharpe, min_variance, risk_parity.",
                        "enum": ["max_sharpe", "min_variance", "risk_parity"],
                        "default": "max_sharpe",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Days of history for covariance estimation.",
                        "default": 504,
                    },
                    "start_date": {"type": "string", "default": "2018-01-01"},
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_risk_metrics",
            "description": (
                "Compute comprehensive risk metrics for a portfolio or single stock: "
                "VaR, CVaR, Sharpe, Sortino, max drawdown, beta, and tail ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "weights": {
                        "type": "object",
                        "description": "Optional {ticker: weight} allocation.  If omitted, equal-weight is assumed.",
                        "additionalProperties": {"type": "number"},
                    },
                    "start_date": {"type": "string", "default": "2018-01-01"},
                    "end_date":   {"type": "string"},
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_regime",
            "description": (
                "Detect the current market regime (Bull/Bear) using a Hidden Markov Model. "
                "Returns regime label, probability, and historical regime statistics. "
                "Pass the same tickers you already fetched so cached data is reused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stock tickers to use as market proxy for regime detection. "
                            "Pass the same tickers from your fetch_universe call to reuse cached data."
                        ),
                    },
                    "start_date": {"type": "string", "default": "2018-01-01"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_factors",
            "description": (
                "Compare multiple factors head-to-head: IC, ICIR, Fama-MacBeth t-stat, "
                "and quintile-backtest Sharpe for the same universe and period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of factor names to compare.",
                    },
                    "tickers":     {"type": "array", "items": {"type": "string"}},
                    "start_date":  {"type": "string", "default": "2018-01-01"},
                    "horizon_days": {"type": "integer", "default": 21},
                },
                "required": ["factor_names", "tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_quant_lab",
            "description": (
                "Run the flagship autonomous quant research lab: tests multiple factors, "
                "optionally builds a market-foundation signal, audits every candidate "
                "with PSR/DSR/PBO statistical defenses, and returns a ranked research memo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock tickers to research. Use tickers from fetch_universe.",
                    },
                    "factor_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional factor subset. Omit to use the lab default research queue.",
                    },
                    "start_date": {"type": "string", "default": "2020-01-01"},
                    "end_date": {"type": "string"},
                    "horizon_days": {"type": "integer", "default": 21},
                    "include_foundation": {"type": "boolean", "default": True},
                    "transaction_cost_bps": {"type": "number", "default": 10},
                },
                "required": ["tickers"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def execute_tool(
    tool_name:   str,
    tool_input:  dict[str, Any],
    cache_dir:   str = "data/cache",
) -> dict[str, Any]:
    """
    Dispatch a tool call to the appropriate Python function.

    Returns a dict that will be serialised as the tool_result content.
    On error, returns {"error": message} so the agent can recover.

    Arguments are validated against the tool's JSON schema first (unknown
    keys, wrong types, malformed dates, unknown factors and out-of-range
    numbers are rejected), so a malformed call comes back as a precise error
    the model can correct, never as a half-run tool with silent defaults.
    """
    if tool_name not in _SCHEMA_BY_NAME:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        tool_input = validate_tool_args(tool_name, tool_input)
    except ToolArgumentError as e:
        return {"error": str(e), "tool": tool_name, "invalid_arguments": True}
    try:
        if tool_name == "fetch_universe":
            return _tool_fetch_universe(tool_input, cache_dir)
        elif tool_name == "compute_factor":
            return _tool_compute_factor(tool_input, cache_dir)
        elif tool_name == "evaluate_factor":
            return _tool_evaluate_factor(tool_input, cache_dir)
        elif tool_name == "run_backtest":
            return _tool_run_backtest(tool_input, cache_dir)
        elif tool_name == "optimize_portfolio":
            return _tool_optimize_portfolio(tool_input, cache_dir)
        elif tool_name == "compute_risk_metrics":
            return _tool_compute_risk(tool_input, cache_dir)
        elif tool_name == "detect_regime":
            return _tool_detect_regime(tool_input, cache_dir)
        elif tool_name == "compare_factors":
            return _tool_compare_factors(tool_input, cache_dir)
        elif tool_name == "run_quant_lab":
            return _tool_run_quant_lab(tool_input, cache_dir)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


# ---------------------------------------------------------------------------
# Argument validation (pydantic models generated from TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

class ToolArgumentError(ValueError):
    """A tool call whose arguments do not match the tool's schema."""


_SCHEMA_BY_NAME: dict[str, dict] = {
    t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS
}

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

#: Bounds the JSON schemas only state in prose.
_BOUNDS: dict[str, dict[str, Any]] = {
    "n":                    {"ge": 1, "le": 30},
    "horizon_days":         {"ge": 1, "le": 252},
    "lookback_days":        {"ge": 21, "le": 5040},
    "transaction_cost_bps": {"ge": 0, "le": 500},
    "tickers":              {"min_length": 1, "max_length": 200},
    "factor_names":         {"min_length": 1, "max_length": 20},
}


def _annotation(spec: dict) -> Any:
    kind = spec.get("type")
    if "enum" in spec:
        return Literal[tuple(spec["enum"])]
    if kind == "string":
        return str
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        return list[_annotation(spec.get("items", {}))]
    if kind == "object":
        return dict[str, _annotation(spec.get("additionalProperties", {}))]
    return Any


@lru_cache(maxsize=None)
def _args_model(tool_name: str) -> type[BaseModel]:
    params = _SCHEMA_BY_NAME[tool_name]
    required = set(params.get("required", []))
    fields: dict[str, Any] = {}
    for name, spec in params.get("properties", {}).items():
        ann = _annotation(spec)
        kw = dict(_BOUNDS.get(name, {}))
        if name.endswith("_date"):
            kw["pattern"] = _DATE_PATTERN
        if name in required:
            fields[name] = (ann, Field(**kw))
        else:
            fields[name] = (Optional[ann], Field(default=None, **kw))
    return create_model(
        f"{tool_name}_args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def validate_tool_args(tool_name: str, tool_input: Any) -> dict[str, Any]:
    """
    Check a tool call's arguments against its schema.

    Returns only the arguments the caller actually set (coerced to the
    schema's types), so each tool's own defaults still apply to the rest.
    Raises ToolArgumentError with every problem listed.
    """
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        raise ToolArgumentError(
            f"arguments for {tool_name} must be a JSON object, got {type(tool_input).__name__}"
        )
    # An explicit null means "not given" (models often send it for optionals).
    given = {k: v for k, v in tool_input.items() if v is not None}
    try:
        args = _args_model(tool_name).model_validate(given)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ToolArgumentError(f"invalid arguments for {tool_name}: {problems}") from None
    out = args.model_dump(exclude_unset=True)

    from arbitron.factors.library import FACTOR_REGISTRY
    names = ([out["factor_name"]] if "factor_name" in out else []) + list(out.get("factor_names", []))
    unknown = [n for n in names if n not in FACTOR_REGISTRY]
    if unknown:
        raise ToolArgumentError(
            f"invalid arguments for {tool_name}: unknown factor(s) {unknown}; "
            f"choose from {sorted(FACTOR_REGISTRY)}"
        )
    for key in ("start_date", "end_date"):
        if key in out:
            try:
                pd.Timestamp(out[key])
            except (ValueError, TypeError):
                raise ToolArgumentError(
                    f"invalid arguments for {tool_name}: {key}={out[key]!r} is not a real date"
                ) from None
    if "start_date" in out and "end_date" in out and out["end_date"] <= out["start_date"]:
        raise ToolArgumentError(
            f"invalid arguments for {tool_name}: end_date must be after start_date"
        )
    return out


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _md(cache_dir: str):
    from arbitron.data.market import MarketData
    return MarketData(cache_dir=cache_dir)


def _tool_fetch_universe(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    md   = MarketData(cache_dir=cache_dir)
    sect = inp.get("sector")
    n    = min(int(inp.get("n", 15)), 30)
    tickers = md.get_universe(sector=sect, n=n)
    return {"tickers": tickers, "n": len(tickers), "sector": sect or "default"}


def _tool_compute_factor(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.factors.library import compute_factor, cross_section_zscore

    md      = _md(cache_dir)
    tickers = inp["tickers"]
    fname   = inp["factor_name"]
    start   = inp.get("start_date", "2018-01-01")
    end     = inp.get("end_date")

    # Always fetch to today (end=None) for maximum cache reuse across tool calls,
    # then slice to the requested end date if one was specified.
    prices = md.get_prices(tickers, start=start, end=None)
    if end:
        prices = prices[prices.index <= pd.Timestamp(end)]
    factor = compute_factor(fname, prices)
    factor_z = cross_section_zscore(factor)

    # Latest cross-section
    last_date = factor_z.dropna(how="all").index[-1] if not factor_z.empty else None
    latest    = {}
    if last_date is not None:
        row = factor_z.loc[last_date].dropna()
        latest = dict(row.sort_values(ascending=False).head(10))

    return {
        "factor_name":   fname,
        "start_date":    str(prices.index[0].date()),
        "end_date":      str(prices.index[-1].date()),
        "n_stocks":      len(prices.columns),
        "n_dates":       len(prices),
        "latest_date":   str(last_date.date()) if last_date else None,
        "top_10_by_factor": {k: round(float(v), 3) for k, v in latest.items()},
    }


def _tool_evaluate_factor(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.factors.library import compute_factor
    from arbitron.factors.evaluation import factor_summary

    md      = _md(cache_dir)
    tickers = inp["tickers"]
    fname   = inp["factor_name"]
    horizon = int(inp.get("horizon_days", 21))
    start   = inp.get("start_date", "2018-01-01")
    end     = inp.get("end_date")

    prices = md.get_prices(tickers, start=start, end=None)
    if end:
        prices = prices[prices.index <= pd.Timestamp(end)]
    factor = compute_factor(fname, prices)
    result = factor_summary(fname, factor, prices, horizon)

    return _clean_dict(result)


def _tool_run_backtest(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.factors.library import compute_factor
    from arbitron.backtest.engine import run_factor_backtest, BacktestConfig

    md      = _md(cache_dir)
    tickers = inp["tickers"]
    fname   = inp["factor_name"]
    start   = inp.get("start_date", "2018-01-01")
    end     = inp.get("end_date")
    tc_bps  = float(inp.get("transaction_cost_bps", 10))

    prices = md.get_prices(tickers, start=start, end=None)
    if end:
        prices = prices[prices.index <= pd.Timestamp(end)]
    factor = compute_factor(fname, prices)

    cfg = BacktestConfig(
        transaction_cost=tc_bps / 10_000,
        long_only=bool(inp.get("long_only", False)),
    )
    result = run_factor_backtest(factor, prices, cfg, factor_name=fname)
    summary = result.summary()

    # Include equity curve as list (truncated to 200 points)
    equity = result.cumulative
    if len(equity) > 200:
        equity = equity.iloc[::len(equity)//200]
    equity_data = [
        {"date": str(d.date()), "value": round(float(v), 4)}
        for d, v in equity.items()
    ]

    return {
        **_clean_dict(summary),
        "equity_curve_sample": equity_data,
        "n_data_points": len(result.returns),
    }


def _tool_optimize_portfolio(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.portfolio.optimizer import (
        ledoit_wolf_covariance, max_sharpe_portfolio,
        min_variance_portfolio, risk_parity_portfolio,
        portfolio_performance,
    )

    md       = _md(cache_dir)
    tickers  = inp["tickers"]
    method   = inp.get("method", "max_sharpe")
    lookback = int(inp.get("lookback_days", 504))
    start    = inp.get("start_date", "2018-01-01")

    returns  = md.get_returns(tickers, start=start)
    returns  = returns.dropna(how="all").iloc[-lookback:]
    cov      = ledoit_wolf_covariance(returns)
    exp_ret  = returns.mean() * 252   # annualised

    if method == "max_sharpe":
        weights = max_sharpe_portfolio(exp_ret, cov)
    elif method == "min_variance":
        weights = min_variance_portfolio(cov)
    elif method == "risk_parity":
        weights = risk_parity_portfolio(cov)
    else:
        weights = min_variance_portfolio(cov)

    perf = portfolio_performance(weights, exp_ret, cov)
    w_dict = weights[weights > 0.005].round(4).to_dict()

    return {
        "method":   method,
        "weights":  {k: round(float(v), 4) for k, v in w_dict.items()},
        **perf,
        "n_tickers_held": int((weights > 0.005).sum()),
    }


def _tool_compute_risk(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.risk.analytics import full_risk_report, var_table
    from arbitron.backtest.metrics import compute_metrics

    md      = _md(cache_dir)
    tickers = inp["tickers"]
    weights_inp = inp.get("weights")
    start   = inp.get("start_date", "2018-01-01")
    end     = inp.get("end_date")

    returns = md.get_returns(tickers, start=start, end=None)
    if end:
        returns = returns[returns.index <= pd.Timestamp(end)]
    mkt_ret = returns.mean(axis=1)

    missing: list[str] = []
    if weights_inp:
        w = pd.Series(weights_inp, dtype=float)
        missing = sorted(set(w.index) - set(returns.columns))
        w = w.drop(missing)
        if w.empty:
            raise ValueError(f"none of the weighted tickers have data: {missing}")
        # Scale by gross exposure, not the net sum: a long-short book nets to
        # ~0 and dividing by it would blow the weights up.
        gross = w.abs().sum()
        if gross == 0:
            raise ValueError("weights are all zero")
        w = w / gross
        port_ret = returns[w.index].multiply(w, axis=1).sum(axis=1, min_count=1).dropna()
    else:
        port_ret = returns.mean(axis=1)

    perf    = compute_metrics(port_ret).to_dict()
    risk    = full_risk_report(port_ret, mkt_ret)
    vt      = var_table(port_ret).to_dict(orient="records")

    return {
        **({"ignored_tickers_without_data": missing} if missing else {}),
        "portfolio_metrics": _clean_dict(perf),
        "risk_report":       _clean_dict(risk),
        "var_table":         vt,
    }


def _tool_detect_regime(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData, DEFAULT_UNIVERSE
    from arbitron.models.regime import build_regime_detector

    md    = _md(cache_dir)
    start = inp.get("start_date", "2018-01-01")  # consistent with all other tools

    # Prefer tickers supplied by caller (already cached from fetch_universe).
    # Fall back to a slice of DEFAULT_UNIVERSE (same start/end=None as other tools).
    tickers = inp.get("tickers") or DEFAULT_UNIVERSE[:15]
    prices  = md.get_prices(tickers, start=start, end=None)
    returns = prices.pct_change(fill_method=None).dropna().mean(axis=1)

    detector = build_regime_detector(use_hmm=True)
    detector.fit(returns)

    current  = detector.current_regime(returns)
    proba_df = detector.predict_proba(returns)
    stats    = detector.regime_stats(returns)

    current_proba = {}
    if not proba_df.empty:
        row = proba_df.iloc[-1]
        current_proba = {k: round(float(v), 3) for k, v in row.items()}

    regime_stats = {}
    if not stats.empty:
        regime_stats = stats.round(4).to_dict(orient="index")

    return {
        "current_regime":       current,
        "current_probabilities": current_proba,
        "regime_statistics":    regime_stats,
        "last_date":            str(returns.index[-1].date()),
    }


def _tool_compare_factors(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.factors.library import compute_factor
    from arbitron.factors.evaluation import factor_summary

    md            = _md(cache_dir)
    tickers       = inp["tickers"]
    factor_names  = inp["factor_names"]
    start         = inp.get("start_date", "2018-01-01")
    horizon       = int(inp.get("horizon_days", 21))

    prices = md.get_prices(tickers, start=start)
    rows   = []
    for fname in factor_names:
        try:
            factor = compute_factor(fname, prices)
            result = factor_summary(fname, factor, prices, horizon)
            rows.append(_clean_dict(result))
        except Exception as e:
            rows.append({"factor": fname, "error": str(e)})

    return {
        "comparison": rows,
        "n_factors":  len(rows),
        "tickers":    tickers,
        "horizon_days": horizon,
    }


def _tool_run_quant_lab(inp: dict, cache_dir: str) -> dict:
    from arbitron.data.market import MarketData
    from arbitron.lab import run_flagship_research_lab

    md = _md(cache_dir)
    tickers = inp["tickers"]
    start = inp.get("start_date", "2020-01-01")
    end = inp.get("end_date")
    horizon = int(inp.get("horizon_days", 21))
    include_foundation = bool(inp.get("include_foundation", True))
    tc_bps = float(inp.get("transaction_cost_bps", 10))

    prices = md.get_prices(tickers, start=start, end=None)
    if end:
        prices = prices[prices.index <= pd.Timestamp(end)]

    report = run_flagship_research_lab(
        prices=prices,
        factor_names=inp.get("factor_names"),
        horizon=horizon,
        include_foundation=include_foundation,
        transaction_cost=tc_bps / 10_000,
    )

    champion = report.get("champion") or {}
    return _clean_json({
        "lab_verdict": report.get("lab_verdict"),
        "date_range": report.get("date_range"),
        "champion": {
            "name": champion.get("name"),
            "kind": champion.get("kind"),
            "research_score": champion.get("research_score"),
            "metrics": champion.get("metrics"),
            "audit": champion.get("audit"),
            "regime_robustness": champion.get("regime_robustness"),
        },
        "defense_summary": report.get("defense_summary"),
        "foundation": report.get("foundation"),
        "research_memo": report.get("research_memo"),
        "top_candidates": [
            {
                "name": c.get("name"),
                "kind": c.get("kind"),
                "research_score": c.get("research_score"),
                "sharpe": (c.get("metrics") or {}).get("sharpe"),
                "verdict": (c.get("audit") or {}).get("verdict"),
                "dsr": (c.get("audit") or {}).get("deflated_sharpe"),
                "pbo": (c.get("audit") or {}).get("pbo"),
            }
            for c in report.get("candidates", [])[:5]
        ],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_dict(d: dict) -> dict:
    """Convert numpy scalars and NaN to JSON-serialisable types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _clean_json(obj):
    """Recursively convert numpy scalars and NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    return obj
