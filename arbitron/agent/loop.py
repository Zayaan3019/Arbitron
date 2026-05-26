"""
Arbitron autonomous quant research agent loop.

ReAct-style (Reason + Act) agent using Groq's API (free tier) with
Llama 3.3 70B and OpenAI-compatible function calling.
The agent autonomously:
  1. Plans a research strategy for a given task
  2. Calls tools (fetch data, compute factors, backtest, optimise, ...)
  3. Reflects on results and decides next action
  4. Produces a structured research report

References:
  - Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"
  - Groq API (free): https://console.groq.com
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator, Optional

from .tools import TOOL_SCHEMAS, execute_tool


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Arbitron, an autonomous quantitative research agent.

Your mission: rigorously discover, test, and validate trading signals using
a scientific, evidence-based approach.

You have access to tools for:
- Fetching market data (prices, returns) for equity universes
- Computing quantitative factors (momentum, value, volatility, quality, ...)
- Evaluating factors via Information Coefficient, Fama-MacBeth regression
- Running walk-forward backtests with realistic transaction costs
- Optimising portfolios (Mean-Variance, Max Sharpe, Risk Parity)
- Computing risk analytics (VaR, CVaR, factor exposures, drawdown metrics)
- Detecting market regimes (Bull/Bear via HMM)
- Comparing multiple factors side-by-side

Research workflow:
1. Define universe and data period
2. Compute candidate factors
3. Evaluate each factor (IC, Fama-MacBeth) to screen for predictive power
4. Backtest promising factors (IC IR > 0.3 or |t-stat| > 2)
5. Optimise portfolio using best factors
6. Compute risk metrics and stress-test
7. Synthesise findings into a structured report

Standards:
- Always use at least 2 years of data for factor evaluation
- Require IC IR > 0.3 AND t-stat > 2 for a factor to pass screening
- Report confidence intervals and statistical significance
- Flag data-mining concerns (multiple testing)
- Be honest about limitations and out-of-sample risk

When you have gathered sufficient evidence, produce a FINAL REPORT with sections:
## Summary, ## Universe & Data, ## Factor Analysis, ## Backtest Results,
## Portfolio Construction, ## Risk Analysis, ## Conclusions & Caveats
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    role: str            # "user" | "assistant" | "tool"
    content: Any         # str | list[dict]
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class ResearchReport:
    task: str
    status: str          # "complete" | "incomplete" | "error"
    iterations: int
    tool_calls_made: list[str]
    final_text: str
    messages: list[dict]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Event types for streaming
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    event_type: str      # "thinking" | "tool_call" | "tool_result" | "text" | "done" | "error"
    data: Any


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ArbitronAgent:
    """
    Autonomous quant research agent (powered by Groq — free tier).

    Usage:
        agent = ArbitronAgent(api_key="gsk_...", cache_dir=".cache")
        report = agent.run("Evaluate momentum factors in the tech sector")

    Streaming:
        for event in agent.stream("Evaluate momentum factors"):
            print(event.event_type, event.data)

    Get a free API key at: https://console.groq.com
    """

    MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 4096
    MAX_ITERATIONS = 15
    TOOL_TIMEOUT = 120    # seconds per tool call
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/arbitron",
        max_iterations: int = MAX_ITERATIONS,
        model: str = MODEL,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ) -> None:
        self.api_key        = api_key
        self.cache_dir      = cache_dir
        self.max_iterations = max_iterations
        self.model          = model
        self.on_event       = on_event   # optional sync callback

        self._client = None   # lazy init

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai is required: pip install openai")

            import os
            key = self.api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                raise ValueError(
                    "Groq API key required. Pass api_key= or set "
                    "GROQ_API_KEY environment variable. "
                    "Get a free key at https://console.groq.com"
                )
            self._client = OpenAI(api_key=key, base_url=self.GROQ_BASE_URL)
        return self._client

    # ------------------------------------------------------------------
    # Core agent loop
    # ------------------------------------------------------------------

    def run(self, task: str) -> ResearchReport:
        """
        Run the full autonomous research loop for a given task.

        Returns a ResearchReport with the final text and metadata.
        """
        events = list(self.stream(task))
        final_text = ""
        for e in events:
            if e.event_type == "text":
                final_text += e.data
            elif e.event_type == "error":
                return ResearchReport(
                    task=task,
                    status="error",
                    iterations=0,
                    tool_calls_made=[],
                    final_text=str(e.data),
                    messages=[],
                    elapsed_seconds=0.0,
                )

        # Pull metadata from done event
        done_event = next((e for e in events if e.event_type == "done"), None)
        meta = done_event.data if done_event else {}

        return ResearchReport(
            task=task,
            status=meta.get("status", "complete"),
            iterations=meta.get("iterations", 0),
            tool_calls_made=meta.get("tool_calls_made", []),
            final_text=final_text,
            messages=meta.get("messages", []),
            elapsed_seconds=meta.get("elapsed_seconds", 0.0),
        )

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """
        Stream agent events as the research loop progresses.

        Yields AgentEvent objects with event_type in:
          "thinking"   — agent is reasoning (text chunk)
          "tool_call"  — tool about to be executed
          "tool_result"— tool execution result
          "text"       — final report text chunk
          "done"       — loop complete, data has metadata
          "error"      — unrecoverable error
        """
        t0 = time.perf_counter()
        client = self._get_client()

        # OpenAI-format: system message lives in the messages list
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": task},
        ]

        tool_calls_made: list[str] = []
        iteration = 0

        try:
            while iteration < self.max_iterations:
                iteration += 1
                yield AgentEvent("thinking", f"[Iteration {iteration}] Calling model...")

                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.MAX_TOKENS,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    messages=messages,
                )

                choice       = response.choices[0]
                msg          = choice.message
                finish_reason = choice.finish_reason

                # Emit any text content the model produced
                if msg.content:
                    yield AgentEvent("thinking", msg.content)

                # --- Tool calls branch ---
                if finish_reason == "tool_calls" and msg.tool_calls:
                    # Serialize assistant message with tool_calls for history
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id":   tc.id,
                                "type": "function",
                                "function": {
                                    "name":      tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    # Execute each tool and append result messages
                    for tc in msg.tool_calls:
                        tool_name  = tc.function.name
                        tool_input = json.loads(tc.function.arguments)
                        tool_id    = tc.id

                        tool_calls_made.append(tool_name)
                        yield AgentEvent("tool_call", {
                            "tool":  tool_name,
                            "input": tool_input,
                            "id":    tool_id,
                        })

                        result = self._execute_tool_safe(tool_name, tool_input)

                        yield AgentEvent("tool_result", {
                            "tool":   tool_name,
                            "result": result,
                            "id":     tool_id,
                        })

                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tool_id,
                            "content":      json.dumps(result, default=str),
                        })

                    continue   # next iteration with tool results in history

                # --- Final response branch ---
                if finish_reason in ("stop", "end_turn") or not msg.tool_calls:
                    final_text = msg.content or ""
                    yield AgentEvent("text", final_text)
                    break

                # length limit hit — request continuation
                if finish_reason == "length":
                    messages.append({
                        "role":    "assistant",
                        "content": msg.content or "",
                    })
                    messages.append({
                        "role":    "user",
                        "content": "Please continue your response.",
                    })
                    continue

        except Exception as exc:
            yield AgentEvent("error", str(exc))
            return

        elapsed = time.perf_counter() - t0
        yield AgentEvent("done", {
            "status":          "complete" if iteration < self.max_iterations else "max_iter",
            "iterations":      iteration,
            "tool_calls_made": tool_calls_made,
            "messages":        messages,
            "elapsed_seconds": round(elapsed, 2),
        })

    # ------------------------------------------------------------------
    # Tool execution with error wrapping
    # ------------------------------------------------------------------

    def _execute_tool_safe(self, tool_name: str, tool_input: dict) -> dict:
        """Execute tool, returning error dict on failure instead of raising."""
        try:
            return execute_tool(tool_name, tool_input, cache_dir=self.cache_dir)
        except Exception as exc:
            return {
                "error": True,
                "tool":  tool_name,
                "message": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # Convenience: quick factor screen
    # ------------------------------------------------------------------

    def quick_factor_screen(
        self,
        universe: str = "SP500",
        factors: Optional[list[str]] = None,
        start_date: str = "2020-01-01",
        end_date:   str = "2024-12-31",
    ) -> ResearchReport:
        """
        Run a pre-defined factor screening task.

        Convenience wrapper for common research use cases.
        """
        if factors is None:
            factors = ["momentum_12_1", "realized_vol", "rsi_contrarian"]
        flist = ", ".join(factors)

        task = (
            f"Screen the following factors for the {universe} universe from "
            f"{start_date} to {end_date}: {flist}. "
            "For each factor: compute IC statistics and run a backtest. "
            "Recommend which factors pass the IC IR > 0.3 hurdle."
        )
        return self.run(task)

    def full_research_pipeline(
        self,
        universe: str = "SP500",
        start_date: str = "2019-01-01",
        end_date:   str = "2024-12-31",
        optimizer:  str = "max_sharpe",
    ) -> ResearchReport:
        """
        Run the complete research pipeline end-to-end.
        """
        task = (
            f"Run a complete quant research pipeline for the {universe} universe "
            f"from {start_date} to {end_date}:\n"
            "1. Fetch price data\n"
            "2. Compute all available factors\n"
            "3. Evaluate each factor with IC analysis\n"
            "4. Backtest the top 3 factors by IC IR\n"
            f"5. Construct an optimised portfolio using {optimizer}\n"
            "6. Compute full risk metrics and detect the current market regime\n"
            "7. Write a comprehensive research report with conclusions\n"
        )
        return self.run(task)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Arbitron: Autonomous Quant Research Agent")
    parser.add_argument("task", nargs="?", default=None, help="Research task description")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--screen",   action="store_true", help="Run quick factor screen")
    parser.add_argument("--universe", default="SP500",        help="Stock universe")
    parser.add_argument("--start",    default="2020-01-01",   help="Start date")
    parser.add_argument("--end",      default="2024-12-31",   help="End date")
    parser.add_argument("--cache",    default=".cache/arbitron", help="Cache directory")
    parser.add_argument("--model",    default=ArbitronAgent.MODEL, help="Groq model (default: llama-3.3-70b-versatile)")
    args = parser.parse_args()

    def on_event(event: AgentEvent):
        if event.event_type == "thinking":
            print(f"\033[90m[think] {str(event.data)[:200]}\033[0m")
        elif event.event_type == "tool_call":
            d = event.data
            print(f"\033[34m[tool → {d['tool']}] {json.dumps(d['input'])[:120]}\033[0m")
        elif event.event_type == "tool_result":
            d = event.data
            r = d['result']
            status = "✓" if not r.get("error") else "✗"
            print(f"\033[32m[result {status}] {d['tool']}\033[0m")
        elif event.event_type == "text":
            print(f"\n\033[1m{event.data}\033[0m")
        elif event.event_type == "done":
            d = event.data
            print(f"\n\033[33m[done] {d['iterations']} iters | "
                  f"{d['elapsed_seconds']}s | "
                  f"tools: {', '.join(d['tool_calls_made'])}\033[0m")
        elif event.event_type == "error":
            print(f"\033[31m[error] {event.data}\033[0m")

    agent = ArbitronAgent(cache_dir=args.cache, model=args.model, on_event=on_event)

    if args.pipeline:
        report = agent.full_research_pipeline(universe=args.universe,
                                               start_date=args.start,
                                               end_date=args.end)
    elif args.screen:
        report = agent.quick_factor_screen(universe=args.universe,
                                           start_date=args.start,
                                           end_date=args.end)
    elif args.task:
        report = agent.run(args.task)
    else:
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("RESEARCH REPORT")
    print("=" * 70)
    print(report.final_text)
    print(f"\nStatus: {report.status} | Iterations: {report.iterations} | "
          f"Time: {report.elapsed_seconds}s")


if __name__ == "__main__":
    _cli()
