"""
Process-wide research-trial registry.

The Deflated Sharpe Ratio is only meaningful if it knows how many strategies
were actually tried.  Counting only the candidates that happen to be in the
final comparison table under-counts the search: every discarded parameter
sweep, every abandoned factor, every re-run of the backtester with a
different window is a trial that inflated the winner's Sharpe.

This module keeps that count.  `run_factor_backtest` records into it
automatically, so `n_trials` reflects every backtest the process has ever
run rather than the handful that survived to the report.

Deliberately dependency-free (no imports from `arbitron.*`) so the backtest
engine can import it without a cycle.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Trial:
    """One backtest that was run."""
    name: str
    sharpe: float


class TrialRegistry:
    """
    Thread-safe counter of every backtest run in this process.

    Persistence is opt-in via the ``ARBITRON_TRIAL_LOG`` environment
    variable (a JSON file path).  When set, the count survives restarts —
    which is the honest behaviour for a long-running research effort, since
    the multiple-testing burden does not reset when you restart Python.
    """

    #: Hard ceiling on retained trial records.  The COUNT keeps growing past
    #: this; only the per-trial detail is capped so memory stays bounded.
    MAX_RETAINED: int = 5_000

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._trials: list[Trial] = []
        self._count = 0
        self._persist_path = persist_path
        self._load()

    # ------------------------------------------------------------------

    def record(self, name: str, sharpe: float) -> int:
        """Record one completed backtest.  Returns the new total count."""
        try:
            sharpe = float(sharpe)
        except (TypeError, ValueError):
            sharpe = float("nan")
        with self._lock:
            self._count += 1
            self._trials.append(Trial(str(name), sharpe))
            if len(self._trials) > self.MAX_RETAINED:
                del self._trials[: len(self._trials) - self.MAX_RETAINED]
            self._save_locked()
            return self._count

    def count(self) -> int:
        """Total number of backtests ever run (never truncated)."""
        with self._lock:
            return self._count

    def sharpes(self) -> list[float]:
        """Annualised Sharpe of every retained trial."""
        with self._lock:
            return [t.sharpe for t in self._trials]

    def trials(self) -> list[Trial]:
        with self._lock:
            return list(self._trials)

    def reset(self) -> None:
        """Clear the registry.  Used by tests to keep DSR deterministic."""
        with self._lock:
            self._trials.clear()
            self._count = 0
            self._save_locked()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n_trials": self._count,
                "n_retained": len(self._trials),
                "persisted": bool(self._persist_path),
            }

    # ------------------------------------------------------------------
    # Optional disk persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            self._count = int(blob.get("count", 0))
            self._trials = [
                Trial(str(t.get("name", "?")), float(t.get("sharpe", "nan")))
                for t in blob.get("trials", [])
            ][-self.MAX_RETAINED:]
        except Exception:
            # A corrupt log must never break a research run; start clean.
            self._count = 0
            self._trials = []

    def _save_locked(self) -> None:
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({
                    "count": self._count,
                    "trials": [{"name": t.name, "sharpe": t.sharpe} for t in self._trials],
                }),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            pass  # persistence is best-effort


#: Process-wide singleton.  Import this, not a fresh TrialRegistry.
TRIALS = TrialRegistry(persist_path=os.environ.get("ARBITRON_TRIAL_LOG") or None)


def record_trial(name: str, sharpe: float) -> int:
    """Record a completed backtest in the process-wide registry."""
    return TRIALS.record(name, sharpe)


def trial_count() -> int:
    """How many backtests this process has run."""
    return TRIALS.count()
