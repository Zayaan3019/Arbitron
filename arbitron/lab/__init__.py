"""Research-lab orchestration, statistical defenses, and audit gates."""

from .research_lab import run_flagship_research_lab
from .statistical_defense import (
    StatisticalDefenseConfig,
    audit_strategy,
    deflated_sharpe_ratio,
    estimate_backtest_overfitting,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "StatisticalDefenseConfig",
    "audit_strategy",
    "deflated_sharpe_ratio",
    "estimate_backtest_overfitting",
    "probabilistic_sharpe_ratio",
    "run_flagship_research_lab",
]
