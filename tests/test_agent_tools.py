"""
Tool-call argument validation for the research agent.

The model writes these arguments, so they are untrusted input: every malformed
call must come back as a precise error the model can act on, and never reach
the tool with silently substituted defaults. None of these tests touch the
network: validation runs before any data is fetched.
"""

import pytest

from arbitron.agent.tools import (
    TOOL_SCHEMAS,
    ToolArgumentError,
    execute_tool,
    validate_tool_args,
)


class TestValidateToolArgs:
    def test_every_schema_builds_a_model(self):
        for tool in TOOL_SCHEMAS:
            name = tool["function"]["name"]
            required = tool["function"]["parameters"].get("required", [])
            if not required:
                assert validate_tool_args(name, {}) == {}

    def test_only_given_arguments_are_returned(self):
        out = validate_tool_args("run_backtest", {"factor_name": "macd", "tickers": ["AAPL"]})
        assert out == {"factor_name": "macd", "tickers": ["AAPL"]}

    def test_null_optional_means_not_given(self):
        out = validate_tool_args(
            "run_backtest", {"factor_name": "macd", "tickers": ["AAPL"], "end_date": None},
        )
        assert "end_date" not in out

    def test_lax_numbers_are_coerced(self):
        out = validate_tool_args(
            "evaluate_factor", {"factor_name": "macd", "tickers": ["A", "B"], "horizon_days": "21"},
        )
        assert out["horizon_days"] == 21

    @pytest.mark.parametrize("args, fragment", [
        ({"tickers": ["AAPL"]}, "factor_name"),                                    # missing required
        ({"factor_name": "macd", "tickers": ["AAPL"], "lookback": 5}, "lookback"),  # unknown key
        ({"factor_name": "macd", "tickers": "AAPL"}, "tickers"),                    # wrong type
        ({"factor_name": "macd", "tickers": []}, "tickers"),                        # empty list
        ({"factor_name": "macd", "tickers": ["A"], "start_date": "01/02/2020"}, "start_date"),
        ({"factor_name": "macd", "tickers": ["A"], "start_date": "2020-13-45"}, "not a real date"),
        ({"factor_name": "macd", "tickers": ["A"], "start_date": "2021-01-01",
          "end_date": "2020-01-01"}, "end_date must be after"),
        ({"factor_name": "magic_alpha", "tickers": ["A"]}, "unknown factor"),
        ({"factor_name": "macd", "tickers": ["A"], "transaction_cost_bps": -5}, "transaction_cost_bps"),
    ])
    def test_rejections(self, args, fragment):
        with pytest.raises(ToolArgumentError, match=fragment):
            validate_tool_args("run_backtest", args)

    def test_enum_is_enforced(self):
        with pytest.raises(ToolArgumentError, match="method"):
            validate_tool_args("optimize_portfolio", {"tickers": ["A"], "method": "yolo"})

    def test_bounds_from_prose_are_enforced(self):
        with pytest.raises(ToolArgumentError, match="n"):
            validate_tool_args("fetch_universe", {"n": 500})
        with pytest.raises(ToolArgumentError, match="horizon_days"):
            validate_tool_args("compare_factors", {
                "factor_names": ["macd"], "tickers": ["A"], "horizon_days": 0,
            })

    def test_unknown_factor_in_a_list(self):
        with pytest.raises(ToolArgumentError, match="nope"):
            validate_tool_args("compare_factors", {"factor_names": ["macd", "nope"], "tickers": ["A"]})

    def test_weights_must_be_numbers(self):
        with pytest.raises(ToolArgumentError, match="weights"):
            validate_tool_args("compute_risk_metrics", {"tickers": ["A"], "weights": {"A": "lots"}})

    def test_non_object_arguments(self):
        with pytest.raises(ToolArgumentError, match="JSON object"):
            validate_tool_args("run_backtest", ["macd"])


class TestExecuteToolRejectsBeforeRunning:
    def test_invalid_call_returns_structured_error(self, tmp_path):
        out = execute_tool("run_backtest", {"factor_name": "macd"}, cache_dir=str(tmp_path))
        assert out["invalid_arguments"] is True
        assert "tickers" in out["error"]
        assert not any(tmp_path.iterdir())   # nothing was fetched or cached

    def test_unknown_tool(self, tmp_path):
        assert "Unknown tool" in execute_tool("rm_rf", {}, cache_dir=str(tmp_path))["error"]
