"""compact_history: which tool results the LLM sees as stubs."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from picos.agents.history import compact_history
from picos.agents.spec import AgentSpec


def _tool(name: str, content: str, status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=f"id-{name}", status=status)


def test_only_successful_listed_results_from_earlier_turns_are_stubbed():
    big = "x" * 500
    messages = [
        HumanMessage("turn 1"),
        AIMessage(content=""),
        _tool("get_forecast", big),
        _tool("search_trails", "503", status="error"),
        _tool("rank_dates", big),
        HumanMessage("turn 2"),
        AIMessage(content=""),
        _tool("get_forecast", big),
    ]
    out, n = compact_history(messages, {"get_forecast", "search_trails"})
    assert n == 1
    assert out[2].text.startswith("[compacted: successful result of get_forecast")
    assert "500 chars" in out[2].text
    assert out[3].text == "503"  # errors are kept
    assert out[4].text == big  # not listed
    assert out[7].text == big  # current turn
    assert messages[2].text == big  # input not mutated
    assert out[2].tool_call_id == messages[2].tool_call_id


def test_no_tools_means_no_change():
    messages = [HumanMessage("a"), _tool("get_forecast", "data"), HumanMessage("b")]
    out, n = compact_history(messages, ())
    assert (out, n) == (messages, 0)


def test_spec_rejects_compacting_tools_it_does_not_have(specs):
    raw = specs["agent_b"].model_dump() | {"compact_tool_results": ["get_forecast"]}
    with pytest.raises(ValidationError, match="not in mcp_servers"):
        AgentSpec.model_validate(raw)
