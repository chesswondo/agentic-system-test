"""Toolsets follow the AgentSpec split and every tool schema is safe (flat arguments)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from picos.agents.graph import make_plan_draft_tool
from picos.config import load_config
from picos.mcp_servers import gear, trails, weather
from picos.runtime.recorder import ListRecorder
from tests.helpers import ROOT, toolset

PRIMITIVES = {"string", "integer", "number", "boolean"}
COMMON = {
    "get_user_profile",
    "get_calendar",
    "get_inventory",
    "build_constraints",
    "check_equipment",
    "send_to_human",
    "send_to_peer",
    "end_turn",
}
WORLD = {"list_locations", "get_forecast", "search_trails", "get_trail", "book_trip"}


async def test_toolsets_match_design_split(scenario, specs, env):
    async with toolset(specs["agent_a"], scenario, env()) as a:
        names_a = set(a)
    async with toolset(specs["agent_b"], scenario, env()) as b:
        names_b = set(b)
    assert names_a == COMMON | WORLD | {
        "merge_constraints",
        "rank_dates",
        "rank_trails",
        "required_gear",
    }
    assert names_b == COMMON | {"rent_equipment"}
    assert "rent_equipment" not in names_a
    assert not names_b & (
        WORLD | {"required_gear", "merge_constraints", "rank_dates", "rank_trails"}
    )


async def test_spec_errors_fail_loudly(scenario, specs, env):
    bad_local = specs["agent_b"].model_copy(update={"local_tools": ["get_calendar", "teleport"]})
    with pytest.raises(ValueError, match="unknown local tools"):
        async with toolset(bad_local, scenario, env()):
            pass
    bad_mcp = specs["agent_b"].model_copy(update={"mcp_servers": {"gear": ["book_trip"]}})
    with pytest.raises(ValueError, match="has no tools"):
        async with toolset(bad_mcp, scenario, env()):
            pass


def flat_schema_violations(params: dict[str, Any]) -> list[str]:
    """Only str/int/float/bool/enum or lists of those; no nesting, refs or unions."""
    problems = [k for k in ("$defs", "$ref", "anyOf", "oneOf", "allOf") if k in params]
    for name, prop in params.get("properties", {}).items():
        if any(k in prop for k in ("$ref", "anyOf", "oneOf", "allOf")):
            problems.append(f"{name}: composite schema")
        elif prop.get("type") in PRIMITIVES:
            continue
        elif prop.get("type") == "array" and prop.get("items", {}).get("type") in PRIMITIVES:
            continue
        else:
            problems.append(f"{name}: {prop}")
    return problems


async def test_every_tool_schema_is_flat(scenario, specs, env):
    checked = set()
    for name in ("agent_a", "agent_b"):
        async with toolset(specs[name], scenario, env()) as tools:
            for tool in tools.values():
                params = convert_to_openai_tool(tool)["function"]["parameters"]
                assert flat_schema_violations(params) == [], tool.name
                checked.add(tool.name)
    # The graph's built-in tool (its injected tool_call_id must not leak into the schema).
    draft = make_plan_draft_tool("agent_a", ListRecorder())
    params = convert_to_openai_tool(draft)["function"]["parameters"]
    assert flat_schema_violations(params) == []
    assert set(params["properties"]) == {"key", "value"}
    checked.add(draft.name)
    assert len(checked) == 19  # every distinct tool of both agents


class _Inner(BaseModel):
    x: int


def test_flat_schema_check_rejects_nested_payloads():
    def nested(payload: _Inner, maybe: int | None = None) -> str:
        """Nested."""
        return ""

    params = convert_to_openai_tool(StructuredTool.from_function(nested))["function"]["parameters"]
    assert len(flat_schema_violations(params)) == 2


async def test_fault_overrides_name_real_mcp_tools():
    known = set()
    for server in (weather, trails, gear):
        known |= {t.name for t in await server.mcp.list_tools()}
    overrides = set(load_config(ROOT / "config" / "app.yaml").faults.tools)
    assert overrides <= known
