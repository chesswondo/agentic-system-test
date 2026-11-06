"""Local tools in-process: JSON validation errors return to the model; results are recorded."""

from __future__ import annotations

import json

import pytest

from picos.runtime.recorder import ListRecorder
from picos.tools.comms import Outbox
from picos.tools.mcp_client import build_local_tools
from picos.tools.observations import Observations
from tests.helpers import call


@pytest.fixture
def bob_tools(scenario, specs):
    rec = ListRecorder()
    tools = build_local_tools(
        specs["agent_b"], scenario, Observations(), Outbox("agent_b", "test-conv"), rec
    )
    return {t.name: t for t in tools}, rec


async def test_build_constraints_applies_elicited_override(bob_tools):
    tools, rec = bob_tools
    msg = await call(tools["build_constraints"], overrides_json='{"avoid_exposure": true}')
    data = json.loads(msg.content)
    assert data["avoid_exposure"] is True
    assert data["available_dates"] == ["2026-11-21", "2026-11-22", "2026-11-28", "2026-11-29"]
    assert [e["kind"] for e in rec.events] == ["tool_call", "tool_result"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ("{not json", "invalid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"fear": "heights"}', "unknown constraint fields"),
        ('{"max_difficulty": "extreme"}', "invalid input: max_difficulty"),
    ],
)
async def test_invalid_arguments_become_error_results(bob_tools, overrides, expected):
    tools, rec = bob_tools
    msg = await call(tools["build_constraints"], overrides_json=overrides)
    assert msg.status == "error" and expected in msg.content
    assert rec.events[-1]["ok"] is False


async def test_check_equipment_uses_own_inventory(bob_tools):
    tools, _ = bob_tools
    msg = await call(
        tools["check_equipment"], required_items=["boots", "headlamp", "trekking_poles", "gaiters"]
    )
    assert json.loads(msg.content) == {
        "has": ["boots", "headlamp"],
        "missing": ["gaiters", "trekking_poles"],
    }


async def test_private_tools_expose_only_own_human(bob_tools, scenario):
    tools, _ = bob_tools
    profile = json.loads((await call(tools["get_user_profile"])).content)
    assert profile["name"] == "Bob"
    inventory = json.loads((await call(tools["get_inventory"])).content)
    assert inventory["items"] == scenario.humans["human_b"].inventory.items
