"""Seeded fault injection: reproducible per seed, retries recorded, exhaustion returned."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from picos.config import FaultRates
from picos.runtime.recorder import ListRecorder
from picos.tools.faults import FaultInjector
from tests.helpers import call, make_faults, text, toolset


def test_injector_is_a_pure_function_of_seed_actor_tool_index_attempt():
    faults = make_faults(p_error=0.3, p_timeout=0.2)
    a1, a2 = FaultInjector(faults, 7, "agent_a"), FaultInjector(faults, 7, "agent_a")
    keys = [
        (tool, i, att)
        for tool in ("get_forecast", "book_trip")
        for i in range(1, 50)
        for att in (1, 2, 3)
    ]
    seq = [a1.draw(*k) for k in keys]
    assert seq == [a2.draw(*k) for k in keys]
    # Order of drawing does not matter (parallel calls / interleaved retries).
    assert seq == [a1.draw(*k) for k in reversed(keys)][::-1]
    assert seq != [FaultInjector(faults, 8, "agent_a").draw(*k) for k in keys]
    assert seq != [FaultInjector(faults, 7, "agent_b").draw(*k) for k in keys]


def test_injector_rates_and_per_tool_overrides():
    faults = make_faults(
        p_error=0.1, p_timeout=0.05, tools={"book_trip": FaultRates(p_error=1.0, p_timeout=0.0)}
    )
    inj = FaultInjector(faults, 1, "agent_a")
    counts = Counter(inj.draw("get_trail", i, 1) for i in range(1, 20001))
    assert abs(counts["error"] / 20000 - 0.1) < 0.01
    assert abs(counts["timeout"] / 20000 - 0.05) < 0.01
    assert {inj.draw("book_trip", i, 1) for i in range(1, 100)} == {"error"}
    off = FaultInjector(make_faults(), 1, "agent_a")
    assert {off.draw("get_trail", i, 1) for i in range(1, 1000)} == {"ok"}


def test_fault_rates_sum_must_not_exceed_one():
    with pytest.raises(ValidationError, match="must not exceed 1"):
        FaultRates(p_error=0.7, p_timeout=0.4)


async def _fault_trace(scenario, spec, env_ctx, faults) -> list[tuple]:
    rec = ListRecorder()
    async with toolset(spec, scenario, env_ctx, faults=faults, recorder=rec) as t:
        for _ in range(4):
            await call(t["list_locations"])
            await call(t["get_trail"], trail_id="vega-de-ario")
    return [
        (e["kind"], e["tool"], e.get("attempt"), e.get("fault"), e.get("ok"))
        for e in rec.events
        if e["kind"] in ("tool_result", "retry")
    ]


async def test_same_seed_same_failures_different_seed_differs(scenario, specs, env):
    faults = make_faults(p_error=0.3, p_timeout=0.2)
    spec = specs["agent_a"].model_copy(
        update={"mcp_servers": {"weather": ["list_locations"], "trails": ["get_trail"]}}
    )
    first = await _fault_trace(scenario, spec, env(42), faults)
    second = await _fault_trace(scenario, spec, env(42), faults)
    other = await _fault_trace(scenario, spec, env(43), faults)
    assert any(kind == "retry" for kind, *_ in first)  # faults really happened
    assert first == second
    assert first != other


async def test_exhausted_retries_return_error_to_agent(scenario, specs, env):
    rec = ListRecorder()
    faults = make_faults(tools={"get_trail": FaultRates(p_error=1.0, p_timeout=0.0)})
    async with toolset(specs["agent_a"], scenario, env(), faults=faults, recorder=rec) as t:
        msg = await call(t["get_trail"], trail_id="vega-de-ario")
    assert msg.status == "error"
    assert "temporarily unavailable" in text(msg) and "after 3 attempts" in text(msg)
    assert [e["attempt"] for e in rec.of_kind("retry")] == [1, 2]
    assert [e["attempt"] for e in rec.of_kind("tool_result")] == [1, 2, 3]
