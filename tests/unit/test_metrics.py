"""compute_metrics: per-run metrics from trajectory events."""

from __future__ import annotations

from typing import Any

from picos.config import Price
from picos.runtime.metrics import compute_metrics

PRICES = {"m-agent": Price(input=1.0, output=10.0)}


def _ev(actor: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {"actor": actor, "kind": kind, **fields}


def _msg(sender: str, recipient: str) -> dict[str, Any]:
    return _ev(sender, "message", **{"from": sender, "to": recipient})


def _llm(actor: str, model: str, tin: int, tout: int, ms: float) -> dict[str, Any]:
    return _ev(actor, "llm_call", model=model, input_tokens=tin, output_tokens=tout, latency_ms=ms)


def test_metrics_from_events():
    events = [
        _msg("human_a", "agent_a"),
        _llm("agent_a", "m-agent", 1000, 100, 200.0),
        _llm("agent_a", "m-agent", 3000, 0, 400.0),
        _llm("human_a", "m-human", 50, 5, 100.0),
        _ev("agent_a", "tool_call", tool="book_trip"),
        _ev("agent_a", "tool_result", tool="book_trip", ok=False, fault="error", latency_ms=0),
        _ev("agent_a", "retry", tool="book_trip"),
        _ev("agent_a", "tool_result", tool="book_trip", ok=True, fault="ok", latency_ms=30),
        _ev("agent_a", "tool_call", tool="rank_dates"),
        _ev("agent_a", "tool_result", tool="rank_dates", ok=False, error="bad json"),
        _ev("agent_a", "retry", target="llm"),
        _msg("agent_a", "agent_b"),
        _msg("agent_a", "agent_b"),
        _ev("agent_b", "violation", rule="topology"),
        _ev("agent_b", "budget_exhausted", steps=12),
        _ev("agent_b", "turn_end", ts=12.5),
    ]
    trajectory = {
        "events": events,
        "termination": {"reason": "success", "tick": 9},
        "outcome": {"matches_expected": True},
    }
    m = compute_metrics(trajectory, PRICES)
    assert (m["termination_reason"], m["ticks"], m["success"]) == ("success", 9, True)
    assert m["outcome_matches_expected"] is True
    assert m["duration_s"] == 12.5
    a = m["llm"]["per_actor"]["agent_a"]
    assert (a["calls"], a["input_tokens"], a["output_tokens"]) == (2, 4000, 100)
    assert a["cost_usd"] == 0.005  # (4000*1 + 100*10) / 1e6
    assert m["llm"]["cost_usd"] == 0.005  # the unpriced human model is not counted ...
    assert m["llm"]["unpriced_models"] == ["m-human"]  # ... but reported
    assert m["llm"]["latency_ms"] == {"mean": 233.3, "p50_ms": 200.0, "max_ms": 400.0}
    assert m["tools"]["per_tool"]["book_trip"] == {
        "calls": 1,
        "attempts": 2,
        "errors": 1,
        "injected_faults": 1,
        "p50_ms": 15.0,
        "max_ms": 30,
    }
    assert m["tools"]["per_tool"]["rank_dates"]["p50_ms"] is None  # local tools: no latency
    assert (m["tools"]["attempts"], m["tools"]["errors"]) == (3, 2)
    assert m["tools"]["error_rate"] == 0.6667
    assert m["retries"] == {"tool": 1, "llm": 1}
    assert m["messages_per_edge"] == {"agent_a->agent_b": 2, "human_a->agent_a": 1}
    assert m["violations"] == {"total": 1, "by_rule": {"topology": 1}}
    assert m["budget_exhaustions"] == 1


def test_empty_run():
    m = compute_metrics(
        {"events": [], "termination": {"reason": "error", "tick": 0}, "outcome": {}}, PRICES
    )
    assert m["success"] is False and m["outcome_matches_expected"] is False
    assert m["tools"]["error_rate"] == 0.0 and m["llm"]["latency_ms"]["mean"] is None
