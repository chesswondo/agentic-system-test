"""sequence.mmd: the diagram uses only the Mermaid subset we emit, and that subset is valid."""

from __future__ import annotations

from picos.runtime.mermaid import sequence_diagram
from tests.helpers import assert_valid_mermaid


def _trajectory() -> dict:
    nasty = 'Plan: "Vega"; #1 {a|b} <ok> 50% \\ done\nnext line ' + "x" * 200
    return {
        "events": [
            {
                "tick": 0,
                "kind": "message",
                "from": "human_a",
                "to": "agent_a",
                "msg_id": "human_a-1",
                "message_kind": "request",
                "text": nasty,
            },
            {"tick": 1, "kind": "tool_call", "actor": "agent_a", "tool": "get_forecast"},
            {
                "tick": 1,
                "kind": "message",
                "from": "agent_a",
                "to": "agent_b",
                "msg_id": "agent_a-1",
                "performative": "REQUEST_CONSTRAINTS",
                "text": "dates?",
            },
        ],
        "termination": {"reason": "success", "tick": 9},
    }


def test_diagram_is_valid_and_has_only_messages_in_order():
    text = sequence_diagram(_trajectory(), {"human_a": "Alice", "human_b": "Bob"})
    assert_valid_mermaid(text)
    lines = text.splitlines()
    assert lines[1:5] == [
        "    participant human_a as Human A (Alice)",
        "    participant agent_a as Agent A",
        "    participant agent_b as Agent B",
        "    participant human_b as Human B (Bob)",
    ]
    arrows = [ln for ln in lines if "->>" in ln]
    assert len(arrows) == 2  # tool calls are not drawn
    assert arrows[0].startswith("    human_a->>agent_a: t0 REQUEST human_a-1 - Plan")
    assert arrows[0].endswith("...") and len(arrows[0].split(": ", 1)[1]) <= 70
    assert arrows[1] == "    agent_a->>agent_b: t1 REQUEST_CONSTRAINTS agent_a-1 - dates?"
    assert lines[-1] == "    Note over human_a,human_b: termination success at tick 9"


def test_validator_rejects_broken_diagrams():
    good = sequence_diagram(_trajectory(), {})
    assert_valid_mermaid(good)
    for broken in (
        good.replace("sequenceDiagram", "graph TD"),
        good.replace("agent_a->>agent_b", "agent_a->>carol"),
        good + "    agent_a->>agent_b: a; b\n",
    ):
        try:
            assert_valid_mermaid(broken)
        except AssertionError:
            continue
        raise AssertionError(f"validator accepted:\n{broken}")
