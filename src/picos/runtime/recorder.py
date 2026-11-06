"""Recorder interface for trajectory events.

Tools, agents, humans and the resilience layer report every call, result and retry here.
`TrajectoryRecorder` (runtime/trajectory.py) is the run implementation; `ListRecorder` is
the in-memory stand-in used by tests.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Literal, Protocol

EventKind = Literal[
    "message",
    "tool_call",
    "tool_result",
    "retry",
    "llm_call",
    "turn_start",
    "turn_end",
    "note",
    "violation",
    "budget_exhausted",
    "error",
]


# Position of the tool call being executed within its AIMessage (set by the agent's tool
# guard). Parallel calls run in separate asyncio tasks, each with its own copy of the
# context, so every event recorded during a call can be attributed to it. The trajectory
# recorder uses it to order concurrent tool events deterministically.
tool_slot: ContextVar[int | None] = ContextVar("tool_slot", default=None)


class Recorder(Protocol):
    def record(self, actor: str, kind: EventKind, **fields: Any) -> None: ...


class NullRecorder:
    """Discards events (for callers that do not observe the trajectory)."""

    def record(self, actor: str, kind: EventKind, **fields: Any) -> None:
        return None


class ListRecorder:
    """Keeps events in memory, in the order they were recorded."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, actor: str, kind: EventKind, **fields: Any) -> None:
        self.events.append({"actor": actor, "kind": kind, **fields})

    def of_kind(self, kind: EventKind) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]
