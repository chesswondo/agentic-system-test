"""Trajectory recorder: buffers events per actor and flushes them in a deterministic order.

Actors run concurrently within a tick, and one agent's parallel tool calls run concurrently
within a step, so the wall-clock order of events is not reproducible. The recorder makes the
logged order reproducible:
- events are buffered per actor and flushed at the end of each tick in canonical actor
  order (human_a, agent_a, agent_b, human_b, then any other actor such as `runtime`);
- inside one actor's buffer, each run of consecutive events recorded during tool calls
  (tagged with `tool_slot`, the call's position in its AIMessage) is stable-sorted by that
  position. Events of one call are sequential, so their own order is kept.
Each event gets `tick`, `seq` (global, gap-free) and `ts` (seconds since the run started;
the only wall-clock field besides latencies).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, get_args

from picos.runtime.bus import ACTORS
from picos.runtime.recorder import EventKind, tool_slot

NONDETERMINISTIC_FIELDS = frozenset({"ts", "latency_ms"})

_Buffered = tuple[int | None, dict[str, Any]]


class TrajectoryRecorder:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._t0 = clock()
        self._buffers: dict[str, list[_Buffered]] = {}
        self._seq = 0
        self.events: list[dict[str, Any]] = []

    def record(self, actor: str, kind: EventKind, **fields: Any) -> None:
        event = {"actor": actor, "kind": kind, **fields, "ts": self.now()}
        self._buffers.setdefault(actor, []).append((tool_slot.get(), event))

    def now(self) -> float:
        return round(self._clock() - self._t0, 4)

    def flush(self, tick: int) -> list[dict[str, Any]]:
        """Move buffered events into `events` with tick and seq; return the new events."""
        extra = sorted(a for a in self._buffers if a not in ACTORS)
        flushed: list[dict[str, Any]] = []
        for actor in [*ACTORS, *extra]:
            for event in _ordered(self._buffers.pop(actor, [])):
                self._seq += 1
                flushed.append({"tick": tick, "seq": self._seq, **event})
        self.events.extend(flushed)
        return flushed


def _ordered(buffer: list[_Buffered]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    run: list[_Buffered] = []
    for slot, event in [*buffer, (None, {})]:  # sentinel closes the last run
        if slot is not None:
            run.append((slot, event))
            continue
        out.extend(e for _, e in sorted(run, key=lambda p: p[0]))  # stable
        run = []
        if event:
            out.append(event)
    return out


def comparable(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events without wall-clock fields, for determinism comparisons."""
    return [{k: v for k, v in e.items() if k not in NONDETERMINISTIC_FIELDS} for e in events]


TOP_LEVEL_FIELDS = (
    "run_id",
    "scenario",
    "seed",
    "config_snapshot",
    "events",
    "termination",
    "outcome",
    "metrics",
)
TERMINATION_REASONS = frozenset({"success", "quiescence", "max_ticks", "error"})


def check_trajectory(data: dict[str, Any]) -> None:
    """Raise ValueError if `data` breaks the trajectory contract (fields, event kinds, order)."""
    missing = [f for f in TOP_LEVEL_FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing top-level fields {missing}")
    if data["termination"].get("reason") not in TERMINATION_REASONS:
        raise ValueError(f"bad termination {data['termination']}")
    kinds = set(get_args(EventKind))
    last_tick = 0
    for i, event in enumerate(data["events"], start=1):
        if event.get("seq") != i:
            raise ValueError(f"event #{i}: seq {event.get('seq')} is not gap-free")
        if event.get("kind") not in kinds or not event.get("actor"):
            raise ValueError(f"event #{i}: bad kind/actor {event.get('kind')}/{event.get('actor')}")
        if event.get("tick", -1) < last_tick:
            raise ValueError(f"event #{i}: tick goes backwards")
        last_tick = event["tick"]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
