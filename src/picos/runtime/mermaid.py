"""Mermaid sequence diagram of a run's messages, one lane per actor.

Only `message` events are drawn (tool calls stay in the trajectory). Each arrow is labelled
with the tick, the performative (agent messages) or kind (human messages), the msg_id and a
shortened text. A closing note shows the termination.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from picos.runtime.bus import ACTORS

TITLES: dict[str, str] = {
    "human_a": "Human A",
    "agent_a": "Agent A",
    "agent_b": "Agent B",
    "human_b": "Human B",
}

MAX_TEXT = 70
# Characters Mermaid treats as syntax inside a message label: ';' ends a statement, '#' starts
# an entity code, '%%' a comment, ':' is fine after the first one but kept out for safety.
_UNSAFE = re.compile(r"[;#%:<>{}\[\]\"`|\\]")


def label(text: str, limit: int = MAX_TEXT) -> str:
    """One-line label without Mermaid syntax characters, at most `limit` chars."""
    clean = re.sub(r"\s+", " ", _UNSAFE.sub(" ", text)).strip()
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def sequence_diagram(trajectory: Mapping[str, Any], human_names: Mapping[str, str]) -> str:
    """Render the diagram. `human_names` maps human actor ids to display names."""
    lines = ["sequenceDiagram"]
    for actor in ACTORS:  # canonical order = left-to-right lanes
        title, name = TITLES[actor], human_names.get(actor)
        lines.append(f"    participant {actor} as {label(f'{title} ({name})' if name else title)}")
    for e in trajectory["events"]:
        if e["kind"] != "message":
            continue
        tag = e.get("performative") or str(e.get("message_kind", "message")).upper()
        text = label(f"t{e['tick']} {tag} {e['msg_id']} - {e.get('text', '')}")
        lines.append(f"    {e['from']}->>{e['to']}: {text}")
    term = trajectory["termination"]
    note = label(f"termination {term['reason']} at tick {term['tick']}")
    lines.append(f"    Note over {ACTORS[0]},{ACTORS[-1]}: {note}")
    return "\n".join(lines) + "\n"
