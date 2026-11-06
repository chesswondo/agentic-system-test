"""Comms tools: the only way an agent talks to anyone. They queue messages in an outbox.

Recipients are fixed by the agent's spec (its own human, the other agent), never chosen by
the model. Payloads are validated against the protocol models here, so a malformed message
becomes an error tool result the model can fix; the bus still enforces the topology.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from picos.domain import FinalPlan, HumanId
from picos.protocol import AgentId, AgentMessage, HumanMessage, HumanMessageKind, Performative

END_TURN_TOOL = "end_turn"
OutboxItem = AgentMessage | HumanMessage


class Outbox:
    """Messages one agent queued during a turn; the runtime drains it after the turn.

    Message ids are `<agent>-<n>` with a per-agent counter, so they are deterministic. An id
    is consumed only when a valid message is queued, so rejected attempts leave no gaps.
    """

    def __init__(
        self,
        owner: AgentId,
        conversation_id: str,
        gate: Callable[[OutboxItem], None] | None = None,
    ) -> None:
        self.owner = owner
        self.conversation_id = conversation_id
        # The runtime sets the bus check here; a rejection (ValueError) becomes the sender's
        # error tool result.
        self.gate = gate
        self._items: list[OutboxItem] = []
        self._count = 0

    def next_msg_id(self) -> str:
        return f"{self.owner}-{self._count + 1}"

    def put(self, item: OutboxItem) -> None:
        if item.msg_id != self.next_msg_id():
            raise RuntimeError(f"out-of-order msg_id {item.msg_id!r}")
        if self.gate is not None:
            self.gate(item)
        self._count += 1
        self._items.append(item)

    def peek(self) -> list[OutboxItem]:
        return list(self._items)

    def drain(self) -> list[OutboxItem]:
        items, self._items = self._items, []
        return items


def comms_functions(agent: AgentId, human: HumanId, outbox: Outbox) -> list[Callable[..., Any]]:
    peer: AgentId = "agent_b" if agent == "agent_a" else "agent_a"

    def send_to_human(text: str, kind: HumanMessageKind = "info", plan_json: str = "") -> dict:
        """Send a message to your own human.

        kind: "question" when you need an answer, "info" for updates, "final_plan" only for
        the confirmed plan. plan_json: required with kind "final_plan" (a FinalPlan JSON
        object: date, trail_id, trail_name, booking_id, weather_summary, participants
        [{participant, has, rented, must_bring}], safety_notes); leave empty otherwise.
        """
        plan = FinalPlan.model_validate_json(plan_json) if plan_json.strip() else None
        msg = HumanMessage(
            msg_id=outbox.next_msg_id(),
            sender=agent,
            recipient=human,
            kind=kind,
            text=text,
            final_plan=plan,
        )
        outbox.put(msg)
        return {"queued": True, "msg_id": msg.msg_id, "to": human}

    def send_to_peer(
        performative: Performative, payload_json: str, text: str, in_reply_to: str = ""
    ) -> dict:
        """Send a typed protocol message to the other agent.

        payload_json: JSON object matching the performative's payload schema.
        text: one short human-readable sentence. in_reply_to: the msg_id you answer, or empty.
        """
        payload = json.loads(payload_json) if payload_json.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("payload_json must be a JSON object")
        msg = AgentMessage(
            msg_id=outbox.next_msg_id(),
            conversation_id=outbox.conversation_id,
            in_reply_to=in_reply_to or None,
            sender=agent,
            recipient=peer,
            performative=performative,
            payload=payload,
            text=text,
        )
        outbox.put(msg)
        return {"queued": True, "msg_id": msg.msg_id, "to": peer}

    def end_turn(reason: str) -> dict:
        """End your turn now (after sending your messages). reason: one short sentence."""
        return {"turn_ended": True, "reason": reason}

    return [send_to_human, send_to_peer, end_turn]
