"""Message bus: the single enforcement point of the topology
(human_a <-> agent_a <-> agent_b <-> human_b) and of message
validity.

Every message, from an agent's comms tool or from a simulated human, passes `accept` at the
moment it is sent. A rejected message is recorded as a `violation` event and raises
`BusViolation` (a ValueError), which the comms tools turn into the sender's error tool
result. Accepted messages are recorded as `message` events and wait in the bus until the
runtime delivers them at the next tick, ordered by (sender canonical order, send order).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from picos.domain import ActorId
from picos.protocol import AgentMessage, HumanMessage
from picos.runtime.recorder import Recorder
from picos.tools.local import format_error

ACTORS: tuple[ActorId, ...] = ("human_a", "agent_a", "agent_b", "human_b")
ALLOWED_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("human_a", "agent_a"),
        ("agent_a", "human_a"),
        ("agent_a", "agent_b"),
        ("agent_b", "agent_a"),
        ("agent_b", "human_b"),
        ("human_b", "agent_b"),
    }
)
AGENTS = frozenset({"agent_a", "agent_b"})

Message = AgentMessage | HumanMessage


class BusViolation(ValueError):
    """A send the bus refused. The message is not delivered."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"message rejected by the bus ({rule}): {detail}")
        self.rule = rule


class MessageBus:
    def __init__(self, recorder: Recorder, max_negotiation_rounds: int) -> None:
        self.recorder = recorder
        self.max_negotiation_rounds = max_negotiation_rounds
        self._pending: list[Message] = []
        self._accepted_ids: set[str] = set()
        self._proposals: defaultdict[str, int] = defaultdict(int)
        self.log: list[Message] = []  # every accepted message, in send order

    def accept(self, item: Message) -> None:
        """Validate a message at send time; record it and queue it for the next tick."""
        try:
            self._check(item)
        except BusViolation as exc:
            self.recorder.record(
                item.sender,
                "violation",
                rule=exc.rule,
                msg_id=item.msg_id,
                **{"from": item.sender, "to": item.recipient},
                error=str(exc),
            )
            raise
        if isinstance(item, AgentMessage) and item.performative == "PROPOSAL":
            self._proposals[item.sender] += 1
        self._accepted_ids.add(item.msg_id)
        self._pending.append(item)
        self.log.append(item)
        self.recorder.record(item.sender, "message", **message_fields(item))

    def _check(self, item: Message) -> None:
        if (item.sender, item.recipient) not in ALLOWED_EDGES:
            raise BusViolation(
                "topology", f"{item.sender} -> {item.recipient} is not an allowed edge"
            )
        if item.msg_id in self._accepted_ids:
            raise BusViolation("duplicate_msg_id", f"msg_id {item.msg_id!r} was already sent")
        # Re-validate: the bus must not trust objects built without validation
        # (e.g. model_construct). A final_plan must carry a valid FinalPlan.
        try:
            type(item).model_validate(item.model_dump())
        except ValidationError as exc:
            raise BusViolation("invalid_payload", format_error(exc)) from None
        if isinstance(item, HumanMessage) and item.kind == "final_plan":
            if item.sender not in AGENTS:
                raise BusViolation("invalid_payload", "only an agent may send a final_plan")
        if isinstance(item, AgentMessage) and item.performative == "PROPOSAL":
            if self._proposals[item.sender] >= self.max_negotiation_rounds:
                raise BusViolation(
                    "negotiation_limit",
                    f"{self.max_negotiation_rounds} proposals already sent; no more rounds. "
                    "Inform your human that no plan was found.",
                )

    def has_pending(self) -> bool:
        return bool(self._pending)

    def deliver(self) -> dict[str, list[Message]]:
        """Pop all pending messages, grouped by recipient, in (sender canonical, send) order."""
        order = {a: i for i, a in enumerate(ACTORS)}
        pending = sorted(enumerate(self._pending), key=lambda p: (order[p[1].sender], p[0]))
        self._pending = []
        inboxes: dict[str, list[Message]] = {}
        for _, msg in pending:
            inboxes.setdefault(msg.recipient, []).append(msg)
        return inboxes


def message_fields(item: Message) -> dict[str, Any]:
    """Trajectory fields of a `message` event."""
    fields: dict[str, Any] = {"from": item.sender, "to": item.recipient, "msg_id": item.msg_id}
    if isinstance(item, AgentMessage):
        fields |= {
            "performative": item.performative,
            "conversation_id": item.conversation_id,
            "in_reply_to": item.in_reply_to,
            "payload": item.payload,
            "text": item.text,
        }
    else:
        fields |= {"message_kind": item.kind, "text": item.text}
        if item.final_plan is not None:
            fields["final_plan"] = item.final_plan.model_dump(mode="json")
    return fields
