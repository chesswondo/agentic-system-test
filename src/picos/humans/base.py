"""A simulated human: one activation produces exactly one reply to its agent.

The reply content comes from a provider (`scripted` rule table or `llm` persona); this
wrapper owns the message ids, the trajectory events and the bus hand-off, so both providers
behave identically towards the runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from picos.domain import HumanId
from picos.protocol import AgentId, HumanMessage, HumanReply
from picos.runtime.recorder import Recorder

AGENT_OF: dict[HumanId, AgentId] = {"human_a": "agent_a", "human_b": "agent_b"}


class HumanProvider(Protocol):
    async def reply(self, inbox: Sequence[HumanMessage]) -> HumanReply: ...


@dataclass(frozen=True)
class HumanActivation:
    message: HumanMessage
    reply: HumanReply
    replied_to_final_plan: bool  # the inbox contained a final_plan

    @property
    def accepted_final_plan(self) -> bool:
        return self.replied_to_final_plan and self.reply.accepts_final_plan is True


class SimulatedHuman:
    def __init__(
        self,
        actor: HumanId,
        provider: HumanProvider,
        recorder: Recorder,
        send: Callable[[HumanMessage], None],
    ) -> None:
        self.actor = actor
        self.agent = AGENT_OF[actor]
        self.provider = provider
        self.recorder = recorder
        self.send = send  # the bus check; raises BusViolation on rejection
        self._count = 0

    def _message(self, text: str) -> HumanMessage:
        self._count += 1
        return HumanMessage(
            msg_id=f"{self.actor}-{self._count}",
            sender=self.actor,
            recipient=self.agent,
            kind="info",
            text=text,
        )

    def initial(self, text: str) -> HumanMessage:
        """The scenario's opening message (tick 0)."""
        msg = self._message(text)
        self.send(msg)
        return msg

    async def activate(self, inbox: Sequence[HumanMessage]) -> HumanActivation:
        if not inbox:
            raise ValueError("a human activation needs at least one inbox item")
        self.recorder.record(self.actor, "turn_start", inbox=[m.msg_id for m in inbox])
        reply = await self.provider.reply(inbox)
        msg = self._message(reply.text)
        self.send(msg)
        activation = HumanActivation(
            message=msg,
            reply=reply,
            replied_to_final_plan=any(m.kind == "final_plan" for m in inbox),
        )
        self.recorder.record(
            self.actor,
            "turn_end",
            reason="reply",
            accepts_final_plan=reply.accepts_final_plan,
        )
        return activation
