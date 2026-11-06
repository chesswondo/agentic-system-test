"""Inter-actor protocol: typed agent-to-agent messages and human-facing messages."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field, model_validator

from picos.domain import ActorId, Constraints, FinalPlan, StrictModel

Performative = Literal[
    "REQUEST_CONSTRAINTS", "CONSTRAINTS", "PROPOSAL", "ACCEPT", "REJECT", "COUNTER", "INFORM"
]
AgentId = Literal["agent_a", "agent_b"]


class RequestConstraintsPayload(StrictModel):
    window_start: date
    window_end: date
    trip_type: str

    @model_validator(mode="after")
    def _window_ordered(self) -> RequestConstraintsPayload:
        if self.window_end < self.window_start:
            raise ValueError("window_end must not be before window_start")
        return self


class ProposalPayload(StrictModel):
    date: date
    trail_id: str
    trail_summary: str
    required_gear: list[str]
    proposal_round: int = Field(ge=1)


class AcceptPayload(StrictModel):
    proposal_msg_id: str
    rented_items: list[str] = Field(default_factory=list)
    rental_confirmation_id: str | None = None

    @model_validator(mode="after")
    def _confirmation_if_rented(self) -> AcceptPayload:
        if self.rented_items and not self.rental_confirmation_id:
            raise ValueError("rental_confirmation_id is required when rented_items is not empty")
        return self


ReasonCode = Literal[
    "date_unavailable", "trail_unsuitable", "gear_unavailable", "human_declined", "other"
]


class RejectPayload(StrictModel):
    reason_code: ReasonCode
    constraints: Constraints | None = None
    alternative: str | None = None


class CounterPayload(RejectPayload):
    pass


class InformPayload(StrictModel):
    details: str = ""


PAYLOAD_SCHEMAS: dict[str, type[StrictModel]] = {
    "REQUEST_CONSTRAINTS": RequestConstraintsPayload,
    "CONSTRAINTS": Constraints,
    "PROPOSAL": ProposalPayload,
    "ACCEPT": AcceptPayload,
    "REJECT": RejectPayload,
    "COUNTER": CounterPayload,
    "INFORM": InformPayload,
}


def validate_payload(performative: Performative, payload: dict[str, Any]) -> StrictModel:
    """Validate a raw payload against its performative schema; raises ValidationError."""
    return PAYLOAD_SCHEMAS[performative].model_validate(payload)


class AgentMessage(StrictModel):
    msg_id: str
    conversation_id: str
    in_reply_to: str | None = None
    sender: AgentId
    recipient: AgentId
    performative: Performative
    payload: dict[str, Any]
    text: str

    @model_validator(mode="after")
    def _check(self) -> AgentMessage:
        if self.sender == self.recipient:
            raise ValueError("sender and recipient must differ")
        # Normalize to the JSON form of the validated payload so the stored dict is canonical.
        self.payload = validate_payload(self.performative, self.payload).model_dump(mode="json")
        return self

    def typed_payload(self) -> StrictModel:
        return validate_payload(self.performative, self.payload)


HumanMessageKind = Literal["question", "info", "final_plan"]


class HumanMessage(StrictModel):
    """Free-text message between a human and their own agent (either direction)."""

    msg_id: str
    sender: ActorId
    recipient: ActorId
    kind: HumanMessageKind = "info"
    text: str
    final_plan: FinalPlan | None = None

    @model_validator(mode="after")
    def _final_plan_iff_kind(self) -> HumanMessage:
        if self.kind == "final_plan" and self.final_plan is None:
            raise ValueError("kind 'final_plan' requires a valid final_plan payload")
        if self.kind != "final_plan" and self.final_plan is not None:
            raise ValueError("final_plan payload is only allowed with kind 'final_plan'")
        return self


class HumanReply(StrictModel):
    """Structured output of a simulated human."""

    text: str
    accepts_final_plan: bool | None = None
