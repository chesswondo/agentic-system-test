"""Protocol models: per-performative payload validation and human message invariants."""

from typing import Any

import pytest
from pydantic import ValidationError

from picos.protocol import AgentMessage, HumanMessage, ProposalPayload, validate_payload


def msg(performative: str, payload: dict[str, Any], **kw: Any) -> AgentMessage:
    base: dict[str, Any] = {
        "msg_id": "m1",
        "conversation_id": "c1",
        "sender": "agent_a",
        "recipient": "agent_b",
        "performative": performative,
        "payload": payload,
        "text": "t",
    }
    return AgentMessage(**(base | kw))


def test_valid_proposal_is_normalized_to_json() -> None:
    m = msg(
        "PROPOSAL",
        {
            "date": "2026-11-22",
            "trail_id": "vega-de-ario",
            "trail_summary": "s",
            "required_gear": ["boots"],
            "proposal_round": 1,
        },
    )
    assert m.payload["date"] == "2026-11-22"
    assert isinstance(m.typed_payload(), ProposalPayload)


@pytest.mark.parametrize(
    ("performative", "payload"),
    [
        ("PROPOSAL", {"date": "2026-11-22"}),  # missing fields
        ("CONSTRAINTS", {"available_dates": [], "max_difficulty": "extreme"}),
        ("ACCEPT", {"proposal_msg_id": "m1", "rented_items": ["gaiters"]}),  # no confirmation
        (
            "REQUEST_CONSTRAINTS",
            {"window_start": "2026-11-30", "window_end": "2026-11-17", "trip_type": "day_hike"},
        ),
        ("INFORM", {"unexpected": 1}),  # extra fields are rejected
    ],
)
def test_invalid_payloads_rejected(performative: str, payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        msg(performative, payload)


def test_self_addressed_message_rejected() -> None:
    with pytest.raises(ValidationError):
        msg("INFORM", {}, recipient="agent_a")


def test_validate_payload_counter_carries_constraints() -> None:
    p = validate_payload(
        "COUNTER",
        {
            "reason_code": "trail_unsuitable",
            "constraints": {
                "available_dates": ["2026-11-22"],
                "max_difficulty": "easy",
                "max_elevation_gain_m": 500,
                "max_duration_h": 4,
                "avoid_exposure": True,
            },
        },
    )
    assert p.constraints is not None and p.constraints.max_difficulty == "easy"  # type: ignore[attr-defined]


def test_final_plan_kind_requires_payload_and_vice_versa() -> None:
    base = {"msg_id": "h1", "sender": "agent_a", "recipient": "human_a", "text": "plan"}
    with pytest.raises(ValidationError):
        HumanMessage(**base, kind="final_plan")
    plan = {
        "date": "2026-11-22",
        "trail_id": "vega-de-ario",
        "trail_name": "Vega de Ario",
        "booking_id": "B1",
        "weather_summary": "clear",
        "participants": [{"participant": "Alice", "must_bring": ["headlamp"]}],
    }
    assert HumanMessage(**base, kind="final_plan", final_plan=plan).final_plan is not None
    with pytest.raises(ValidationError):
        HumanMessage(**base, kind="info", final_plan=plan)
