"""Run outcome: what was actually planned, booked and rented, compared with the golden one.

The plan comes from the last `final_plan` Agent A delivered to Human A; bookings and rentals
come from the environment's idempotency ledgers (the side effects that really happened), so
"exactly one booking / one rental" is checked against records, not against what agents say.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from picos.domain import HumanId, ParticipantGear, Scenario
from picos.mcp_servers.common import IdempotentLedger
from picos.protocol import AgentMessage, HumanMessage

BOOKINGS_FILE = "bookings.json"
RENTALS_FILE = "rentals.json"


def _human_id(scenario: Scenario, p: ParticipantGear) -> HumanId | None:
    """A participant may be named by actor id ("human_a") or by name ("Alice")."""
    key = p.participant.strip().lower()
    for hid, human in scenario.humans.items():
        if key in (hid, human.name.lower()):
            return hid
    return None


def extract_outcome(
    scenario: Scenario, messages: Iterable[AgentMessage | HumanMessage], env_dir: Path
) -> dict[str, Any]:
    bookings = IdempotentLedger(env_dir / BOOKINGS_FILE, "BKG").records()
    rentals = IdempotentLedger(env_dir / RENTALS_FILE, "RNT").records()
    rented = sorted({item for r in rentals for item in r["items"]})
    plans = [
        m.final_plan
        for m in messages
        if isinstance(m, HumanMessage)
        and m.final_plan is not None
        and (m.sender, m.recipient) == ("agent_a", "human_a")
    ]
    outcome: dict[str, Any] = {
        "final_plan_delivered": bool(plans),
        "bookings": len(bookings),
        "rentals": len(rentals),
        "rented_items": rented,
    }
    if not plans:
        return outcome | {"matches_expected": False}

    plan = plans[-1]
    booking = next((b for b in bookings if b["booking_id"] == plan.booking_id), None)
    confirmed = (
        booking is not None
        and booking["status"] == "confirmed"
        and booking["trail_id"] == plan.trail_id
        and booking["date"] == plan.date.isoformat()
    )
    must_bring = {
        hid: sorted(p.must_bring)
        for p in plan.participants
        if (hid := _human_id(scenario, p)) is not None
    }
    expected = scenario.expected_outcome
    matches = (
        plan.date == expected.date
        and plan.trail_id == expected.trail_id
        and confirmed == expected.booking_confirmed
        and len(bookings) == (1 if expected.booking_confirmed else 0)
        and rented == sorted(expected.rented_items)
        and len(rentals) == (1 if expected.rented_items else 0)
        and all(must_bring.get(h) == sorted(v) for h, v in expected.must_bring.items())
    )
    return outcome | {
        "date": plan.date.isoformat(),
        "trail_id": plan.trail_id,
        "booking_id": plan.booking_id,
        "booking_confirmed": confirmed,
        "must_bring": must_bring,
        "matches_expected": matches,
    }
