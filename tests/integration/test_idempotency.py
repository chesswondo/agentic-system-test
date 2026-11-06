"""book_trip / rent_equipment are idempotent per key, including across lost responses."""

from __future__ import annotations

import json

from picos.config import FaultRates
from picos.runtime.recorder import ListRecorder
from picos.tools.faults import FaultInjector
from tests.helpers import call, make_faults, structured, text, toolset

BOOK = {"trail_id": "vega-de-ario", "date": "2026-11-22", "party_size": 2}


def _ledger(env_ctx, name: str) -> dict:
    return json.loads((env_ctx.state_dir / name).read_text(encoding="utf-8"))


async def test_book_trip_same_key_yields_one_booking(scenario, specs, env):
    ctx = env()
    async with toolset(specs["agent_a"], scenario, ctx) as t:
        first = structured(await call(t["book_trip"], **BOOK, idempotency_key="trip-1"))
        second = structured(await call(t["book_trip"], **BOOK, idempotency_key="trip-1"))
        conflict = await call(t["book_trip"], **{**BOOK, "party_size": 3}, idempotency_key="trip-1")
    assert first == second and first["booking_id"] == "BKG-0001"
    assert conflict.status == "error" and "already used with different arguments" in text(conflict)
    assert len(_ledger(ctx, "bookings.json")) == 1


async def test_rent_equipment_same_key_yields_one_rental(scenario, specs, env):
    ctx = env()
    args = {"items": ["trekking_poles", "gaiters"], "date": "2026-11-22"}
    async with toolset(specs["agent_b"], scenario, ctx) as t:
        first = structured(await call(t["rent_equipment"], **args, idempotency_key="rent-1"))
        # Same request, items in another order: same canonical request, same rental.
        again = structured(
            await call(
                t["rent_equipment"],
                items=["gaiters", "trekking_poles"],
                date="2026-11-22",
                idempotency_key="rent-1",
            )
        )
    assert first == again and first["rental_id"] == "RNT-0001"
    assert len(_ledger(ctx, "rentals.json")) == 1


def _seed_with_lost_first_response(faults, actor: str, tool: str) -> int:
    """Smallest seed whose first call of `tool` times out on attempt 1 and succeeds on 2."""
    for seed in range(1000):
        inj = FaultInjector(faults, seed, actor)
        if inj.draw(tool, 1, 1) == "timeout" and inj.draw(tool, 1, 2) == "ok":
            return seed
    raise AssertionError("no such seed in range")


async def test_retry_after_lost_response_does_not_double_book(scenario, specs, env):
    faults = make_faults(tools={"book_trip": FaultRates(p_error=0.0, p_timeout=0.5)})
    seed = _seed_with_lost_first_response(faults, "agent_a", "book_trip")
    ctx = env(seed)
    rec = ListRecorder()
    async with toolset(specs["agent_a"], scenario, ctx, faults=faults, recorder=rec) as t:
        booking = structured(await call(t["book_trip"], **BOOK, idempotency_key="trip-1"))
    results = [e for e in rec.of_kind("tool_result") if e["tool"] == "book_trip"]
    assert [(e["attempt"], e["fault"], e["ok"]) for e in results] == [
        (1, "timeout", False),
        (2, "ok", True),
    ]
    # The server processed both attempts, but the ledger holds exactly one booking.
    assert booking["booking_id"] == "BKG-0001"
    assert len(_ledger(ctx, "bookings.json")) == 1
