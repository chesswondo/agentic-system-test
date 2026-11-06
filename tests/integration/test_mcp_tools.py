"""Each MCP tool through the real wiring: stdio servers, adapters, interceptors (faults off)."""

from __future__ import annotations

import json

from picos.runtime.recorder import ListRecorder
from picos.tools.decision import constraints_from_profile
from tests.helpers import call, structured, text, toolset


async def test_agent_a_mcp_tools_and_decisions_reach_golden_choice(scenario, specs, env):
    rec = ListRecorder()
    async with toolset(specs["agent_a"], scenario, env(), recorder=rec) as t:
        locations = structured(await call(t["list_locations"]))["location_ids"]
        assert locations == ["central_massif", "eastern_liebana", "western_massif"]

        # Decision tools refuse to rank before the environment was observed.
        alice = (await call(t["build_constraints"])).content
        refused = await call(t["rank_dates"], constraints_json=alice, location_ids=locations)
        assert refused.status == "error" and "call get_forecast first" in text(refused)

        for loc in locations:
            fc = structured(
                await call(
                    t["get_forecast"],
                    location_id=loc,
                    start_date="2026-11-17",
                    end_date="2026-11-30",
                )
            )
            assert fc["location_id"] == loc and len(fc["days"]) == 14

        trails = structured(await call(t["search_trails"], max_difficulty="moderate"))["trails"]
        assert trails and all(tr["difficulty"] in ("easy", "moderate") for tr in trails)
        assert (
            structured(await call(t["get_trail"], trail_id="vega-de-ario"))["id"] == "vega-de-ario"
        )

        bob = constraints_from_profile(
            scenario.humans["human_b"].profile,
            scenario.humans["human_b"].calendar,
            {"avoid_exposure": True},
        ).model_dump_json()
        merged = (await call(t["merge_constraints"], constraints_json_list=[alice, bob])).content
        dates = json.loads(
            (await call(t["rank_dates"], constraints_json=merged, location_ids=locations)).content
        )
        assert dates[0]["date"] == "2026-11-22"
        # search_trails above was filtered by difficulty; fetch the rest so the ranking sees all.
        structured(await call(t["search_trails"]))
        ranked = json.loads(
            (await call(t["rank_trails"], constraints_json=merged, date="2026-11-22")).content
        )
        assert ranked["ranked"][0]["id"] == scenario.expected_outcome.trail_id
        assert ranked["excluded"]["ruta-del-cares"] == ["exposure"]

        gear = structured(
            await call(t["required_gear"], trail_id="vega-de-ario", date="2026-11-22")
        )
        assert gear["season"] == "winter"
        assert {"headlamp", "trekking_poles", "gaiters"} <= set(gear["items"])

        booking = structured(
            await call(
                t["book_trip"],
                trail_id="vega-de-ario",
                date="2026-11-22",
                party_size=2,
                idempotency_key="trip-1",
            )
        )
        assert booking["status"] == "confirmed" and booking["booking_id"] == "BKG-0001"

    calls = [e for e in rec.of_kind("tool_call")]
    results = [e for e in rec.of_kind("tool_result")]
    assert len(calls) == len(results)  # no faults: one result per call
    assert {e["tool"] for e in calls} >= {
        "list_locations",
        "get_forecast",
        "search_trails",
        "get_trail",
        "required_gear",
        "book_trip",
        "rank_dates",
    }


async def test_agent_b_rent_equipment(scenario, specs, env):
    async with toolset(specs["agent_b"], scenario, env()) as t:
        rental = structured(
            await call(
                t["rent_equipment"],
                items=["trekking_poles", "gaiters"],
                date="2026-11-22",
                idempotency_key="rent-1",
            )
        )
    assert rental == {
        "rental_id": "RNT-0001",
        "status": "confirmed",
        "items": ["gaiters", "trekking_poles"],
        "date": "2026-11-22",
    }


async def test_server_side_errors_return_to_model_without_retry(scenario, specs, env):
    rec = ListRecorder()
    async with toolset(specs["agent_b"], scenario, env(), recorder=rec) as t:
        bad = await call(
            t["rent_equipment"], items=["boots"], date="2026-11-22", idempotency_key="k"
        )
        unknown = await call(
            t["rent_equipment"], items=["jetpack"], date="2026-11-22", idempotency_key="k"
        )
        bad_date = await call(
            t["rent_equipment"], items=["gaiters"], date="22/11", idempotency_key="k"
        )
    assert bad.status == "error" and "not available for rent" in text(bad)
    assert unknown.status == "error" and "unknown items" in text(unknown)
    assert bad_date.status == "error" and "ISO date" in text(bad_date)
    assert rec.of_kind("retry") == []
    assert [e["attempt"] for e in rec.of_kind("tool_result")] == [1, 1, 1]
