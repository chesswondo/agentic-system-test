"""Unit tests for the deterministic decision functions (synthetic inputs, no data files)."""

from datetime import date
from typing import Any

import pytest

from picos.domain import (
    Calendar,
    Constraints,
    DailyForecast,
    DateScoring,
    GearCatalog,
    HealthLimits,
    Profile,
    Trail,
)
from picos.tools.decision import (
    check_equipment,
    constraints_from_profile,
    merge_constraints,
    rank_dates,
    rank_trails,
    required_gear,
    season_of,
    trail_exclusions,
    weather_code_class,
)

D = date


def constraints(**kw: Any) -> Constraints:
    base: dict[str, Any] = {
        "available_dates": [D(2026, 11, 21), D(2026, 11, 22)],
        "max_difficulty": "hard",
        "max_elevation_gain_m": 2000,
        "max_duration_h": 10.0,
        "avoid_exposure": False,
    }
    return Constraints(**(base | kw))


def trail(id: str, **kw: Any) -> Trail:
    base: dict[str, Any] = {
        "id": id,
        "name": id,
        "area": "a",
        "start_point": "x",
        "lat": 43.0,
        "lon": -4.0,
        "distance_km": 10.0,
        "duration_h": 4.0,
        "elevation_gain_m": 500,
        "max_altitude_m": 1000,
        "difficulty": "moderate",
        "exposure": False,
        "loop": False,
        "season_notes": "",
        "source": "test",
    }
    return Trail(**(base | kw))


def day(d: date, **kw: Any) -> DailyForecast:
    base: dict[str, Any] = {
        "date": d,
        "weather_code": 0,
        "temperature_2m_max": 10.0,
        "temperature_2m_min": 2.0,
        "precipitation_sum": 0.0,
        "precipitation_probability_max": 0,
        "snowfall_sum": 0.0,
        "wind_speed_10m_max": 10.0,
        "daylight_duration": 33000.0,
    }
    return DailyForecast(**(base | kw))


SCORING = DateScoring(
    precipitation_mm=1.0,
    precipitation_probability=5.0,
    wind_threshold_kmh=30,
    wind_per_kmh_over=0.2,
    snowfall_cm=2.0,
    weather_code_class={
        "clear": 0,
        "cloudy": 1,
        "fog": 2,
        "rain": 4,
        "snow": 6,
        "thunderstorm": 10,
    },
)


# merge_constraints

def test_merge_takes_most_restrictive_values() -> None:
    a = constraints(
        available_dates=[D(2026, 11, 22), D(2026, 11, 19), D(2026, 11, 21)],
        max_difficulty="moderate",
        max_elevation_gain_m=800,
        max_duration_h=9.0,
        notes=["knee"],
    )
    b = constraints(
        available_dates=[D(2026, 11, 21), D(2026, 11, 22), D(2026, 11, 28)],
        max_elevation_gain_m=1500,
        max_duration_h=6.0,
        avoid_exposure=True,
        notes=["heights", "knee"],
    )
    m = merge_constraints([a, b])
    assert m.available_dates == [D(2026, 11, 21), D(2026, 11, 22)]
    assert (m.max_difficulty, m.max_elevation_gain_m, m.max_duration_h) == ("moderate", 800, 6.0)
    assert m.avoid_exposure is True
    assert m.notes == ["heights", "knee"]
    assert merge_constraints([b, a]) == m  # order-independent


def test_merge_empty_date_intersection_gives_empty_dates_and_no_ranking() -> None:
    a = constraints(available_dates=[D(2026, 11, 19)])
    b = constraints(available_dates=[D(2026, 11, 21)])
    m = merge_constraints([a, b])
    assert m.available_dates == []
    assert rank_dates(m.available_dates, {"a": [day(D(2026, 11, 19))]}, SCORING) == []


def test_merge_requires_input() -> None:
    with pytest.raises(ValueError):
        merge_constraints([])


def test_constraints_from_profile_applies_overrides() -> None:
    profile = Profile(
        name="Bob",
        fitness="good",
        health_limits=HealthLimits(
            max_difficulty="hard", max_elevation_gain_m=1500, max_duration_h=9.0
        ),
    )
    cal = Calendar(available_dates=[D(2026, 11, 22), D(2026, 11, 21), D(2026, 11, 22)])
    c = constraints_from_profile(profile, cal, {"avoid_exposure": True})
    assert c.available_dates == [D(2026, 11, 21), D(2026, 11, 22)]
    assert c.avoid_exposure is True
    with pytest.raises(ValueError):
        constraints_from_profile(profile, cal, {"max_difficulty": "extreme"})


# rank_dates

def test_rank_dates_orders_by_penalty() -> None:
    fc = {
        "a": [
            day(
                D(2026, 11, 21),
                weather_code=63,
                precipitation_sum=14,
                precipitation_probability_max=90,
            ),
            day(D(2026, 11, 22), weather_code=1, precipitation_probability_max=10),
            day(D(2026, 11, 29), weather_code=3, wind_speed_10m_max=65),
        ]
    }
    ranked = rank_dates([D(2026, 11, 21), D(2026, 11, 22), D(2026, 11, 29)], fc, SCORING)
    assert [r.date for r in ranked] == [D(2026, 11, 22), D(2026, 11, 29), D(2026, 11, 21)]
    assert ranked[0].penalty == pytest.approx(0.5)
    assert ranked[1].penalty == pytest.approx(1 + 35 * 0.2)


def test_rank_dates_tie_breaks_by_earlier_date() -> None:
    fc = {"a": [day(D(2026, 11, 29)), day(D(2026, 11, 22))]}
    ranked = rank_dates([D(2026, 11, 29), D(2026, 11, 22)], fc, SCORING)
    assert [r.date for r in ranked] == [D(2026, 11, 22), D(2026, 11, 29)]
    assert ranked[0].penalty == ranked[1].penalty


def test_rank_dates_averages_areas_and_drops_dates_without_full_coverage() -> None:
    fc = {
        "a": [day(D(2026, 11, 21), precipitation_sum=4.0), day(D(2026, 11, 22))],
        "b": [day(D(2026, 11, 21))],  # no forecast for the 22nd in area b
    }
    ranked = rank_dates([D(2026, 11, 21), D(2026, 11, 22)], fc, SCORING)
    assert [r.date for r in ranked] == [D(2026, 11, 21)]
    assert ranked[0].penalty == pytest.approx(2.0)
    assert ranked[0].per_area == {"a": 4.0, "b": 0.0}


def test_rank_dates_rejects_no_areas_and_unknown_codes() -> None:
    with pytest.raises(ValueError):
        rank_dates([D(2026, 11, 22)], {}, SCORING)
    with pytest.raises(ValueError):
        weather_code_class(42)


# rank_trails

def test_rank_trails_filters_each_hard_constraint() -> None:
    c = constraints(
        max_difficulty="moderate", max_elevation_gain_m=800, max_duration_h=6.0, avoid_exposure=True
    )
    cases = {
        "difficulty": trail("t", difficulty="hard"),
        "elevation_gain": trail("t", elevation_gain_m=900),
        "duration": trail("t", duration_h=6.5),
        "exposure": trail("t", exposure=True),
        "seasonal_altitude": trail("t", max_altitude_m=1800),
    }
    for reason, t in cases.items():
        assert trail_exclusions(t, c, max_altitude_m=1700) == [reason]
        assert rank_trails([t], c, max_altitude_m=1700) == []
    assert trail_exclusions(trail("ok"), c, max_altitude_m=1700) == []
    # No seasonal limit given: altitude is not a filter.
    assert trail_exclusions(trail("t", max_altitude_m=3000), c) == []


def test_rank_trails_no_match_returns_empty() -> None:
    c = constraints(max_difficulty="easy")
    assert rank_trails([trail("a"), trail("b", difficulty="hard")], c) == []


def test_rank_trails_prefers_longest_then_lower_gain_then_id() -> None:
    c = constraints()
    trails = [
        trail("short", duration_h=2.0),
        trail("long-steep", duration_h=5.0, elevation_gain_m=700),
        trail("long-gentle-b", duration_h=5.0, elevation_gain_m=400),
        trail("long-gentle-a", duration_h=5.0, elevation_gain_m=400),
    ]
    expected = ["long-gentle-a", "long-gentle-b", "long-steep", "short"]
    assert [t.id for t in rank_trails(trails, c)] == expected
    assert [t.id for t in rank_trails(reversed(trails), c)] == expected


# required_gear / check_equipment

CATALOG = GearCatalog.model_validate(
    {
        "items": [
            {"id": i, "name": i, "category": "c", "rentable": True}
            for i in ["boots", "headlamp", "poles", "crampons"]
        ],
        "rules": [
            {"id": "base", "when": {}, "items": ["boots"], "reason": ""},
            {"id": "w", "when": {"seasons": ["winter"]}, "items": ["headlamp"], "reason": ""},
            {"id": "p1", "when": {"min_difficulty": "moderate"}, "items": ["poles"], "reason": ""},
            {"id": "p2", "when": {"min_elevation_gain_m": 500}, "items": ["poles"], "reason": ""},
            {
                "id": "ice",
                "when": {"seasons": ["winter"], "min_max_altitude_m": 1800},
                "items": ["crampons"],
                "reason": "",
            },
        ],
    }
)


def test_required_gear_applies_rules_by_season_difficulty_and_altitude() -> None:
    easy_low = trail("t", difficulty="easy", elevation_gain_m=200, max_altitude_m=800)
    assert required_gear(easy_low, "summer", CATALOG) == ["boots"]
    assert required_gear(easy_low, "winter", CATALOG) == ["boots", "headlamp"]
    # Poles via either rule, listed once.
    assert required_gear(trail("t", elevation_gain_m=600), "summer", CATALOG) == ["boots", "poles"]
    high = trail("t", max_altitude_m=2000)
    assert required_gear(high, "winter", CATALOG) == ["boots", "crampons", "headlamp", "poles"]
    assert "crampons" not in required_gear(high, "summer", CATALOG)


def test_gear_catalog_rejects_unknown_items() -> None:
    with pytest.raises(ValueError):
        GearCatalog.model_validate(
            {"items": [], "rules": [{"id": "r", "when": {}, "items": ["x"], "reason": ""}]}
        )


def test_check_equipment_splits_has_and_missing() -> None:
    res = check_equipment(["poles", "boots", "headlamp"], ["headlamp", "boots", "jacket"])
    assert res.has == ["boots", "headlamp"]
    assert res.missing == ["poles"]
    assert check_equipment([], ["boots"]).missing == []
    assert check_equipment(["boots"], []).missing == ["boots"]


def test_season_of() -> None:
    seasons = {"winter": [11, 12, 1, 2, 3], "summer": [4, 5, 6, 7, 8, 9, 10]}
    assert season_of(D(2026, 11, 22), seasons) == "winter"
    assert season_of(D(2026, 7, 1), seasons) == "summer"
    with pytest.raises(ValueError):
        season_of(D(2026, 7, 1), {"winter": [1]})
