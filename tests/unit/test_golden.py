"""Golden outcome from scenario + data, using only the deterministic decision functions."""

from datetime import date

import pytest

from picos.domain import Constraints, DailyForecast, GearCatalog, Scenario, Trail, load_scenario
from picos.tools.datasets import load_gear_catalog, load_trails, load_weather_snapshot
from picos.tools.decision import (
    check_equipment,
    constraints_from_profile,
    merge_constraints,
    rank_dates,
    rank_trails,
    required_gear,
    season_of,
    trail_exclusions,
)
from tests.helpers import ROOT


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return load_scenario(ROOT / "scenarios" / "default.yaml")


@pytest.fixture(scope="module")
def trails() -> list[Trail]:
    return load_trails(ROOT / "data")


@pytest.fixture(scope="module")
def catalog() -> GearCatalog:
    return load_gear_catalog(ROOT / "data")


@pytest.fixture(scope="module")
def weather() -> dict[str, list[DailyForecast]]:
    return load_weather_snapshot(ROOT / "data")


def person_constraints(scenario: Scenario, human: str, with_hidden: bool = True) -> Constraints:
    h = scenario.humans[human]  # type: ignore[index]
    overrides = {k: v for f in h.persona.hidden_facts for k, v in f.implies.items()}
    return constraints_from_profile(h.profile, h.calendar, overrides if with_hidden else None)


def merged(scenario: Scenario) -> Constraints:
    return merge_constraints(
        [person_constraints(scenario, "human_a"), person_constraints(scenario, "human_b")]
    )


def season_limit(scenario: Scenario, d: date) -> int | None:
    season = season_of(d, scenario.planning.seasons)
    return scenario.planning.max_altitude_by_season_m.get(season)


def test_data_files_are_valid_and_large_enough(trails, catalog, weather, scenario) -> None:
    assert len(trails) >= 10
    assert {t.area for t in trails} <= set(weather)
    for days in weather.values():
        assert [d.date for d in days][0] == scenario.window_start
        assert [d.date for d in days][-1] == scenario.window_end


def test_golden_outcome(scenario, trails, catalog, weather) -> None:
    exp = scenario.expected_outcome
    m = merged(scenario)

    dates = rank_dates(m.available_dates, weather, scenario.planning.date_scoring)
    best_date = dates[0].date
    assert best_date == exp.date

    ranked = rank_trails(trails, m, season_limit(scenario, best_date))
    best_trail = ranked[0]
    assert best_trail.id == exp.trail_id

    gear = required_gear(best_trail, season_of(best_date, scenario.planning.seasons), catalog)
    alice = check_equipment(gear, scenario.humans["human_a"].inventory.items)
    bob = check_equipment(gear, scenario.humans["human_b"].inventory.items)
    assert bob.missing == sorted(exp.rented_items)
    assert {"human_a": alice.missing} == exp.must_bring
    # Bob's missing items can actually be rented.
    rentable = {i.id for i in catalog.items if i.rentable}
    assert set(bob.missing) <= rentable


def test_golden_is_robust_to_single_area_weather(scenario, weather) -> None:
    """Whichever area the agent fetches, the best common date is the same."""
    m = merged(scenario)
    for area, days in weather.items():
        best = rank_dates(m.available_dates, {area: days}, scenario.planning.date_scoring)[0]
        assert best.date == scenario.expected_outcome.date, area


def test_cares_excluded_only_by_exposure(scenario, trails) -> None:
    m = merged(scenario)
    limit = season_limit(scenario, scenario.expected_outcome.date)
    cares = next(t for t in trails if t.id == "ruta-del-cares")
    assert trail_exclusions(cares, m, limit) == ["exposure"]
    assert "ruta-del-cares" not in [t.id for t in rank_trails(trails, m, limit)]
    # Without Bob's hidden fact, Cares would win: the elicited constraint changes the outcome.
    no_hidden = merge_constraints(
        [person_constraints(scenario, "human_a"), person_constraints(scenario, "human_b", False)]
    )
    assert rank_trails(trails, no_hidden, limit)[0].id == "ruta-del-cares"


def test_best_weather_date_excluded_by_bob_availability(scenario, weather) -> None:
    scoring = scenario.planning.date_scoring
    all_dates = [d.date for d in next(iter(weather.values()))]
    overall_best = rank_dates(all_dates, weather, scoring)[0]
    common_best = rank_dates(merged(scenario).available_dates, weather, scoring)[0]
    assert overall_best.date != scenario.expected_outcome.date
    assert overall_best.penalty < common_best.penalty
    assert overall_best.date.weekday() < 5  # a weekday
    assert overall_best.date in scenario.humans["human_a"].calendar.available_dates
    assert overall_best.date not in scenario.humans["human_b"].calendar.available_dates
    assert overall_best.date not in merged(scenario).available_dates
