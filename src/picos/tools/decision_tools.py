"""Deterministic decision functions (tools/decision.py) exposed as safe tools.

Arguments are flat; structured inputs (Constraints) are JSON strings validated by pydantic.
Environment data comes from the agent's own `Observations`, never from LLM-copied blobs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

from picos.domain import Constraints, HumanSpec, PlanningParams
from picos.tools import decision
from picos.tools.observations import Observations


def _constraints(raw: str) -> Constraints:
    return Constraints.model_validate_json(raw)


def decision_functions(
    human: HumanSpec, planning: PlanningParams, observations: Observations
) -> list[Callable[..., Any]]:
    def build_constraints(overrides_json: str = "") -> Constraints:
        """Your human's trip constraints from their profile and calendar.

        overrides_json: optional JSON object of Constraints fields learned from your human in
        conversation, e.g. {"avoid_exposure": true}. Allowed fields: available_dates,
        max_difficulty, max_elevation_gain_m, max_duration_h, avoid_exposure, notes.
        """
        overrides: Any = json.loads(overrides_json) if overrides_json.strip() else {}
        if not isinstance(overrides, dict):
            raise ValueError("overrides_json must be a JSON object")
        unknown = sorted(set(overrides) - set(Constraints.model_fields))
        if unknown:
            raise ValueError(f"unknown constraint fields: {unknown}")
        return decision.constraints_from_profile(human.profile, human.calendar, overrides)

    def merge_constraints(constraints_json_list: list[str]) -> Constraints:
        """Combine several people's Constraints (each a JSON string) into the most
        restrictive joint Constraints: common dates, lowest limits, exposure avoided if
        anyone avoids it."""
        return decision.merge_constraints(_constraints(c) for c in constraints_json_list)

    def rank_dates(constraints_json: str, location_ids: list[str]) -> list[decision.DateScore]:
        """Rank the available dates of the given Constraints by weather, best first, using
        forecasts you already fetched with get_forecast for these location ids."""
        constraints = _constraints(constraints_json)
        if not location_ids:
            raise ValueError("location_ids must not be empty")
        missing = sorted(set(location_ids) - set(observations.forecasts))
        if missing:
            raise ValueError(f"no forecast fetched yet for {missing}; call get_forecast first")
        forecasts = {loc: list(observations.forecasts[loc].values()) for loc in location_ids}
        return decision.rank_dates(constraints.available_dates, forecasts, planning.date_scoring)

    def rank_trails(constraints_json: str, date: str) -> dict[str, Any]:
        """Rank the trails you already fetched (search_trails/get_trail) against the given
        Constraints for the given ISO date, best first, with exclusion reasons for the rest."""
        constraints = _constraints(constraints_json)
        if not observations.trails:
            raise ValueError("no trails fetched yet; call search_trails first")
        season = decision.season_of(_parse_date(date), planning.seasons)
        max_alt = planning.max_altitude_by_season_m.get(season)
        trails = sorted(observations.trails.values(), key=lambda t: t.id)
        ranked = decision.rank_trails(trails, constraints, max_alt)
        excluded = {
            t.id: reasons
            for t in trails
            if (reasons := decision.trail_exclusions(t, constraints, max_alt))
        }
        return {
            "season": season,
            "ranked": [t.model_dump(mode="json") for t in ranked],
            "excluded": excluded,
        }

    def check_equipment(required_items: list[str]) -> decision.EquipmentCheck:
        """Split required gear item ids into what your human has and what is missing."""
        return decision.check_equipment(required_items, human.inventory.items)

    return [build_constraints, merge_constraints, rank_dates, rank_trails, check_equipment]


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"date must be an ISO date YYYY-MM-DD, got {value!r}") from None
