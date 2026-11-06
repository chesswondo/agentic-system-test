"""Deterministic, outcome-defining decision functions (pure).

The agents reach them through the wrappers in `decision_tools.py`.

Every ranking has an explicit total-order tie-break so results never depend on input order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from picos.domain import (
    DIFFICULTY_ORDER,
    Calendar,
    Constraints,
    DailyForecast,
    DateScoring,
    GearCatalog,
    GearCondition,
    Profile,
    StrictModel,
    Trail,
)


class DateScore(StrictModel):
    date: date
    penalty: float  # lower is better; mean over areas
    per_area: dict[str, float]


class EquipmentCheck(StrictModel):
    has: list[str]
    missing: list[str]


def constraints_from_profile(
    profile: Profile, calendar: Calendar, overrides: Mapping[str, Any] | None = None
) -> Constraints:
    """Build one person's Constraints from structured profile data plus elicited overrides."""
    limits = profile.health_limits
    data: dict[str, Any] = {
        "available_dates": sorted(set(calendar.available_dates)),
        "max_difficulty": limits.max_difficulty,
        "max_elevation_gain_m": limits.max_elevation_gain_m,
        "max_duration_h": limits.max_duration_h,
        "avoid_exposure": limits.avoid_exposure,
        "notes": sorted(set(limits.notes)),
    }
    data.update(overrides or {})
    return Constraints.model_validate(data)


def merge_constraints(parts: Iterable[Constraints]) -> Constraints:
    """Most restrictive combination: date intersection, min limits, OR of exposure avoidance.

    An empty date intersection yields `available_dates == []` (not an error), so callers can
    report "no common date" instead of crashing.
    """
    items = list(parts)
    if not items:
        raise ValueError("merge_constraints needs at least one Constraints")
    dates = set.intersection(*(set(c.available_dates) for c in items))
    return Constraints(
        available_dates=sorted(dates),
        max_difficulty=min((c.max_difficulty for c in items), key=DIFFICULTY_ORDER.__getitem__),
        max_elevation_gain_m=min(c.max_elevation_gain_m for c in items),
        max_duration_h=min(c.max_duration_h for c in items),
        avoid_exposure=any(c.avoid_exposure for c in items),
        notes=sorted({n for c in items for n in c.notes}),
    )


def weather_code_class(code: int) -> str:
    """Map a WMO weather code to a coarse class (WMO 4677 subset used by Open-Meteo)."""
    if code in (0, 1):
        return "clear"
    if code in (2, 3):
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "rain"
    if 71 <= code <= 77 or code in (85, 86):
        return "snow"
    if 95 <= code <= 99:
        return "thunderstorm"
    raise ValueError(f"unknown WMO weather code: {code}")


def day_penalty(day: DailyForecast, scoring: DateScoring) -> float:
    """Weather penalty of one day in one area; lower is better."""
    cls = weather_code_class(day.weather_code)
    if cls not in scoring.weather_code_class:
        raise ValueError(f"date_scoring.weather_code_class has no weight for {cls!r}")
    wind_over = max(0.0, day.wind_speed_10m_max - scoring.wind_threshold_kmh)
    return (
        scoring.precipitation_mm * day.precipitation_sum
        + scoring.precipitation_probability * day.precipitation_probability_max / 100
        + scoring.wind_per_kmh_over * wind_over
        + scoring.snowfall_cm * day.snowfall_sum
        + scoring.weather_code_class[cls]
    )


def rank_dates(
    dates: Iterable[date],
    forecasts_by_area: Mapping[str, Iterable[DailyForecast]],
    scoring: DateScoring,
) -> list[DateScore]:
    """Rank candidate dates by mean weather penalty over the given areas.

    Dates missing from any area's forecast are dropped (no data, no recommendation).
    Tie-break: earlier date first.
    """
    if not forecasts_by_area:
        raise ValueError("rank_dates needs forecasts for at least one area")
    by_area = {area: {d.date: d for d in days} for area, days in forecasts_by_area.items()}
    scores: list[DateScore] = []
    for d in sorted(set(dates)):
        if not all(d in days for days in by_area.values()):
            continue
        per_area = {
            area: round(day_penalty(days[d], scoring), 6) for area, days in sorted(by_area.items())
        }
        penalty = round(sum(per_area.values()) / len(per_area), 6)
        scores.append(DateScore(date=d, penalty=penalty, per_area=per_area))
    return sorted(scores, key=lambda s: (s.penalty, s.date))


def season_of(d: date, seasons: Mapping[str, list[int]]) -> str:
    for name, months in seasons.items():
        if d.month in months:
            return name
    raise ValueError(f"no season covers month {d.month}")


def trail_exclusions(
    trail: Trail, constraints: Constraints, max_altitude_m: int | None = None
) -> list[str]:
    """Reasons a trail fails the hard filters (empty list means it is acceptable)."""
    reasons: list[str] = []
    if DIFFICULTY_ORDER[trail.difficulty] > DIFFICULTY_ORDER[constraints.max_difficulty]:
        reasons.append("difficulty")
    if trail.elevation_gain_m > constraints.max_elevation_gain_m:
        reasons.append("elevation_gain")
    if trail.duration_h > constraints.max_duration_h:
        reasons.append("duration")
    if constraints.avoid_exposure and trail.exposure:
        reasons.append("exposure")
    if max_altitude_m is not None and trail.max_altitude_m > max_altitude_m:
        reasons.append("seasonal_altitude")
    return reasons


def rank_trails(
    trails: Iterable[Trail], constraints: Constraints, max_altitude_m: int | None = None
) -> list[Trail]:
    """Acceptable trails, best first.

    "Optimal length" = the fullest day the limits allow: longest duration first, then lower
    elevation gain (gentler on knees), then id.
    """
    ok = [t for t in trails if not trail_exclusions(t, constraints, max_altitude_m)]
    return sorted(ok, key=lambda t: (-t.duration_h, t.elevation_gain_m, t.id))


def _condition_holds(cond: GearCondition, trail: Trail, season: str) -> bool:
    if cond.seasons is not None and season not in cond.seasons:
        return False
    if cond.min_difficulty is not None and (
        DIFFICULTY_ORDER[trail.difficulty] < DIFFICULTY_ORDER[cond.min_difficulty]
    ):
        return False
    if cond.min_elevation_gain_m is not None and trail.elevation_gain_m < cond.min_elevation_gain_m:
        return False
    if cond.min_max_altitude_m is not None and trail.max_altitude_m < cond.min_max_altitude_m:
        return False
    return True


def required_gear(trail: Trail, season: str, catalog: GearCatalog) -> list[str]:
    """Union of items of all catalog rules that apply to this trail and season, sorted by id."""
    return sorted(
        {i for r in catalog.rules if _condition_holds(r.when, trail, season) for i in r.items}
    )


def check_equipment(required: Iterable[str], inventory: Iterable[str]) -> EquipmentCheck:
    """Split required items into what the person has and what is missing (both sorted)."""
    req, inv = set(required), set(inventory)
    return EquipmentCheck(has=sorted(req & inv), missing=sorted(req - inv))
