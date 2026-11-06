"""Domain models: constraints, trails, gear, weather, human data, final plan, scenario."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Difficulty = Literal["easy", "moderate", "hard"]
DIFFICULTY_ORDER: dict[str, int] = {"easy": 0, "moderate": 1, "hard": 2}

HumanId = Literal["human_a", "human_b"]
ActorId = Literal["human_a", "agent_a", "agent_b", "human_b"]


class StrictModel(BaseModel):
    """Base for all cross-module data: unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# Constraints and plan

class Constraints(StrictModel):
    available_dates: list[date]
    max_difficulty: Difficulty
    max_elevation_gain_m: int = Field(ge=0)
    max_duration_h: float = Field(gt=0)
    avoid_exposure: bool
    notes: list[str] = Field(default_factory=list)


class ParticipantGear(StrictModel):
    participant: str
    has: list[str] = Field(default_factory=list)
    rented: list[str] = Field(default_factory=list)
    must_bring: list[str] = Field(default_factory=list)


class FinalPlan(StrictModel):
    date: date
    trail_id: str
    trail_name: str
    booking_id: str
    weather_summary: str
    participants: list[ParticipantGear] = Field(min_length=1)
    safety_notes: list[str] = Field(default_factory=list)


# World data

class Trail(StrictModel):
    id: str
    name: str
    area: str
    start_point: str
    lat: float
    lon: float
    distance_km: float = Field(gt=0)
    duration_h: float = Field(gt=0)
    elevation_gain_m: int = Field(ge=0)
    max_altitude_m: int = Field(ge=0)
    difficulty: Difficulty
    exposure: bool
    loop: bool
    season_notes: str
    source: str


class DailyForecast(StrictModel):
    """One day of a simplified Open-Meteo daily response (units: degC, mm, %, cm, km/h, s)."""

    date: date
    weather_code: int
    temperature_2m_max: float
    temperature_2m_min: float
    precipitation_sum: float = Field(ge=0)
    precipitation_probability_max: int = Field(ge=0, le=100)
    snowfall_sum: float = Field(ge=0)
    wind_speed_10m_max: float = Field(ge=0)
    daylight_duration: float = Field(ge=0)


class GearItem(StrictModel):
    id: str
    name: str
    category: str
    rentable: bool


class GearCondition(StrictModel):
    """All set fields must hold for the rule to apply; an empty condition always applies."""

    seasons: list[str] | None = None
    min_difficulty: Difficulty | None = None
    min_elevation_gain_m: int | None = None
    min_max_altitude_m: int | None = None


class GearRule(StrictModel):
    id: str
    when: GearCondition
    items: list[str] = Field(min_length=1)
    reason: str


class GearCatalog(StrictModel):
    items: list[GearItem]
    rules: list[GearRule]

    @model_validator(mode="after")
    def _rules_reference_known_items(self) -> GearCatalog:
        known = {i.id for i in self.items}
        unknown = sorted({i for r in self.rules for i in r.items} - known)
        if unknown:
            raise ValueError(f"gear rules reference unknown items: {unknown}")
        return self


# Private human data

class HealthLimits(StrictModel):
    max_difficulty: Difficulty
    max_elevation_gain_m: int = Field(ge=0)
    max_duration_h: float = Field(gt=0)
    avoid_exposure: bool = False
    notes: list[str] = Field(default_factory=list)


class Profile(StrictModel):
    name: str
    fitness: Literal["low", "moderate", "good"]
    health_limits: HealthLimits
    notes: str = ""


class Calendar(StrictModel):
    available_dates: list[date]
    notes: str = ""


class Inventory(StrictModel):
    items: list[str]


# Scenario

class HiddenFact(StrictModel):
    """Known only to the simulated human; `implies` lists Constraints fields it overrides."""

    text: str
    implies: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _implies_known_fields(self) -> HiddenFact:
        unknown = sorted(set(self.implies) - set(Constraints.model_fields))
        if unknown:
            raise ValueError(f"hidden fact implies unknown constraint fields: {unknown}")
        return self


class PersonaDecisions(StrictModel):
    consent_to_share: bool | None = None
    rent_missing_gear: bool | None = None
    accept_final_plan: bool = True


class Persona(StrictModel):
    style: str
    known_facts: list[str] = Field(default_factory=list)
    hidden_facts: list[HiddenFact] = Field(default_factory=list)
    decisions: PersonaDecisions


class HumanSpec(StrictModel):
    name: str
    persona: Persona
    profile: Profile
    calendar: Calendar
    inventory: Inventory


class InitialMessage(StrictModel):
    sender: HumanId
    recipient: Literal["agent_a", "agent_b"]
    text: str


class DateScoring(StrictModel):
    """Weights of the per-day weather penalty (lower penalty = better day)."""

    precipitation_mm: float = Field(ge=0)
    precipitation_probability: float = Field(ge=0)
    wind_threshold_kmh: float = Field(ge=0)
    wind_per_kmh_over: float = Field(ge=0)
    snowfall_cm: float = Field(ge=0)
    weather_code_class: dict[str, float]


class PlanningParams(StrictModel):
    seasons: dict[str, list[int]]
    max_altitude_by_season_m: dict[str, int] = Field(default_factory=dict)
    date_scoring: DateScoring

    @model_validator(mode="after")
    def _months_unique(self) -> PlanningParams:
        months = [m for ms in self.seasons.values() for m in ms]
        if sorted(months) != list(range(1, 13)):
            raise ValueError("seasons must cover each month 1..12 exactly once")
        return self


class ExpectedOutcome(StrictModel):
    date: date
    trail_id: str
    booking_confirmed: bool
    rented_items: list[str]
    must_bring: dict[HumanId, list[str]]


class Scenario(StrictModel):
    name: str
    today: date
    window_start: date
    window_end: date
    trip_type: str
    planning: PlanningParams
    humans: dict[HumanId, HumanSpec]
    initial_messages: list[InitialMessage] = Field(min_length=1)
    expected_outcome: ExpectedOutcome


def load_scenario(path: Path | str) -> Scenario:
    """Load and validate a scenario YAML; raises FileNotFoundError / ValidationError."""
    with Path(path).open(encoding="utf-8") as f:
        return Scenario.model_validate(yaml.safe_load(f))
