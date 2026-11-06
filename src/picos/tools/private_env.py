"""Private environment tools: read access to one human's data only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from picos.domain import Calendar, HumanSpec, Inventory, Profile


def private_env_functions(human: HumanSpec) -> list[Callable[..., Any]]:
    def get_user_profile() -> Profile:
        """Your human's profile: fitness and structured health limits."""
        return human.profile

    def get_calendar() -> Calendar:
        """Dates your human is available for the trip."""
        return human.calendar

    def get_inventory() -> Inventory:
        """Gear item ids your human already owns."""
        return human.inventory

    return [get_user_profile, get_calendar, get_inventory]
