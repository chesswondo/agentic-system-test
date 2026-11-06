"""trails-mcp: trail catalog of Picos de Europa and idempotent trip booking."""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from picos.domain import DIFFICULTY_ORDER, Trail
from picos.mcp_servers.common import (
    SERVER_LOG_LEVEL,
    IdempotentLedger,
    data_dir,
    parse_date,
    state_dir,
)
from picos.tools.datasets import load_trails

mcp = FastMCP("trails", log_level=SERVER_LOG_LEVEL)


@cache
def _trails() -> dict[str, Trail]:
    return {t.id: t for t in load_trails(data_dir())}


def ledger() -> IdempotentLedger:
    return IdempotentLedger(state_dir() / "bookings.json", "BKG")


def _trail(trail_id: str) -> Trail:
    trails = _trails()
    if trail_id not in trails:
        raise ToolError(f"unknown trail_id {trail_id!r}; use search_trails to list trails")
    return trails[trail_id]


@mcp.tool()
def search_trails(
    area: str = "", max_difficulty: Literal["easy", "moderate", "hard"] = "hard"
) -> dict[str, Any]:
    """List trail records, optionally filtered by area id and maximum difficulty.

    This is a catalog lookup, not a recommendation: rank trails with `rank_trails`.
    """
    trails = sorted(_trails().values(), key=lambda t: t.id)
    if area:
        areas = sorted({t.area for t in trails})
        if area not in areas:
            raise ToolError(f"unknown area {area!r}; valid: {areas}")
        trails = [t for t in trails if t.area == area]
    limit = DIFFICULTY_ORDER[max_difficulty]
    trails = [t for t in trails if DIFFICULTY_ORDER[t.difficulty] <= limit]
    return {"trails": [t.model_dump(mode="json") for t in trails]}


@mcp.tool()
def get_trail(trail_id: str) -> dict[str, Any]:
    """Full record of one trail."""
    return _trail(trail_id).model_dump(mode="json")


@mcp.tool()
def book_trip(trail_id: str, date: str, party_size: int, idempotency_key: str) -> dict[str, Any]:
    """Book a guided-trip slot. Idempotent: repeating a call with the same idempotency_key
    returns the original confirmation instead of creating a second booking."""
    trail = _trail(trail_id)
    day = parse_date(date)
    if party_size < 1:
        raise ToolError("party_size must be at least 1")
    request = {"trail_id": trail.id, "date": day.isoformat(), "party_size": party_size}
    return ledger().execute(
        idempotency_key,
        request,
        lambda booking_id: {"booking_id": booking_id, "status": "confirmed", **request},
    )


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
