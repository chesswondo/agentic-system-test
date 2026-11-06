"""gear-mcp: required gear per trail and season, and idempotent equipment rental."""

from __future__ import annotations

from functools import cache
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from picos.domain import GearCatalog, Trail
from picos.mcp_servers.common import (
    SERVER_LOG_LEVEL,
    IdempotentLedger,
    data_dir,
    parse_date,
    scenario,
    state_dir,
)
from picos.tools import decision
from picos.tools.datasets import load_gear_catalog, load_trails

mcp = FastMCP("gear", log_level=SERVER_LOG_LEVEL)


@cache
def _catalog() -> GearCatalog:
    return load_gear_catalog(data_dir())


@cache
def _trails() -> dict[str, Trail]:
    return {t.id: t for t in load_trails(data_dir())}


def ledger() -> IdempotentLedger:
    return IdempotentLedger(state_dir() / "rentals.json", "RNT")


@mcp.tool()
def required_gear(trail_id: str, date: str) -> dict[str, Any]:
    """Gear every participant needs for this trail on this date (season-aware)."""
    trails = _trails()
    if trail_id not in trails:
        raise ToolError(f"unknown trail_id {trail_id!r}")
    trail = trails[trail_id]
    season = decision.season_of(parse_date(date), scenario().planning.seasons)
    items = decision.required_gear(trail, season, _catalog())
    return {"trail_id": trail_id, "date": date, "season": season, "items": items}


@mcp.tool()
def rent_equipment(items: list[str], date: str, idempotency_key: str) -> dict[str, Any]:
    """Rent gear items (catalog ids) for one day. Idempotent: repeating a call with the same
    idempotency_key returns the original confirmation instead of a second rental."""
    if not items:
        raise ToolError("items must not be empty")
    catalog = {i.id: i for i in _catalog().items}
    unknown = sorted(set(items) - set(catalog))
    if unknown:
        raise ToolError(f"unknown items {unknown}; valid: {sorted(catalog)}")
    not_rentable = sorted(i for i in set(items) if not catalog[i].rentable)
    if not_rentable:
        raise ToolError(f"items not available for rent: {not_rentable}")
    request = {"items": sorted(set(items)), "date": parse_date(date).isoformat()}
    return ledger().execute(
        idempotency_key,
        request,
        lambda rental_id: {"rental_id": rental_id, "status": "confirmed", **request},
    )


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
