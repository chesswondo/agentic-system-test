"""weather-mcp: daily forecasts per park area (mode `snapshot`: data/weather_snapshot.json)."""

from __future__ import annotations

from functools import cache
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from picos.domain import DailyForecast
from picos.mcp_servers.common import SERVER_LOG_LEVEL, data_dir, parse_date
from picos.tools.datasets import load_weather_snapshot

mcp = FastMCP("weather", log_level=SERVER_LOG_LEVEL)


@cache
def _forecasts() -> dict[str, list[DailyForecast]]:
    return load_weather_snapshot(data_dir())


@mcp.tool()
def list_locations() -> dict[str, Any]:
    """List the location ids (park areas) that have forecasts."""
    return {"location_ids": sorted(_forecasts())}


@mcp.tool()
def get_forecast(location_id: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Daily forecast for one park area between two ISO dates (inclusive).

    Units: temperature degC, precipitation mm, probability %, snowfall cm, wind km/h,
    daylight seconds.
    """
    forecasts = _forecasts()
    if location_id not in forecasts:
        raise ToolError(f"unknown location_id {location_id!r}; valid: {sorted(forecasts)}")
    start, end = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
    if start > end:
        raise ToolError("start_date must not be after end_date")
    days = [d.model_dump(mode="json") for d in forecasts[location_id] if start <= d.date <= end]
    return {"location_id": location_id, "days": days}


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
