"""Per-agent cache of environment data the agent has actually received from MCP tools.

Decision tools (`rank_dates`, `rank_trails`) read from here instead of asking the LLM to
copy large JSON blobs into their arguments. Only results that reached the agent are cached:
`ObservationInterceptor` sits outside the resilience layer, so responses lost to an injected
timeout are never observed. Each agent owns its own instance (agent states stay independent).
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from mcp.types import CallToolResult

from picos.domain import DailyForecast, Trail
from picos.tools.faults import Handler


class Observations:
    def __init__(self) -> None:
        self.forecasts: dict[str, dict[str, DailyForecast]] = {}  # location -> iso date -> day
        self.trails: dict[str, Trail] = {}

    def add(self, tool: str, data: dict[str, Any]) -> None:
        if tool == "get_forecast":
            days = self.forecasts.setdefault(data["location_id"], {})
            for raw in data["days"]:
                day = DailyForecast.model_validate(raw)
                days[day.date.isoformat()] = day
        elif tool == "search_trails":
            for raw in data["trails"]:
                trail = Trail.model_validate(raw)
                self.trails[trail.id] = trail
        elif tool == "get_trail":
            trail = Trail.model_validate(data)
            self.trails[trail.id] = trail


class ObservationInterceptor:
    def __init__(self, observations: Observations) -> None:
        self.observations = observations

    async def __call__(self, request: MCPToolCallRequest, handler: Handler) -> MCPToolCallResult:
        result = await handler(request)
        if (
            isinstance(result, CallToolResult)
            and not result.isError
            and result.structuredContent is not None
        ):
            self.observations.add(request.name, result.structuredContent)
        return result
