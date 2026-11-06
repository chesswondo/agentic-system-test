"""Helpers shared by the MCP servers: launch environment, date parsing, idempotent ledger.

Servers are configured only through environment variables set by the client wiring
(`picos.tools.mcp_client`), so one server process serves exactly one run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError

from picos.domain import Scenario, load_scenario

ENV_DATA_DIR = "PICOS_DATA_DIR"
ENV_SCENARIO = "PICOS_SCENARIO"
ENV_STATE_DIR = "PICOS_STATE_DIR"
# Servers log to stderr, which the client inherits; per-request INFO lines are noise there.
SERVER_LOG_LEVEL = "WARNING"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"MCP server started without required env var {name}")
    return value


def data_dir() -> Path:
    return Path(_required_env(ENV_DATA_DIR))


def state_dir() -> Path:
    return Path(_required_env(ENV_STATE_DIR))


@cache
def scenario() -> Scenario:
    return load_scenario(_required_env(ENV_SCENARIO))


def parse_date(value: str, field: str = "date") -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"{field} must be an ISO date YYYY-MM-DD, got {value!r}") from None


class IdempotentLedger:
    """Run-scoped, file-backed record of side-effecting operations keyed by idempotency key.

    A repeated key with the same request returns the original record; a repeated key with a
    different request is a conflict. The file survives server restarts within a run and is
    an audit artifact (exactly one booking / rental per run). Writes are atomic (tmp + replace).
    Tools in FastMCP that are plain functions run on the server's event loop one at a time,
    so no lock is needed inside one process; each ledger file has a single writing server.
    """

    def __init__(self, path: Path, id_prefix: str) -> None:
        self.path = path
        self.id_prefix = id_prefix

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    def _save(self, entries: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def execute(
        self, key: str, request: dict[str, Any], make_record: Callable[[str], dict[str, Any]]
    ) -> dict[str, Any]:
        if not key.strip():
            raise ToolError("idempotency_key must be a non-empty string")
        entries = self._load()
        if key in entries:
            if entries[key]["request"] != request:
                raise ToolError(
                    f"idempotency_key {key!r} was already used with different arguments "
                    f"{entries[key]['request']}; use a new key for a different request"
                )
            return entries[key]["record"]
        record = make_record(f"{self.id_prefix}-{len(entries) + 1:04d}")
        entries[key] = {"request": request, "record": record}
        self._save(entries)
        return record

    def records(self) -> list[dict[str, Any]]:
        return [e["record"] for e in self._load().values()]
