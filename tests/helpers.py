"""Shared test helpers: project paths, tool wiring and invocation, result accessors, and a
strict validator for the Mermaid subset the runtime emits."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from picos.agents.spec import AgentSpec
from picos.config import FaultRates, FaultsConfig, load_config
from picos.domain import Scenario
from picos.runtime.recorder import ListRecorder
from picos.tools.comms import Outbox
from picos.tools.mcp_client import EnvContext, open_toolset

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "default.yaml"


def make_faults(
    p_error: float = 0.0, p_timeout: float = 0.0, tools: dict[str, FaultRates] | None = None
) -> FaultsConfig:
    """App fault config with the given rates and zero backoff (fast, still exercises retries)."""
    base = load_config(ROOT / "config" / "app.yaml").faults
    return base.model_copy(
        update={
            "default": FaultRates(p_error=p_error, p_timeout=p_timeout),
            "tools": tools or {},
            "retry": base.retry.model_copy(update={"base_delay_s": 0.0, "max_delay_s": 0.0}),
        }
    )


@asynccontextmanager
async def toolset(
    spec: AgentSpec,
    scenario: Scenario,
    env: EnvContext,
    faults: FaultsConfig | None = None,
    recorder: ListRecorder | None = None,
    outbox: Outbox | None = None,
) -> AsyncIterator[dict[str, BaseTool]]:
    async with open_toolset(
        spec,
        scenario,
        faults or make_faults(),
        env,
        outbox or Outbox(spec.name, "test-conv"),
        recorder or ListRecorder(),
    ) as tools:
        yield {t.name: t for t in tools}


_call_counter = 0


async def call(tool: BaseTool, **args: Any) -> ToolMessage:
    """Invoke a tool the way LangGraph's ToolNode does (a ToolCall in, a ToolMessage out)."""
    global _call_counter
    _call_counter += 1
    msg = await tool.ainvoke(
        {"name": tool.name, "args": args, "id": f"t{_call_counter}", "type": "tool_call"}
    )
    assert isinstance(msg, ToolMessage)
    return msg


def structured(msg: ToolMessage) -> dict[str, Any]:
    """Structured payload of a successful MCP tool result."""
    assert msg.status == "success", msg.content
    assert msg.artifact is not None
    return msg.artifact["structured_content"]


def text(msg: ToolMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return " ".join(b["text"] for b in msg.content if isinstance(b, dict) and "text" in b)


# Grammar of what we emit (a subset of Mermaid's sequenceDiagram syntax).
_LABEL = r"([^;#:%<>{}\[\]\"`|\\\n]+)"
_PARTICIPANT = re.compile(r"^ {4}participant (\w+) as " + _LABEL + "$")
_ARROW = re.compile(r"^ {4}(\w+)->>(\w+): " + _LABEL + "$")
_NOTE = re.compile(r"^ {4}Note over (\w+),(\w+): " + _LABEL + "$")


def assert_valid_mermaid(text: str) -> None:
    """Strict check: header first, participants declared before use, every line matches one
    statement form, labels contain no Mermaid syntax characters."""
    lines = text.rstrip("\n").split("\n")
    assert lines[0] == "sequenceDiagram"
    declared: list[str] = []
    for line in lines[1:]:
        if m := _PARTICIPANT.match(line):
            assert m.group(1) not in declared, f"duplicate participant: {line}"
            declared.append(m.group(1))
        elif m := _ARROW.match(line) or _NOTE.match(line):
            assert m.group(1) in declared and m.group(2) in declared, f"undeclared: {line}"
            assert m.group(3).strip(), f"empty label: {line}"
        else:
            raise AssertionError(f"not valid Mermaid: {line!r}")
