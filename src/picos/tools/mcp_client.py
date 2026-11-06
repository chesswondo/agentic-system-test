"""Toolset wiring: given an AgentSpec, return exactly the tools that agent may use.

MCP servers are launched over stdio as `python -m picos.mcp_servers.<name>`, one persistent
session per (agent, server) for the whole run (the adapter's default of one process per
call is slow and would lose in-process state). Every MCP tool is wrapped by the
interceptors [ObservationInterceptor, ResilienceInterceptor] (outermost first).
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools

from picos.agents.spec import AgentSpec
from picos.config import FaultsConfig
from picos.domain import Scenario
from picos.mcp_servers.common import ENV_DATA_DIR, ENV_SCENARIO, ENV_STATE_DIR
from picos.runtime.recorder import Recorder
from picos.tools.comms import Outbox, comms_functions
from picos.tools.decision_tools import decision_functions
from picos.tools.faults import ResilienceInterceptor
from picos.tools.local import LocalToolFactory
from picos.tools.observations import ObservationInterceptor, Observations
from picos.tools.private_env import private_env_functions

SERVER_MODULES: dict[str, str] = {
    "weather": "picos.mcp_servers.weather",
    "trails": "picos.mcp_servers.trails",
    "gear": "picos.mcp_servers.gear",
}


@dataclass(frozen=True)
class EnvContext:
    """Run-scoped environment settings shared by both agents' server processes."""

    run_seed: int
    scenario_path: Path
    data_dir: Path
    state_dir: Path  # idempotency ledgers live here (e.g. runs/<run_id>/env)


def server_connection(server: str, env: EnvContext) -> StdioConnection:
    if server not in SERVER_MODULES:
        raise ValueError(f"unknown MCP server {server!r}; known: {sorted(SERVER_MODULES)}")
    return StdioConnection(
        transport="stdio",
        command=sys.executable,
        args=["-m", SERVER_MODULES[server]],
        env={
            ENV_DATA_DIR: str(env.data_dir.resolve()),
            ENV_SCENARIO: str(env.scenario_path.resolve()),
            ENV_STATE_DIR: str(env.state_dir.resolve()),
        },
    )


def build_local_tools(
    spec: AgentSpec,
    scenario: Scenario,
    observations: Observations,
    outbox: Outbox,
    recorder: Recorder,
) -> list[BaseTool]:
    if outbox.owner != spec.name:
        raise ValueError(f"outbox of {outbox.owner!r} given to {spec.name!r}")
    human = scenario.humans[spec.human]
    available = {
        fn.__name__: fn
        for fn in private_env_functions(human)
        + decision_functions(human, scenario.planning, observations)
        + comms_functions(spec.name, spec.human, outbox)
    }
    unknown = sorted(set(spec.local_tools) - set(available))
    if unknown:
        raise ValueError(f"{spec.name}: unknown local tools {unknown}")
    factory = LocalToolFactory(spec.name, recorder)
    return [factory.make(available[name]) for name in spec.local_tools]


@asynccontextmanager
async def open_toolset(
    spec: AgentSpec,
    scenario: Scenario,
    faults: FaultsConfig,
    env: EnvContext,
    outbox: Outbox,
    recorder: Recorder,
) -> AsyncIterator[list[BaseTool]]:
    """Yield the agent's tools (local + allow-listed MCP), sorted by name; sessions stay open
    until the context exits. Comms tools queue messages in `outbox`."""
    observations = Observations()
    tools = build_local_tools(spec, scenario, observations, outbox, recorder)
    interceptors = [
        ObservationInterceptor(observations),
        ResilienceInterceptor(faults, env.run_seed, spec.name, recorder),
    ]
    client = MultiServerMCPClient(
        {server: server_connection(server, env) for server in spec.mcp_servers}
    )
    try:
        async with AsyncExitStack() as stack:
            for server, allowed in sorted(spec.mcp_servers.items()):
                session = await stack.enter_async_context(client.session(server))
                served = await load_mcp_tools(
                    session, tool_interceptors=interceptors, server_name=server
                )
                by_name = {t.name: t for t in served}
                missing = sorted(set(allowed) - set(by_name))
                if missing:
                    raise ValueError(f"{spec.name}: server {server!r} has no tools {missing}")
                tools.extend(by_name[name] for name in allowed)
            names = [t.name for t in tools]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(f"{spec.name}: duplicate tool names {duplicates}")
            yield sorted(tools, key=lambda t: t.name)
    except BaseExceptionGroup as group:
        # The stdio sessions run in anyio task groups, which wrap any exception raised inside
        # them (including the caller's own) in a group. Re-raise a lone error as itself.
        leaves = _leaves(group)
        if len(leaves) == 1:
            raise leaves[0] from None
        raise


def _leaves(group: BaseExceptionGroup) -> list[BaseException]:
    out: list[BaseException] = []
    for exc in group.exceptions:
        out.extend(_leaves(exc) if isinstance(exc, BaseExceptionGroup) else [exc])
    return out
