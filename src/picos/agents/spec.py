"""AgentSpec: everything that distinguishes Agent A from Agent B, loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from picos.domain import HumanId, StrictModel


class AgentSpec(StrictModel):
    name: Literal["agent_a", "agent_b"]
    role: str
    human: HumanId  # whose private environment this agent is bound to
    instructions_path: Path
    local_tools: list[str]
    mcp_servers: dict[str, list[str]] = Field(default_factory=dict)  # server -> allow-list
    max_steps_per_turn: int | None = Field(default=None, gt=0)  # None: runtime default
    # MCP tools whose successful results from earlier turns are replaced by a short stub in
    # what the LLM sees (the data stays in the observation cache for the decision tools).
    compact_tool_results: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compact_only_own_mcp_tools(self) -> AgentSpec:
        own = {t for tools in self.mcp_servers.values() for t in tools}
        unknown = sorted(set(self.compact_tool_results) - own)
        if unknown:
            raise ValueError(f"compact_tool_results names tools not in mcp_servers: {unknown}")
        return self


def load_agent_spec(path: Path | str, base_dir: Path | str = ".") -> AgentSpec:
    """Load a spec; a relative `instructions_path` is resolved against `base_dir` (the project
    root, i.e. the working directory of the CLI)."""
    with Path(path).open(encoding="utf-8") as f:
        spec = AgentSpec.model_validate(yaml.safe_load(f))
    if not spec.instructions_path.is_absolute():
        spec = spec.model_copy(
            update={"instructions_path": Path(base_dir) / spec.instructions_path}
        )
    return spec
