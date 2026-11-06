"""Load scripted agent responses (for `llm.provider: fake`) from YAML.

Format: {agent_id: [step, ...]}; a step is {text: str, tool_calls: [{name, args}]}. One step
is one LLM response, consumed in order across all of that agent's turns. Tool call ids are
generated deterministically (`<agent>-c<n>`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage
from pydantic import Field

from picos.domain import StrictModel
from picos.protocol import AgentId


class FakeToolCall(StrictModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class FakeStep(StrictModel):
    text: str = ""
    tool_calls: list[FakeToolCall] = Field(default_factory=list)


class FakeScripts(StrictModel):
    agent_a: list[FakeStep] = Field(default_factory=list)
    agent_b: list[FakeStep] = Field(default_factory=list)

    def messages(self, agent: AgentId) -> list[AIMessage]:
        out: list[AIMessage] = []
        n = 0
        for step in getattr(self, agent):
            calls = []
            for call in step.tool_calls:
                n += 1
                calls.append(
                    {
                        "name": call.name,
                        "args": call.args,
                        "id": f"{agent}-c{n}",
                        "type": "tool_call",
                    }
                )
            out.append(AIMessage(content=step.text, tool_calls=calls))
        return out


def load_fake_scripts(path: Path | str) -> FakeScripts:
    with Path(path).open(encoding="utf-8") as f:
        return FakeScripts.model_validate(yaml.safe_load(f))


def default_fake_script_path(scenario_path: Path) -> Path:
    """`scenarios/default.yaml` -> `scenarios/default.fake.yaml`."""
    return scenario_path.with_name(f"{scenario_path.stem}.fake.yaml")
