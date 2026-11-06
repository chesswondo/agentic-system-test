"""Prompt wiring: per-human persona templates, scenario context, prompt/toolset agreement."""

from __future__ import annotations

import re

import pytest
import yaml
from pydantic import ValidationError

from picos.agents.graph import PLAN_DRAFT_TOOL, render_system_prompt
from picos.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from picos.humans.llm import render_persona
from picos.runtime.simulation import agent_context
from tests.helpers import ROOT

CFG = load_config(ROOT / DEFAULT_CONFIG_PATH)


def _tools(spec) -> set[str]:
    mcp = {t for allowed in spec.mcp_servers.values() for t in allowed}
    return {*spec.local_tools, *mcp, PLAN_DRAFT_TOOL}


def test_human_prompts_must_cover_every_human() -> None:
    raw = yaml.safe_load((ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    del raw["humans"]["prompts"]["human_b"]
    with pytest.raises(ValidationError, match="human_b"):
        AppConfig.model_validate(raw)


@pytest.mark.parametrize("hid", ["human_a", "human_b"])
def test_human_template_renders_persona(scenario, hid) -> None:
    template = (ROOT / CFG.humans.prompts[hid]).read_text(encoding="utf-8")
    spec = scenario.humans[hid]
    rendered = render_persona(template, spec)  # KeyError/ValueError on a bad placeholder
    assert spec.name in rendered and spec.persona.style in rendered
    for fact in [*spec.persona.known_facts, *(f.text for f in spec.persona.hidden_facts)]:
        assert fact in rendered


def test_system_prompt_carries_scenario_context(scenario, specs) -> None:
    spec = specs["agent_a"]
    prompt = render_system_prompt(spec, "Do things.", {}, agent_context(scenario, spec))
    assert "Today: 2026-11-16 (Monday)" in prompt
    assert "Planning window: 2026-11-17 to 2026-11-30" in prompt
    assert "Your human: Alice (human_a)" in prompt


def test_agent_prompts_mention_only_own_tools(specs) -> None:
    """A prompt must never tell an agent to call a tool it does not have (e.g. B booking)."""
    all_tools = set().union(*(_tools(s) for s in specs.values()))
    for spec in specs.values():
        text = spec.instructions_path.read_text(encoding="utf-8")
        mentioned = set(re.findall(r"`([a-z_]+)`", text)) & all_tools
        assert mentioned, f"{spec.name}: prompt names no tools"
        assert mentioned <= _tools(spec), f"{spec.name}: foreign tools {mentioned - _tools(spec)}"


def test_system_prompt_names_the_other_participant(scenario, specs) -> None:
    prompt = agent_context(scenario, specs["agent_b"])
    assert "Your human: Bob (human_b)" in prompt
    assert "Other participant: Alice (human_a), represented by the other assistant (agent_a)" in (
        prompt
    )


def test_prompt_files_are_not_tied_to_the_scenario_cast(scenario, specs) -> None:
    """Names and personal facts come from the scenario at run time, never from prompt text."""
    names = {h.name for h in scenario.humans.values()}
    paths = [s.instructions_path for s in specs.values()]
    paths += [ROOT / p for p in CFG.humans.prompts.values()]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        found = {n for n in names if re.search(rf"\b{n}\b", text)}
        assert not found, f"{path.name} names {sorted(found)}"
