"""LLM factory: provider selection from config, fail-fast on a missing key."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from picos.agents.fake import ScriptedChatModel, ScriptExhaustedError
from picos.agents.graph import AgentOptions
from picos.agents.llm import API_KEY_ENV, make_llm
from picos.config import ConfigError, load_config
from tests.helpers import ROOT


@pytest.fixture
def cfg():
    return load_config(ROOT / "config" / "app.yaml")


def test_google_genai_without_key_fails_fast(cfg, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm_cfg = cfg.llm.model_copy(update={"provider": "google_genai"})
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY is not set"):
        make_llm(llm_cfg, "agents")


def test_google_genai_uses_config_model_and_no_sdk_retries(cfg, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "dummy-key-for-construction-only")
    llm_cfg = cfg.llm.model_copy(update={"provider": "google_genai"})
    agents, humans = make_llm(llm_cfg, "agents"), make_llm(llm_cfg, "humans")
    assert agents.model == cfg.llm.models.agents
    assert humans.model == cfg.llm.models.humans
    assert agents.max_retries == 1  # retries are done (and recorded) by the agent graph
    assert agents.temperature is None  # config null: not sent (Gemini 3.x guidance)


def test_google_genai_passes_explicit_temperature(cfg, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "dummy-key-for-construction-only")
    temps = cfg.llm.temperature.model_copy(update={"agents": 0.3})
    llm_cfg = cfg.llm.model_copy(update={"provider": "google_genai", "temperature": temps})
    assert make_llm(llm_cfg, "agents").temperature == 0.3


def test_fake_needs_a_script_and_reports_exhaustion(cfg):
    assert cfg.llm.provider == "fake"
    with pytest.raises(ConfigError, match="needs a script"):
        make_llm(cfg.llm, "agents")
    llm = make_llm(cfg.llm, "agents", fake_script=[AIMessage(content="hi")])
    assert isinstance(llm, ScriptedChatModel)
    assert llm.invoke("x").content == "hi"
    with pytest.raises(ScriptExhaustedError):
        llm.invoke("x")


def test_spec_step_budget_overrides_runtime_default(cfg, specs):
    assert AgentOptions.from_config(cfg, specs["agent_a"]).max_steps_per_turn == (
        cfg.runtime.max_steps_per_turn
    )
    custom = specs["agent_a"].model_copy(update={"max_steps_per_turn": 5})
    assert AgentOptions.from_config(cfg, custom).max_steps_per_turn == 5
