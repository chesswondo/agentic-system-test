"""Application config: YAML -> pydantic models, plus .env loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from picos.domain import HumanId

DEFAULT_CONFIG_PATH = Path("config/app.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Temperatures(_Strict):
    # None: do not send a temperature; the provider default applies (Gemini 3.x guidance).
    agents: float | None = Field(default=None, ge=0, le=2)
    humans: float | None = Field(default=None, ge=0, le=2)


class ModelIds(_Strict):
    agents: str
    humans: str


class RetryConfig(_Strict):
    max_attempts: int = Field(ge=1)
    base_delay_s: float = Field(ge=0)
    max_delay_s: float = Field(ge=0)


class LLMConfig(_Strict):
    provider: Literal["google_genai", "fake"]
    models: ModelIds
    temperature: Temperatures
    retry: RetryConfig  # recorded retries of retryable model errors (429, 5xx, timeouts)


class RuntimeConfig(_Strict):
    max_ticks: int = Field(gt=0)
    max_steps_per_turn: int = Field(gt=0)
    max_negotiation_rounds: int = Field(gt=0)
    runs_dir: Path


class FaultRates(_Strict):
    p_error: float = Field(ge=0, le=1)
    p_timeout: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _total_at_most_one(self) -> FaultRates:
        if self.p_error + self.p_timeout > 1:
            raise ValueError("p_error + p_timeout must not exceed 1")
        return self


class FaultsConfig(_Strict):
    seed_salt: int = 0
    retry: RetryConfig
    tool_timeout_s: float = Field(gt=0)
    default: FaultRates
    tools: dict[str, FaultRates] = Field(default_factory=dict)

    def rates_for(self, tool: str) -> FaultRates:
        return self.tools.get(tool, self.default)


class ObservabilityConfig(_Strict):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    langsmith: Literal["auto", "off"] = "auto"


class HumansConfig(_Strict):
    # `llm`: persona-driven LLM replies; `scripted`: rule table (offline). With
    # llm.provider `fake`, humans are always scripted.
    provider: Literal["llm", "scripted"]
    # Persona template per human for the llm provider, relative to project root.
    prompts: dict[HumanId, Path]

    @model_validator(mode="after")
    def _all_humans(self) -> HumansConfig:
        missing = sorted(set(get_args(HumanId)) - set(self.prompts))
        if missing:
            raise ValueError(f"humans.prompts is missing {missing}")
        return self


class Price(_Strict):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class AppConfig(_Strict):
    llm: LLMConfig
    runtime: RuntimeConfig
    faults: FaultsConfig
    humans: HumansConfig
    observability: ObservabilityConfig
    prices: dict[str, Price]


class ConfigError(RuntimeError):
    """A configuration problem the user must fix (missing secret, inconsistent settings)."""


def load_env() -> None:
    """Load variables from `.env` (if present) into the process environment."""
    load_dotenv()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate the app config; raises FileNotFoundError / ValidationError."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
