"""Scripted chat model for offline tests and runs: replays AIMessages in order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, PrivateAttr, field_validator


class ScriptExhaustedError(RuntimeError):
    """The scripted model was called more often than its script allows (a test bug)."""


class ScriptedChatModel(BaseChatModel):
    """Returns `script[i]` on the i-th call, across all turns of one actor.

    An entry may also be an exception instance, which is raised instead (to simulate provider
    errors). `bind_tools` keeps the bound tool names so tests can assert on them; the script
    is not checked against them, because calling an unknown tool is a behaviour to exercise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Any + explicit check: pydantic would otherwise try to coerce exceptions into AIMessage.
    script: list[Any]
    model: str = "fake-scripted"
    _cursor: int = PrivateAttr(default=0)
    _bound_tools: list[str] = PrivateAttr(default_factory=list)
    _seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @field_validator("script")
    @classmethod
    def _entries(cls, script: list[Any]) -> list[Any]:
        for i, entry in enumerate(script):
            if not isinstance(entry, AIMessage | Exception):
                raise ValueError(f"script[{i}] must be an AIMessage or an exception instance")
        return script

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> int:
        return self._cursor

    @property
    def bound_tools(self) -> list[str]:
        return list(self._bound_tools)

    @property
    def seen(self) -> list[list[BaseMessage]]:
        """The message lists the model received, one per call."""
        return list(self._seen)

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self._bound_tools = sorted(getattr(t, "name", str(t)) for t in tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._cursor >= len(self.script):
            raise ScriptExhaustedError(
                f"script has {len(self.script)} responses; call #{self._cursor + 1} has none"
            )
        entry = self.script[self._cursor]
        self._cursor += 1
        self._seen.append(list(messages))
        if isinstance(entry, Exception):
            raise entry
        return ChatResult(generations=[ChatGeneration(message=entry.model_copy(deep=True))])
