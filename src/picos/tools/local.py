"""Helper to turn plain functions into recorded LangChain tools with model-facing errors."""

from __future__ import annotations

import functools
import inspect
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ValidationError

from picos.runtime.recorder import Recorder


def format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc'])) or 'value'}: {e['msg']}" for e in exc.errors()]
        return "invalid input: " + "; ".join(parts)
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc}"
    return str(exc)


def to_json(value: Any) -> str:
    """Canonical JSON for tool results (sorted keys, so results are byte-stable)."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class LocalToolFactory:
    """Builds tools for one actor; every call and result goes to the recorder.

    `ValueError` (incl. pydantic `ValidationError` and JSON decode errors) raised by a tool
    function becomes an error tool result the model can react to; anything else is a bug and
    propagates.
    """

    def __init__(self, actor: str, recorder: Recorder) -> None:
        self.actor = actor
        self.recorder = recorder
        self._counts: defaultdict[str, int] = defaultdict(int)

    def make(self, fn: Callable[..., Any]) -> BaseTool:
        name = fn.__name__

        # A coroutine, not a sync function: LangChain would run sync tools in executor threads,
        # making the order of recorded events depend on thread scheduling.
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            self._counts[name] += 1
            call_id = f"{name}#{self._counts[name]}"
            self.recorder.record(self.actor, "tool_call", tool=name, call_id=call_id, args=kwargs)
            try:
                result = to_json(fn(*args, **kwargs))
            except ValueError as exc:
                message = format_error(exc)
                self.recorder.record(
                    self.actor, "tool_result", tool=name, call_id=call_id, ok=False, error=message
                )
                raise ToolException(message) from exc
            self.recorder.record(self.actor, "tool_result", tool=name, call_id=call_id, ok=True)
            return result

        return StructuredTool.from_function(
            coroutine=wrapper,
            name=name,
            description=inspect.cleandoc(fn.__doc__ or name),
            handle_tool_error=True,
        )
