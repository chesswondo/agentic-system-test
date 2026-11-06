"""LLM factory: `google_genai` (Gemini) or `fake` (scripted), configured from YAML."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from picos.agents.fake import ScriptedChatModel
from picos.config import ConfigError, LLMConfig, RetryConfig
from picos.runtime.recorder import Recorder

Role = Literal["agents", "humans"]
API_KEY_ENV = "GOOGLE_API_KEY"


def make_llm(
    cfg: LLMConfig,
    role: Role,
    *,
    fake_script: Sequence[AIMessage | Exception] | None = None,
) -> BaseChatModel:
    """Build the chat model for one actor role.

    `fake` needs `fake_script` (one scripted model per actor). `google_genai` needs
    GOOGLE_API_KEY in the environment and fails fast with ConfigError without it.
    """
    if cfg.provider == "fake":
        if fake_script is None:
            raise ConfigError("llm.provider 'fake' needs a script of responses for each actor")
        return ScriptedChatModel(script=list(fake_script))

    if not os.environ.get(API_KEY_ENV):
        raise ConfigError(
            f"llm.provider is 'google_genai' but {API_KEY_ENV} is not set. Put it in .env "
            "(see .env.example) or the environment, or run offline with --llm fake."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": getattr(cfg.models, role),
        "google_api_key": os.environ[API_KEY_ENV],
        # 1 = a single attempt: retries happen in the agent graph, where they are recorded.
        "max_retries": 1,
    }
    temperature = getattr(cfg.temperature, role)
    if temperature is not None:  # unset: the library sends none for Gemini 3+
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)


def is_retryable(exc: BaseException) -> bool:
    """Provider-neutral: langchain-core classifies 429/5xx/timeouts as retryable ModelErrors."""
    return isinstance(exc, ModelError) and exc.is_retryable


async def ainvoke_with_retry(
    runnable: Runnable[Any, Any],
    messages: list[BaseMessage],
    *,
    retry: RetryConfig,
    recorder: Recorder,
    actor: str,
    model: str,
) -> Any:
    """Invoke a model; retryable provider errors are retried with backoff and each retry is
    recorded as a `retry` event (the SDK's own retries would be invisible in the trajectory).
    Other errors and exhausted retries propagate."""

    def on_retry(rs: RetryCallState) -> None:
        exc = rs.outcome.exception() if rs.outcome else None
        recorder.record(
            actor,
            "retry",
            target="llm",
            model=model,
            attempt=rs.attempt_number,
            error=f"{type(exc).__name__}: {exc}",
            next_delay_s=rs.next_action.sleep if rs.next_action else None,
        )

    async for attempt in AsyncRetrying(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(retry.max_attempts),
        wait=wait_exponential(multiplier=retry.base_delay_s, max=retry.max_delay_s),
        before_sleep=on_retry,
        reraise=True,
    ):
        with attempt:
            return await runnable.ainvoke(messages)
    raise AssertionError("unreachable: tenacity either returns or raises")


def model_name(llm: BaseChatModel) -> str:
    return str(getattr(llm, "model", None) or llm._llm_type)
