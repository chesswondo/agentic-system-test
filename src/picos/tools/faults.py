"""Seeded fault injection and transport-level resilience for MCP tools.

`ResilienceInterceptor` is a langchain-mcp-adapters tool interceptor. For every call it:
1. assigns a per-(actor, tool) call index synchronously, before any await;
2. per attempt, draws a fault from an RNG seeded by
   sha256(run_seed, seed_salt, actor, tool, call_index, attempt), so the same seed gives the
   same faults regardless of how parallel calls and retries interleave;
3. retries transient failures with exponential backoff (tenacity), recording every attempt;
4. after the last attempt, returns an error tool result to the agent (layer 2).

Fault semantics:
- `error`: the request never reaches the server (simulated 503).
- `timeout`: the server executes the request but the response is lost. No real sleep is
  done; the retry then re-sends the same idempotency key, which is what idempotency is for.
Errors the server reports itself (`isError`, e.g. a bad trail id) are not retried: they are
returned to the model unchanged so it can correct its arguments.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Literal

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from picos.config import FaultsConfig
from picos.runtime.recorder import Recorder

Fault = Literal["ok", "error", "timeout"]
Handler = Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]


class TransientToolError(Exception):
    """A failure worth retrying: injected 503/timeout, real timeout, or MCP transport error."""


class FaultInjector:
    """Deterministic fault draws for one actor."""

    def __init__(self, config: FaultsConfig, run_seed: int, actor: str) -> None:
        self.config = config
        self.run_seed = run_seed
        self.actor = actor

    def draw(self, tool: str, call_index: int, attempt: int) -> Fault:
        key = f"{self.run_seed}|{self.config.seed_salt}|{self.actor}|{tool}|{call_index}|{attempt}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        roll = random.Random(seed).random()
        rates = self.config.rates_for(tool)
        if roll < rates.p_error:
            return "error"
        if roll < rates.p_error + rates.p_timeout:
            return "timeout"
        return "ok"


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


class ResilienceInterceptor:
    """Fault injection + retries + per-attempt recording for one actor's MCP tools."""

    def __init__(self, config: FaultsConfig, run_seed: int, actor: str, recorder: Recorder) -> None:
        self.config = config
        self.actor = actor
        self.recorder = recorder
        self.injector = FaultInjector(config, run_seed, actor)
        self._call_counts: defaultdict[str, int] = defaultdict(int)

    def _next_index(self, tool: str) -> int:
        self._call_counts[tool] += 1
        return self._call_counts[tool]

    async def __call__(self, request: MCPToolCallRequest, handler: Handler) -> MCPToolCallResult:
        tool = request.name
        call_index = self._next_index(tool)  # before any await: order = call order
        call_id = getattr(request.runtime, "tool_call_id", None) or f"{tool}#{call_index}"
        self.recorder.record(
            self.actor,
            "tool_call",
            tool=tool,
            call_id=call_id,
            args=request.args,
            server=request.server_name,
        )
        retry_cfg = self.config.retry

        def before_sleep(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            self.recorder.record(
                self.actor,
                "retry",
                tool=tool,
                call_id=call_id,
                attempt=state.attempt_number,
                next_attempt=state.attempt_number + 1,
                wait_s=round(state.upcoming_sleep, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

        retrying = AsyncRetrying(
            stop=stop_after_attempt(retry_cfg.max_attempts),
            wait=wait_exponential(multiplier=retry_cfg.base_delay_s, max=retry_cfg.max_delay_s),
            retry=retry_if_exception_type(TransientToolError),
            before_sleep=before_sleep,
            reraise=True,
        )
        attempt = 0
        try:
            async for attempt_ctx in retrying:
                with attempt_ctx:
                    attempt = attempt_ctx.retry_state.attempt_number
                    return await self._attempt(request, handler, call_id, call_index, attempt)
        except TransientToolError as exc:
            return _error_result(
                f"{tool} is temporarily unavailable ({exc}) after {attempt} attempts. "
                "Retry later in this turn, use a fallback, or inform your human."
            )
        raise AssertionError("unreachable: tenacity either returns or raises")

    async def _attempt(
        self,
        request: MCPToolCallRequest,
        handler: Handler,
        call_id: str,
        call_index: int,
        attempt: int,
    ) -> MCPToolCallResult:
        tool = request.name
        fault = self.injector.draw(tool, call_index, attempt)
        started = time.perf_counter()

        def record(ok: bool, error: str | None = None) -> None:
            self.recorder.record(
                self.actor,
                "tool_result",
                tool=tool,
                call_id=call_id,
                ok=ok,
                error=error,
                attempt=attempt,
                fault=fault,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )

        try:
            if fault == "error":
                raise TransientToolError("TransientError: 503 Service Unavailable (injected)")
            try:
                result = await asyncio.wait_for(handler(request), self.config.tool_timeout_s)
            except TimeoutError:
                raise TransientToolError(
                    f"Timeout: no response within {self.config.tool_timeout_s}s"
                ) from None
            except McpError as exc:
                raise TransientToolError(f"TransportError: {exc}") from exc
            if fault == "timeout":
                raise TransientToolError(
                    f"Timeout: no response within {self.config.tool_timeout_s}s "
                    "(injected; the request may have been processed)"
                )
        except TransientToolError as exc:
            record(False, str(exc))
            raise
        is_error = isinstance(result, CallToolResult) and result.isError
        record(not is_error, _result_text(result) if is_error else None)
        return result


def _result_text(result: MCPToolCallResult) -> str:
    if isinstance(result, CallToolResult):
        return " ".join(c.text for c in result.content if isinstance(c, TextContent))
    return str(result)
