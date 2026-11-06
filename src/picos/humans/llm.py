"""LLM-backed simulated human: persona prompt + structured `HumanReply` output.

The human keeps its own conversation history (independent of any agent state). Model calls
use the same recorded retry policy as the agents and emit `llm_call` events.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.messages import HumanMessage as LCHumanMessage

from picos.agents.llm import ainvoke_with_retry, model_name
from picos.config import RetryConfig
from picos.domain import HumanSpec
from picos.protocol import HumanMessage, HumanReply
from picos.runtime.recorder import Recorder


class HumanReplyError(RuntimeError):
    """The model did not produce a valid HumanReply."""


def render_persona(template: str, spec: HumanSpec) -> str:
    persona = spec.persona

    def bullets(items: Sequence[str]) -> str:
        return "\n".join(f"- {i}" for i in items) or "- (none)"

    return template.format(
        name=spec.name,
        style=persona.style,
        known_facts=bullets(persona.known_facts),
        hidden_facts=bullets([f.text for f in persona.hidden_facts]),
        decisions=json.dumps(persona.decisions.model_dump(), sort_keys=True),
    )


def format_for_human(inbox: Sequence[HumanMessage]) -> str:
    blocks = []
    for m in inbox:
        block = f"[{m.kind} from your assistant]\n{m.text}"
        if m.final_plan is not None:
            block += "\nPlan details: " + m.final_plan.model_dump_json()
        blocks.append(block)
    return "\n\n".join(blocks)


class LLMHumanProvider:
    def __init__(
        self,
        actor: str,
        spec: HumanSpec,
        llm: BaseChatModel,
        template: str,
        *,
        retry: RetryConfig,
        recorder: Recorder,
    ) -> None:
        self.actor = actor
        self.system = SystemMessage(render_persona(template, spec))
        self.model = model_name(llm)
        self.structured = llm.with_structured_output(HumanReply, include_raw=True)
        self.retry = retry
        self.recorder = recorder
        self.history: list[BaseMessage] = []

    async def reply(self, inbox: Sequence[HumanMessage]) -> HumanReply:
        self.history.append(LCHumanMessage(content=format_for_human(inbox)))
        started = time.perf_counter()
        out: dict[str, Any] = await ainvoke_with_retry(
            self.structured,
            [self.system, *self.history],
            retry=self.retry,
            recorder=self.recorder,
            actor=self.actor,
            model=self.model,
        )
        raw = out.get("raw")
        usage = (raw.usage_metadata if isinstance(raw, AIMessage) else None) or {}
        self.recorder.record(
            self.actor,
            "llm_call",
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        parsed = out.get("parsed")
        if not isinstance(parsed, HumanReply):
            raise HumanReplyError(f"no valid HumanReply from the model: {out.get('parsing_error')}")
        self.history.append(AIMessage(content=parsed.text))
        return parsed
