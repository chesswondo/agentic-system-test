"""What the LLM sees of an agent's history.

Bulky environment results (forecasts, trail catalogs) are needed only in the turn they
arrive: the decision tools read them from the per-agent observation cache afterwards. Resending
them on every later LLM call made Agent A's context grow to ~22k tokens per call.
`compact_history` replaces such results from earlier turns with a short stub. It is a pure
view: the checkpointed state keeps the full messages.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage


def stub_text(tool: str, size: int) -> str:
    return (
        f"[compacted: successful result of {tool} from an earlier turn ({size} chars). "
        "The data is cached for the decision tools; call the tool again if you need details.]"
    )


def compact_history(
    messages: Sequence[BaseMessage], tools: Collection[str]
) -> tuple[list[BaseMessage], int]:
    """Return the messages with old results of `tools` stubbed, and how many were stubbed.

    "Earlier turn" = before the last inbox message (each turn starts with one HumanMessage).
    Error results are short and informative, so they are kept as they are."""
    if not tools:
        return list(messages), 0
    turn_start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    out: list[BaseMessage] = []
    compacted = 0
    for i, msg in enumerate(messages):
        if (
            i < turn_start
            and isinstance(msg, ToolMessage)
            and msg.name in tools
            and msg.status != "error"
        ):
            size = len(msg.text)
            msg = msg.model_copy(update={"content": stub_text(str(msg.name), size)})
            compacted += 1
        out.append(msg)
    return out, compacted
