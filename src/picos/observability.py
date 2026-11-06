"""Optional LangSmith tracing.

LangChain/LangGraph trace every model and graph call to LangSmith when `LANGSMITH_TRACING` is
true and `LANGSMITH_API_KEY` is set. We decide that flag once at startup:
- `observability.langsmith: auto` -> tracing on iff `LANGSMITH_API_KEY` is set (an explicit
  `LANGSMITH_TRACING` from the environment is respected);
- `off` -> tracing always off.
Without a key everything works as before; no network calls are made.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Literal

log = logging.getLogger(__name__)


def configure_langsmith(
    mode: Literal["auto", "off"], env: MutableMapping[str, str] | None = None
) -> bool:
    """Set `LANGSMITH_TRACING` in `env` (default: os.environ); return whether tracing is on."""
    env = os.environ if env is None else env
    if mode == "off" or not env.get("LANGSMITH_API_KEY"):
        env["LANGSMITH_TRACING"] = "false"
        return False
    env.setdefault("LANGSMITH_TRACING", "true")
    enabled = env["LANGSMITH_TRACING"].strip().lower() == "true"
    if enabled:
        log.info("LangSmith tracing on (project: %s)", env.get("LANGSMITH_PROJECT", "default"))
    return enabled
