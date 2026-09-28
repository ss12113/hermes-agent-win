"""Live context-window usage helpers.

Hermes receives exact prompt-token counts only after a provider call.  Between
turns (or after compression resets ``last_prompt_tokens``) status surfaces should
still show request pressure using the same rough estimator as the compression
pre-flight path: messages + cached system prompt + tool schemas.  Keep this in a
small transport-neutral helper so CLI, gateway, and TUI agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from agent.model_metadata import estimate_request_tokens_rough

ContextUsageSource = Literal["reported", "estimated", "session_total", "none"]


@dataclass(frozen=True)
class ContextUsageEstimate:
    """Current context-window usage for display surfaces."""

    tokens: int
    context_length: int
    source: ContextUsageSource

    @property
    def estimated(self) -> bool:
        return self.source in {"estimated", "session_total"}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tool_list(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    tools = [dict(item) for item in value if isinstance(item, Mapping)]
    return tools or None


def estimate_live_context_usage(
    agent: Any,
    *,
    fallback_messages: Any = None,
) -> ContextUsageEstimate:
    """Return context usage for a live/cached agent.

    Priority:
    1. Provider-reported ``context_compressor.last_prompt_tokens`` when positive.
    2. Rough request estimate from ``_session_messages`` (or caller fallback),
       cached system prompt, and tool schemas.
    3. Session total tokens as a last-resort activity estimate.

    Negative prompt-token sentinels (used immediately after compression) are
    treated as unknown, not as real usage.
    """

    compressor = getattr(agent, "context_compressor", None)
    context_length = max(0, _as_int(getattr(compressor, "context_length", 0)))
    reported = _as_int(getattr(compressor, "last_prompt_tokens", 0))
    if reported > 0:
        return ContextUsageEstimate(reported, context_length, "reported")

    messages = _message_list(getattr(agent, "_session_messages", None))
    if not messages:
        messages = _message_list(fallback_messages)

    raw_system_prompt = getattr(agent, "_cached_system_prompt", "")
    system_prompt = raw_system_prompt if isinstance(raw_system_prompt, str) else ""
    tools = _tool_list(getattr(agent, "tools", None))

    if messages or system_prompt or tools:
        estimated = estimate_request_tokens_rough(
            messages,
            system_prompt=system_prompt,
            tools=tools,
        )
        if estimated > 0:
            return ContextUsageEstimate(estimated, context_length, "estimated")

    session_total = max(0, _as_int(getattr(agent, "session_total_tokens", 0)))
    if session_total > 0:
        return ContextUsageEstimate(session_total, context_length, "session_total")

    return ContextUsageEstimate(0, context_length, "none")
