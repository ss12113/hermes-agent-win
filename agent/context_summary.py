"""Claude-compatible /context summary for Hermes sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.context_usage import estimate_live_context_usage
from agent.model_metadata import estimate_messages_tokens_rough


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tool_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _fmt_tokens(value: int) -> str:
    return f"{max(0, int(value)):,}"


def _pct(tokens: int, context_length: int) -> str:
    if context_length <= 0:
        return "—"
    return f"{max(0.0, tokens / context_length * 100):.1f}%"


def _source_label(source: str) -> str:
    return {
        "reported": "provider-reported last prompt",
        "estimated": "rough estimate from live request payload",
        "session_total": "session token-total fallback",
        "none": "no token data yet",
    }.get(source, source or "unknown")


def _display_model(agent: Any, *, provider: str | None = None, model: str | None = None) -> str:
    provider_name = provider or getattr(agent, "provider", None) or ""
    model_name = model or getattr(agent, "model", None) or ""
    if provider_name and model_name:
        return f"{provider_name}/{model_name}"
    return model_name or provider_name or "unknown"


def _last_compaction_lines(compressor: Any) -> list[str]:
    """Format the provider-backed verdict for the latest compaction boundary."""
    verdict = str(getattr(compressor, "_last_compaction_verdict", "none") or "none")
    if verdict not in {"pending", "cleared", "retrigger", "paused", "unverified"}:
        return []

    prompt_tokens = max(
        0,
        _as_int(getattr(compressor, "_last_compaction_prompt_tokens", 0)),
    )
    threshold_tokens = max(
        0,
        _as_int(
            getattr(compressor, "_last_compaction_threshold_tokens", 0)
            or getattr(compressor, "threshold_tokens", 0)
        ),
    )
    rough_tokens = max(
        0,
        _as_int(getattr(compressor, "last_compression_rough_tokens", 0)),
    )
    rough_is_post_compaction = bool(
        getattr(compressor, "_last_compaction_rough_is_post_compaction", True)
    )
    strikes = max(
        0,
        _as_int(
            getattr(
                compressor,
                "_last_compaction_ineffective_count",
                getattr(compressor, "_ineffective_compression_count", 0),
            )
        ),
    )

    verdict_label = {
        "pending": "pending provider verification",
        "cleared": "cleared the auto-compaction threshold",
        "retrigger": "still above threshold; one retry remains eligible",
        "paused": "still above threshold; auto-compaction paused",
        "unverified": "provider usage unavailable; verdict not charged later",
    }[verdict]

    if verdict == "pending":
        if not rough_is_post_compaction:
            risk = "native runtime estimate unavailable; provider result decides"
        elif threshold_tokens and rough_tokens >= threshold_tokens:
            risk = "estimate remains at/above threshold; provider result decides"
        elif threshold_tokens and rough_tokens:
            risk = "estimate is below threshold; provider result decides"
        else:
            risk = "provider result decides"
    elif verdict == "cleared":
        risk = "automatic compaction not expected"
    elif verdict == "retrigger":
        risk = "eligible to retry after 1 ineffective compaction"
    elif verdict == "paused":
        risk = f"paused after {strikes} ineffective compactions"
    else:
        risk = "not adjudicated; no unrelated later request will be charged"

    lines = [
        "",
        "### Last compaction",
        "",
        f"**Verdict:** {verdict_label}  ",
    ]
    if prompt_tokens:
        lines.append(f"**Provider prompt at verdict:** {_fmt_tokens(prompt_tokens)}  ")
    if rough_tokens and rough_is_post_compaction:
        lines.append(f"**Rough request after rewrite:** ~{_fmt_tokens(rough_tokens)}  ")
    elif rough_tokens:
        lines.append("**Rough request after rewrite:** unavailable for native runtime  ")
    if threshold_tokens:
        lines.append(f"**Threshold at verdict:** {_fmt_tokens(threshold_tokens)}  ")
    lines.append(f"**Next-turn risk:** {risk}")
    return lines


def _summary_route_health_lines(compressor: Any) -> list[str]:
    """Expose the bounded auto-summary failure circuit without provider I/O."""
    failures = max(
        0,
        _as_int(getattr(compressor, "_consecutive_auto_summary_failures", 0)),
    )
    if failures <= 0:
        return []
    limit = max(
        1,
        _as_int(getattr(compressor, "_auto_summary_failure_limit", 3)) or 3,
    )
    if failures >= limit:
        status = "paused until a successful manual `/compress` or `/new`"
    else:
        status = "retry eligible after the current cooldown"
    return [
        "",
        "### Summary route health",
        "",
        f"**Consecutive automatic summary failures:** {failures}  ",
        f"**Automatic compaction:** {status}",
    ]


def format_context_usage_summary(
    agent: Any = None,
    *,
    fallback_messages: Any = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Return a markdown /context summary.

    Hermes does not render Claude's colored terminal grid in gateway/Desktop
    surfaces, so the compatibility command uses Claude's non-interactive shape:
    a concise markdown table grounded in Hermes' live context estimator.
    """

    usage = estimate_live_context_usage(agent, fallback_messages=fallback_messages)
    context_length = max(0, usage.context_length)
    percent = _pct(usage.tokens, context_length)
    token_prefix = "~" if usage.estimated else ""

    messages = _message_list(getattr(agent, "_session_messages", None))
    if not messages:
        messages = _message_list(fallback_messages)
    system_prompt = getattr(agent, "_cached_system_prompt", "")
    if not isinstance(system_prompt, str):
        system_prompt = ""
    tools = _tool_list(getattr(agent, "tools", None))

    system_tokens = (len(system_prompt) + 3) // 4 if system_prompt else 0
    message_tokens = estimate_messages_tokens_rough(messages) if messages else 0
    tool_tokens = (len(str(tools)) + 3) // 4 if tools else 0
    free_tokens = max(0, context_length - usage.tokens) if context_length else 0
    compressions = _as_int(getattr(getattr(agent, "context_compressor", None), "compression_count", 0))

    if context_length:
        token_line = f"{token_prefix}{_fmt_tokens(usage.tokens)} / {_fmt_tokens(context_length)} ({percent})"
    else:
        token_line = f"{token_prefix}{_fmt_tokens(usage.tokens)}"

    lines = [
        "## Context Usage",
        "",
        f"**Model:** {_display_model(agent, provider=provider, model=model)}  ",
        f"**Tokens:** {token_line}  ",
        f"**Source:** {_source_label(usage.source)}  ",
        f"**Messages:** {_fmt_tokens(len(messages))}  ",
        f"**Tool schemas:** {_fmt_tokens(len(tools))}  ",
        f"**Compressions:** {_fmt_tokens(compressions)}",
        "",
        "### Estimated usage by category",
        "",
        "| Category | Tokens | Percentage |",
        "|----------|--------|------------|",
        f"| System prompt | {_fmt_tokens(system_tokens)} | {_pct(system_tokens, context_length)} |",
        f"| Conversation messages | {_fmt_tokens(message_tokens)} | {_pct(message_tokens, context_length)} |",
        f"| Tool schemas | {_fmt_tokens(tool_tokens)} | {_pct(tool_tokens, context_length)} |",
    ]
    if context_length:
        lines.append(f"| Free space | {_fmt_tokens(free_tokens)} | {_pct(free_tokens, context_length)} |")
    lines.extend(_last_compaction_lines(getattr(agent, "context_compressor", None)))
    lines.extend(_summary_route_health_lines(getattr(agent, "context_compressor", None)))
    lines.extend(
        [
            "",
            "Note: `~` means Hermes is estimating the next request payload; exact provider counts appear after an API call.",
        ]
    )
    return "\n".join(lines).rstrip()
