"""Claude-compatible rate-limit / extra-usage informational output for Hermes.

Claude Code exposes small in-product flows for plan/extra-usage choices when
limits are reached. Hermes cannot safely perform provider billing, admin
requests, or quota changes from an offline slash command, so these
compatibility shims stay read-only and point users at the Hermes surfaces that
do expose usage/model state.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SECRETISH = re.compile(
    r"(?i)(sk-[a-z0-9_-]{12,}|api[_-]?key|access[_-]?token|secret|password|bearer\s+[a-z0-9._-]{12,})"
)


def _safe_display(value: Any) -> str:
    """Return a compact user-facing value without echoing obvious secrets."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if _SECRETISH.search(text):
        return "[REDACTED]"
    return text


def _model_route_from_config(config: Mapping[str, Any] | None) -> tuple[str, str, str]:
    """Extract provider/model/service-tier display hints from a config mapping."""

    if not isinstance(config, Mapping):
        return "", "", ""

    provider = ""
    model = ""
    service_tier = ""

    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        provider = _safe_display(model_cfg.get("provider"))
        model = _safe_display(
            model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name")
        )
    elif isinstance(model_cfg, str):
        model = _safe_display(model_cfg)

    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, Mapping):
        service_tier = _safe_display(agent_cfg.get("service_tier"))

    return provider, model, service_tier


def _load_config_snapshot() -> Mapping[str, Any]:
    """Read Hermes config without network I/O; fail closed to an empty mapping."""

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return {}
    return config if isinstance(config, Mapping) else {}


def _format_limit_guidance(
    *,
    title: str,
    claude_behavior: str,
    config: Mapping[str, Any] | None = None,
    provider: Any = None,
    model: Any = None,
) -> str:
    """Render offline-safe guidance for Claude-style quota commands.

    The function intentionally performs no network calls and does not query any
    billing portal. Optional *provider*/*model* overrides let live surfaces show
    the active route; otherwise the local config is used as a best-effort hint.
    """

    cfg = _load_config_snapshot() if config is None else config
    cfg_provider, cfg_model, service_tier = _model_route_from_config(cfg)

    provider_display = _safe_display(provider) or cfg_provider or "auto / provider default"
    model_display = _safe_display(model) or cfg_model or "not configured"

    route_lines = [
        "Current route:",
        f"- Provider: {provider_display}",
        f"- Model: {model_display}",
    ]
    if service_tier:
        route_lines.append(f"- Service tier: {service_tier}")

    return "\n".join(
        [
            title,
            "",
            f"Claude Code uses this command for {claude_behavior}. Hermes does not purchase, upgrade, create admin requests, or change provider quotas from a slash command; limits are enforced by your active provider/account.",
            "",
            *route_lines,
            "",
            "Useful Hermes actions:",
            "- `/usage` — show session token usage, recent provider limit headers when available, and Nous credits/limits.",
            "- `/credits` — view or top up Nous credits when you use Nous billing.",
            "- `/model` — switch to another model/provider for the next turn.",
            "- Wait for the provider reset window, then retry (`/retry`) if the last turn hit a limit.",
            "",
            "For hard quota, billing, or plan changes, use your provider's dashboard or billing portal. This command is informational only and performs no network or purchase action.",
        ]
    )


def format_rate_limit_options_info(
    *,
    config: Mapping[str, Any] | None = None,
    provider: Any = None,
    model: Any = None,
) -> str:
    """Render the Hermes /rate-limit-options compatibility message."""

    return _format_limit_guidance(
        title="Rate limit options",
        claude_behavior="an in-app plan or extra-usage menu",
        config=config,
        provider=provider,
        model=model,
    )


def format_extra_usage_info(
    *,
    config: Mapping[str, Any] | None = None,
    provider: Any = None,
    model: Any = None,
) -> str:
    """Render the Hermes /extra-usage compatibility message.

    Claude's command can open billing/admin usage flows or submit a team admin
    request. Hermes keeps the slash surface read-only: it reports the active
    route and points at safe local commands instead of touching billing state.
    """

    return _format_limit_guidance(
        title="Extra usage options",
        claude_behavior="a billing/admin extra-usage flow when context or rate limits are hit",
        config=config,
        provider=provider,
        model=model,
    )
