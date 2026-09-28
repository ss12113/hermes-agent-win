"""Claude-style `/effort` compatibility helpers for Hermes.

Hermes's native knob is ``agent.reasoning_effort`` with values
``minimal|low|medium|high|xhigh|max|ultra`` plus ``none`` through `/reasoning`.
The `/effort` compatibility surface keeps ``xhigh`` and ``max`` distinct so
models such as GPT-5.6 can receive their true maximum wire value.  This module
keeps the mapping pure and shared across CLI, gateway, and TUI routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EffortAction = Literal["status", "help", "set", "clear", "error"]

EFFORT_USAGE = """Usage: /effort [low|medium|high|xhigh|max|auto]

Effort levels:
- low: Quick, straightforward implementation
- medium: Balanced approach with standard testing
- high: Comprehensive implementation with extensive testing
- xhigh: Very high reasoning effort
- max: Maximum available Hermes reasoning effort
- auto: Use the configured/model default effort level"""

_EFFORT_DESCRIPTIONS: dict[str, str] = {
    "low": "Quick, straightforward implementation",
    "medium": "Balanced approach with standard testing",
    "high": "Comprehensive implementation with extensive testing",
    "xhigh": "Very high reasoning effort",
    "max": "Maximum available Hermes reasoning effort",
    "auto": "Use the configured/model default effort level",
    "none": "Reasoning disabled via /reasoning none",
}


@dataclass(frozen=True)
class EffortCommandResult:
    action: EffortAction
    message: str
    reasoning_value: str | None = None
    display_value: str | None = None


def display_effort_value(value: str | None) -> str:
    """Return the Claude-style display label for a Hermes reasoning value."""

    normalized = (value or "").strip().lower()
    if not normalized:
        return "auto"
    if normalized in {"xhigh", "max"}:
        return normalized
    return normalized


def reasoning_value_for_effort(value: str) -> str | None:
    """Map a Claude-style effort label to Hermes's persisted value."""

    normalized = value.strip().lower()
    if normalized in {"auto", "unset", "reset"}:
        return ""
    if normalized in {"max", "xhigh"}:
        return normalized
    if normalized in {"low", "medium", "high"}:
        return normalized
    return None


def parse_effort_command_arg(arg: str | None) -> EffortCommandResult:
    """Parse `/effort` arguments without touching config or session state."""

    raw = (arg or "").strip()
    normalized = raw.lower()
    if normalized in {"", "current", "status"}:
        return EffortCommandResult(action="status", message="")
    if normalized in {"help", "-h", "--help"}:
        return EffortCommandResult(action="help", message=EFFORT_USAGE)
    if normalized in {"auto", "unset", "reset"}:
        return EffortCommandResult(
            action="clear",
            message="Effort level set to auto",
            reasoning_value="",
            display_value="auto",
        )

    reasoning_value = reasoning_value_for_effort(normalized)
    if reasoning_value is None:
        return EffortCommandResult(
            action="error",
            message=f"Invalid argument: {raw}. Valid options are: low, medium, high, xhigh, max, auto",
        )

    display = display_effort_value(reasoning_value)
    desc = _EFFORT_DESCRIPTIONS.get(display, display)
    return EffortCommandResult(
        action="set",
        message=f"Set effort level to {display}: {desc}",
        reasoning_value=reasoning_value,
        display_value=display,
    )


def _effort_from_reasoning_config(reasoning_config: dict | None) -> str:
    if not reasoning_config:
        return "auto"
    if reasoning_config.get("enabled") is False:
        return "none"
    return display_effort_value(str(reasoning_config.get("effort") or "medium"))


def format_effort_status(
    reasoning_config: dict | None = None,
    *,
    config_value: str | None = None,
) -> str:
    """Format current effort state for CLI/Gateway/TUI status output."""

    if config_value is not None:
        display = display_effort_value(config_value)
    else:
        display = _effort_from_reasoning_config(reasoning_config)

    if display == "auto":
        return "Effort level: auto (Hermes will use the configured/model default, usually medium)"
    description = _EFFORT_DESCRIPTIONS.get(display, display)
    return f"Current effort level: {display} ({description})"
