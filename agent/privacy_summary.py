"""Read-only privacy/redaction summary for slash-command surfaces.

Claude Code exposes a local ``/privacy-settings`` command that opens data
privacy controls. Hermes has different privacy controls (secret redaction,
gateway PII redaction, private URL guards, browser recording), so this module
formats a transport-neutral status summary instead of mutating config from chat.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_ENV_UNSET = object()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return False
    return default


def _status(enabled: bool) -> str:
    return "enabled" if enabled else "disabled"


def _allowed(enabled: bool) -> str:
    return "allowed" if enabled else "blocked"


def _redaction_env_enabled(value: Any) -> bool:
    """Mirror agent.redact's import-time env parsing without revealing value."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        return cfg if isinstance(cfg, Mapping) else {}
    except Exception:
        return {}


def _live_redact_enabled() -> bool:
    try:
        from agent import redact

        return bool(getattr(redact, "_REDACT_ENABLED", True))
    except Exception:
        # Secure default; the redactor itself defaults to enabled.
        return True


def collect_privacy_settings(
    config: Mapping[str, Any] | None = None,
    *,
    live_redact_enabled: bool | None = None,
    redact_env_value: Any = _ENV_UNSET,
) -> dict[str, bool]:
    """Return normalized privacy/security toggles for display.

    ``config`` is intentionally injectable so tests and callers can summarize a
    read-only snapshot without touching the user's real ``config.yaml``. Missing
    keys use Hermes's secure/default behavior. ``redact_env_value`` mirrors the
    import-time ``HERMES_REDACT_SECRETS`` signal without printing the raw env
    value, matching Claude-style privacy status by naming the controlling
    signal instead of exposing sensitive process state.
    """
    cfg = _as_mapping(config) if config is not None else _load_config()
    security = _as_mapping(cfg.get("security"))
    privacy = _as_mapping(cfg.get("privacy"))
    browser = _as_mapping(cfg.get("browser"))

    redaction_config = _bool_value(security.get("redact_secrets"), default=True)
    live_redaction = _live_redact_enabled() if live_redact_enabled is None else bool(live_redact_enabled)
    env_value = os.getenv("HERMES_REDACT_SECRETS") if redact_env_value is _ENV_UNSET else redact_env_value

    return {
        "security.redact_secrets": redaction_config,
        "live.secret_redaction": live_redaction,
        "env.HERMES_REDACT_SECRETS.set": env_value is not None,
        "env.HERMES_REDACT_SECRETS.enabled": True if env_value is None else _redaction_env_enabled(env_value),
        "privacy.redact_pii": _bool_value(privacy.get("redact_pii"), default=False),
        "security.allow_private_urls": _bool_value(security.get("allow_private_urls"), default=False),
        "browser.allow_private_urls": _bool_value(browser.get("allow_private_urls"), default=False),
        "browser.record_sessions": _bool_value(browser.get("record_sessions"), default=False),
    }


def format_privacy_summary(
    config: Mapping[str, Any] | None = None,
    *,
    live_redact_enabled: bool | None = None,
    redact_env_value: Any = _ENV_UNSET,
) -> str:
    """Format a transport-neutral privacy status summary."""
    settings = collect_privacy_settings(
        config,
        live_redact_enabled=live_redact_enabled,
        redact_env_value=redact_env_value,
    )

    config_redaction = settings["security.redact_secrets"]
    live_redaction = settings["live.secret_redaction"]
    mismatch = config_redaction != live_redaction
    env_signal = (
        f"set → {_status(settings['env.HERMES_REDACT_SECRETS.enabled'])}"
        if settings["env.HERMES_REDACT_SECRETS.set"]
        else "unset"
    )

    lines = [
        "Privacy and redaction settings",
        "",
        f"- Secret redaction (config security.redact_secrets): {_status(config_redaction)}",
        f"- Secret redaction (live import snapshot): {_status(live_redaction)}",
        f"- Secret redaction env signal (HERMES_REDACT_SECRETS): {env_signal}",
    ]
    if mismatch:
        lines.append(
            "  Restart Hermes for the config change to affect live tool-output redaction."
        )

    lines.extend(
        [
            f"- Gateway PII redaction (privacy.redact_pii): {_status(settings['privacy.redact_pii'])}",
            f"- Terminal/web private URLs (security.allow_private_urls): {_allowed(settings['security.allow_private_urls'])}",
            f"- Browser private/internal URLs (browser.allow_private_urls): {_allowed(settings['browser.allow_private_urls'])}",
            f"- Browser session recording (browser.record_sessions): {_status(settings['browser.record_sessions'])}",
            "",
            "Notes:",
            "- This is a read-only status view; change settings with `hermes config edit` or by editing config.yaml from a terminal.",
            "- PII redaction can hash gateway user IDs and strip phone numbers before model context; Slack/Discord mention IDs may be preserved so mentions still work.",
            "- Claude-style `/privacy-settings` and `/privacy_settings` are compatibility aliases for `/privacy`.",
        ]
    )
    return "\n".join(lines)
