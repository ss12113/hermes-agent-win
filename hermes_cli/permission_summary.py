# pyright: reportMissingImports=false
"""Claude-style /permissions and /sandbox summaries for Hermes.

Claude Code exposes interactive permission/sandbox editors.  Hermes' native
surface is different (approval guardrails + terminal backends), so these helpers
provide a read-only compatibility/status view that is safe to render in CLI,
gateway, TUI, and desktop command surfaces.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from typing import Any

_SANDBOXED_TERMINAL_BACKENDS = {
    "docker",
    "podman",
    "container",
    "singularity",
    "modal",
    "managed_modal",
    "daytona",
}
_SANDBOX_STATUS_ARGS = {"", "status", "current", "show"}
_SANDBOX_HELP_ARGS = {"help", "usage", "-h", "--help"}


def _load_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, Mapping) else {}
    except Exception:
        return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_approval_mode(value: Any) -> str:
    raw = str(value or "manual").strip().lower()
    if raw in {"off", "disable", "disabled", "false", "0", "no", "never", "allow", "approve"}:
        return "off"
    if raw in {"smart", "auto", "llm", "model"}:
        return "smart"
    return "manual"


def _normalize_cron_mode(value: Any) -> str:
    raw = str(value or "deny").strip().lower()
    if raw in {"approve", "off", "allow", "yes", "true", "1"}:
        return "approve"
    return "deny"


def _bool_label(value: bool) -> str:
    return "on" if value else "off"


def _coerce_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = 60
    return max(0, timeout)


def _allowlist_size(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 0


def _policy_count(cfg: Mapping[str, Any], approvals: Mapping[str, Any], key: str) -> int:
    """Count a policy list from either legacy top-level or nested approvals config."""
    nested = _allowlist_size(approvals.get(key))
    if nested:
        return nested
    return _allowlist_size(_mapping(cfg).get(key))


def _terminal_backend_snapshot(cfg: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve terminal backend the same way runtime surfaces see it.

    CLI/TUI/gateway bridge ``terminal.backend`` into ``TERMINAL_ENV`` before tool
    execution. Prefer that live env var so `/sandbox` reports the actual active
    backend, then fall back to documented (`backend`) and legacy (`env_type`)
    config keys.
    """
    env_backend = str(os.getenv("TERMINAL_ENV") or "").strip().lower()
    if env_backend:
        return env_backend, "TERMINAL_ENV"

    terminal_cfg = _mapping(_mapping(cfg).get("terminal"))
    for key in ("backend", "env_type"):
        value = str(terminal_cfg.get(key) or "").strip().lower()
        if value:
            return value, f"config.yaml terminal.{key}"

    return "local", "default"


def _sandbox_usage() -> str:
    return "\n".join(
        [
            "## Sandbox",
            "",
            "Usage:",
            "- `/sandbox` or `/sandbox status` — show current terminal isolation and approval posture",
            "- `/sandbox current` / `/sandbox show` — same status view",
            "- `/sandbox help` — show this help",
            "- `/sandbox exclude \"command pattern\"` — parse a Claude-style exclusion request and show Hermes equivalents without changing config",
            "",
            "Compatibility aliases: `/sandbox-toggle`, `/sandbox_toggle`.",
            "Read-only: this command never changes config. Use `/permissions` or an explicit config edit for approval policy changes.",
        ]
    )


def _extract_sandbox_exclude_pattern(arg: str) -> str:
    rest = arg.strip().split(None, 1)
    if len(rest) < 2:
        return ""
    pattern = rest[1].strip()
    if not pattern:
        return ""
    try:
        parts = shlex.split(pattern)
    except ValueError:
        parts = []
    if parts:
        return " ".join(parts).strip()
    return pattern.strip().strip("'\"")


def _format_sandbox_exclude(pattern: str) -> str:
    if not pattern:
        return "\n".join(
            [
                "## Sandbox",
                "",
                "Error: `/sandbox exclude` requires a command pattern.",
                "Example: `/sandbox exclude \"npm run test:*\"`",
                "Run `/sandbox help` for Hermes-compatible options.",
            ]
        )
    return "\n".join(
        [
            "## Sandbox",
            "",
            "Claude-style sandbox exclusion parsed.",
            f"Pattern received: `{len(pattern)} character(s)` (not echoed to avoid leaking secrets).",
            "No Hermes config was changed.",
            "",
            "Hermes equivalents:",
            "- Use `/permissions` to inspect current approval posture.",
            "- Add reviewed commands to `command_allowlist` for permanent approval shortcuts.",
            "- Use the normal approval prompt for one-off command decisions.",
            "",
            "This read-only shim avoids silently weakening command approvals from chat.",
        ]
    )


def _process_yolo_default() -> bool:
    try:
        from tools.approval import _YOLO_MODE_FROZEN

        return bool(_YOLO_MODE_FROZEN)
    except Exception:
        return str(os.getenv("HERMES_YOLO_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


def _session_yolo_default(session_key: str) -> bool:
    if not session_key:
        return False
    try:
        from tools.approval import is_session_yolo_enabled

        return bool(is_session_yolo_enabled(session_key))
    except Exception:
        return False


def permission_state(
    config: Mapping[str, Any] | None = None,
    *,
    session_key: str = "",
    process_yolo: bool | None = None,
    session_yolo: bool | None = None,
) -> dict[str, Any]:
    """Return a redaction-safe snapshot of Hermes' approval posture."""
    cfg = config if config is not None else _load_config()
    cfg_map = _mapping(cfg)
    approvals = _mapping(cfg_map.get("approvals"))
    mode = _normalize_approval_mode(approvals.get("mode", "manual"))
    proc_yolo = _process_yolo_default() if process_yolo is None else bool(process_yolo)
    sess_yolo = _session_yolo_default(session_key) if session_yolo is None else bool(session_yolo)
    return {
        "approval_mode": mode,
        "timeout_seconds": _coerce_timeout(approvals.get("timeout", 60)),
        "cron_mode": _normalize_cron_mode(approvals.get("cron_mode", "deny")),
        "command_allowlist_count": _policy_count(cfg_map, approvals, "command_allowlist"),
        "permanent_approval_count": _allowlist_size(approvals.get("permanent_approved")),
        "process_yolo": proc_yolo,
        "session_yolo": sess_yolo,
        "effective_bypass": bool(proc_yolo or sess_yolo or mode == "off"),
    }


def sandbox_state(
    config: Mapping[str, Any] | None = None,
    *,
    session_key: str = "",
    process_yolo: bool | None = None,
    session_yolo: bool | None = None,
) -> dict[str, Any]:
    """Return a redaction-safe snapshot of Hermes' execution isolation posture."""
    cfg = config if config is not None else _load_config()
    backend, backend_source = _terminal_backend_snapshot(_mapping(cfg))
    permissions = permission_state(
        cfg,
        session_key=session_key,
        process_yolo=process_yolo,
        session_yolo=session_yolo,
    )
    return {
        "terminal_backend": backend,
        "terminal_backend_source": backend_source,
        "sandboxed_backend": backend in _SANDBOXED_TERMINAL_BACKENDS,
        "permissions": permissions,
    }


def format_permissions_summary(
    config: Mapping[str, Any] | None = None,
    *,
    session_key: str = "",
    process_yolo: bool | None = None,
    session_yolo: bool | None = None,
) -> str:
    state = permission_state(
        config,
        session_key=session_key,
        process_yolo=process_yolo,
        session_yolo=session_yolo,
    )
    lines = [
        "## Permissions",
        "",
        f"Dangerous command approval mode: `{state['approval_mode']}`",
        f"Effective approval bypass: `{_bool_label(bool(state['effective_bypass']))}`",
        f"Process YOLO: `{_bool_label(bool(state['process_yolo']))}`",
        f"Session YOLO: `{_bool_label(bool(state['session_yolo']))}`",
        f"Approval timeout: `{state['timeout_seconds']}s`",
        f"Cron dangerous-command mode: `{state['cron_mode']}`",
        f"Command allowlist entries: `{state['command_allowlist_count']}`",
        f"Permanent approval entries: `{state['permanent_approval_count']}`",
        "",
        "Claude compatibility: `/allowed-tools` is an alias for `/permissions`.",
        "Use `/yolo` to toggle session-scoped auto-approval for dangerous commands.",
    ]
    return "\n".join(lines)


def format_sandbox_summary(
    config: Mapping[str, Any] | None = None,
    *,
    arg: str = "",
    session_key: str = "",
    process_yolo: bool | None = None,
    session_yolo: bool | None = None,
) -> str:
    subcommand = (arg or "").strip()
    normalized = subcommand.lower()
    if normalized in _SANDBOX_HELP_ARGS:
        return _sandbox_usage()
    if normalized.startswith("exclude"):
        return _format_sandbox_exclude(_extract_sandbox_exclude_pattern(subcommand))
    if normalized not in _SANDBOX_STATUS_ARGS:
        return "\n".join(
            [
                "## Sandbox",
                "",
                "Error: unknown `/sandbox` subcommand.",
                "Run `/sandbox help` for supported Hermes-compatible options.",
            ]
        )

    state = sandbox_state(
        config,
        session_key=session_key,
        process_yolo=process_yolo,
        session_yolo=session_yolo,
    )
    permissions = state["permissions"]
    sandbox_label = "on" if state["sandboxed_backend"] else "off"
    lines = [
        "## Sandbox",
        "",
        f"Terminal backend: `{state['terminal_backend']}`",
        f"Terminal backend source: `{state['terminal_backend_source']}`",
        f"Effective terminal sandbox: `{sandbox_label}`",
        f"Dangerous command approval mode: `{permissions['approval_mode']}`",
        f"Effective approval bypass: `{_bool_label(bool(permissions['effective_bypass']))}`",
        f"Command allowlist entries: `{permissions['command_allowlist_count']}`",
        "",
        "Hermes maps Claude-style `/sandbox` to its execution posture:",
        "- terminal isolation comes from `terminal.backend` / `TERMINAL_ENV`",
        "- risky command execution is governed by approvals and `/yolo`",
        "- `/sandbox exclude ...` is parsed as read-only guidance, not a silent config mutation",
    ]
    if not state["sandboxed_backend"]:
        lines.append("")
        lines.append("Current backend is local/SSH-like, so terminal/file actions may run with direct host or remote-user access.")
    return "\n".join(lines)
