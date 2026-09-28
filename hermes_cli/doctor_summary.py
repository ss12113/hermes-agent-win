"""Lightweight /doctor diagnostics summary for slash-command surfaces.

Claude Code's ``/doctor`` opens a read-only diagnostics pane. Hermes already has
an exhaustive terminal subcommand (``hermes doctor``) that can run network probes
and optional fixes; slash-command surfaces need a lighter, transport-neutral
view that is safe to run mid-session and from gateways.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

_CONTEXT_FILE_WARN_CHARS = 40_000
_TOOL_SCHEMA_WARN_TOKENS = 25_000
_MAX_LISTED_ITEMS = 5


def _display_path(value: str | os.PathLike[str] | None) -> str:
    """Return a user-facing path without leaking anything beyond the path text."""

    if value is None:
        return "unknown"
    text = str(value) or "unknown"
    try:
        home = str(Path.home())
        if home and text == home:
            return "~"
        if home and text.startswith(home + os.sep):
            return "~" + text[len(home):]
    except Exception:
        pass
    return text


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return _display_path(path)


def _hermes_version() -> str:
    try:
        from hermes_cli import __version__

        if __version__:
            return str(__version__)
    except Exception:
        pass
    for package_name in ("hermes-agent", "hermes_cli", "hermes-cli"):
        try:
            return metadata.version(package_name)
        except Exception:
            continue
    return "unknown"


def _project_root() -> Path:
    try:
        from hermes_cli.config import get_project_root

        return Path(get_project_root())
    except Exception:
        return Path(__file__).resolve().parents[1]


def _hermes_home_display() -> str:
    try:
        from hermes_constants import display_hermes_home

        return str(display_hermes_home())
    except Exception:
        try:
            from hermes_cli.config import get_hermes_home

            return _display_path(get_hermes_home())
        except Exception:
            return _display_path(Path.home() / ".hermes")


def _hermes_home_path() -> Path:
    try:
        from hermes_cli.config import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _installation_type(root: Path) -> str:
    if os.environ.get("HERMES_CONTAINER") or (Path("/.dockerenv").exists()):
        return "container/source"
    if (root / ".git").exists():
        return "development/source"
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "virtualenv"
    return "python-package"


def _invoked_binary(root: Path) -> str:
    candidates = [
        sys.argv[0] if sys.argv else "",
        shutil.which("hermes") or "",
        sys.executable,
    ]
    for candidate in candidates:
        if candidate:
            try:
                path = Path(candidate)
                if path.exists():
                    return _display_path(path.resolve())
            except Exception:
                return str(candidate)
            return str(candidate)
    return _display_path(root)


def _ripgrep_line() -> tuple[str, str | None]:
    rg_path = shutil.which("rg")
    if rg_path:
        return f"└ Search: OK ({_display_path(rg_path)})", None
    return "└ Search: Python fallback (ripgrep `rg` not found)", "Install ripgrep for faster repository search."


def _configuration_lines() -> tuple[list[str], list[str]]:
    """Return display lines plus warning/action lines for local config state."""

    lines: list[str] = ["Configuration"]
    warnings: list[str] = []
    home = _hermes_home_path()
    home_display = _hermes_home_display()

    config_path = home / "config.yaml"
    env_path = home / ".env"
    lines.append(
        f"└ config.yaml: {'found' if config_path.exists() else 'missing'} ({home_display}/config.yaml)"
    )
    lines.append(
        f"└ .env: {'found' if env_path.exists() else 'missing'} ({home_display}/.env; values redacted)"
    )

    try:
        from hermes_cli.config import validate_config_structure

        issues = validate_config_structure() or []
    except Exception as exc:
        warnings.append(f"Config validation skipped: {exc}")
        return lines, warnings

    if not issues:
        lines.append("└ Structure: OK")
        return lines, warnings

    lines.append(f"└ Structure: {len(issues)} issue(s)")
    for issue in issues[:_MAX_LISTED_ITEMS]:
        severity = str(getattr(issue, "severity", "warning") or "warning").upper()
        message = str(getattr(issue, "message", issue))
        lines.append(f"  └ [{severity}] {message}")
        hint = str(getattr(issue, "hint", "") or "").strip()
        if hint:
            first_hint = hint.splitlines()[0]
            lines.append(f"    Fix: {first_hint}")
        warnings.append(message)
    if len(issues) > _MAX_LISTED_ITEMS:
        lines.append(f"  └ ... {len(issues) - _MAX_LISTED_ITEMS} more")
    return lines, warnings


def _security_lines() -> tuple[list[str], list[str]]:
    lines = ["Security Advisories"]
    warnings: list[str] = []
    try:
        from hermes_cli.security_advisories import detect_compromised, filter_unacked

        hits = filter_unacked(detect_compromised())
    except Exception as exc:
        lines.append(f"└ Check skipped: {exc}")
        return lines, [f"Security advisory check skipped: {exc}"]

    if not hits:
        lines.append("└ No active security advisories")
        return lines, warnings

    for hit in hits[:_MAX_LISTED_ITEMS]:
        advisory = hit.advisory
        lines.append(
            f"└ {advisory.severity.upper()}: {advisory.id} — {hit.package}=={hit.installed_version}"
        )
        if advisory.remediation:
            lines.append(f"  Fix: {advisory.remediation[0]}")
        warnings.append(f"Resolve security advisory {advisory.id}")
    if len(hits) > _MAX_LISTED_ITEMS:
        lines.append(f"└ ... {len(hits) - _MAX_LISTED_ITEMS} more")
    return lines, warnings


def _context_warning_lines(cwd: str | os.PathLike[str] | None = None) -> tuple[list[str], list[str]]:
    lines = ["Context Usage Warnings"]
    warnings: list[str] = []
    home = _hermes_home_path()
    cwd_path = Path(cwd or os.getcwd())
    candidates = [
        home / "memories" / "MEMORY.md",
        home / "memories" / "USER.md",
        home / "SOUL.md",
        cwd_path / "AGENTS.md",
        cwd_path / "CLAUDE.md",
    ]

    large_files: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _CONTEXT_FILE_WARN_CHARS:
            large_files.append((path, size))

    if not large_files:
        lines.append("└ No large Hermes memory/persona files detected")
        return lines, warnings

    for path, size in sorted(large_files, key=lambda item: item[1], reverse=True)[:_MAX_LISTED_ITEMS]:
        label = _safe_rel(path, cwd_path)
        lines.append(f"└ ⚠ {label}: {size:,} chars")
        warnings.append(f"Large context file: {label}")
    if len(large_files) > _MAX_LISTED_ITEMS:
        lines.append(f"└ ... {len(large_files) - _MAX_LISTED_ITEMS} more")
    return lines, warnings


def _provider_model_lines(provider: str | None, model: str | None, session_id: str | None) -> list[str]:
    lines: list[str] = []
    clean_provider = (provider or "").strip()
    clean_model = (model or "").strip()
    clean_session = (session_id or "").strip()
    if clean_provider or clean_model:
        if clean_provider and clean_model:
            lines.append(f"└ Runtime model: {clean_provider}/{clean_model}")
        elif clean_model:
            lines.append(f"└ Runtime model: {clean_model}")
        else:
            lines.append(f"└ Runtime provider: {clean_provider}")
    if clean_session:
        lines.append(f"└ Session: {clean_session}")
    return lines


def _tool_schema_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if isinstance(fn, dict):
        name = fn.get("name")
    else:
        name = tool.get("name")
    return str(name or "unknown")


def _stable_schema_text(tool: dict[str, Any] | list[dict[str, Any]]) -> str:
    return json.dumps(tool, ensure_ascii=False, sort_keys=True, default=str)


def _tool_schema_token_estimate(tools: list[dict[str, Any]]) -> int:
    try:
        from agent.model_metadata import estimate_request_tokens_rough

        return int(estimate_request_tokens_rough([], tools=tools))
    except Exception:
        return (len(_stable_schema_text(tools)) + 3) // 4


def _tool_schema_warning_lines(
    runtime_tools: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Summarize live tool-schema context pressure without rebuilding toolsets."""

    lines = ["Tool Schema Context"]
    warnings: list[str] = []
    if runtime_tools is None:
        lines.append("└ Runtime schemas: unavailable until a session agent is built")
        return lines, warnings

    tools = [tool for tool in runtime_tools if isinstance(tool, dict)]
    token_estimate = _tool_schema_token_estimate(tools)
    lines.append(f"└ Runtime schemas: {len(tools)} tools, ~{token_estimate:,} tokens")

    if token_estimate <= _TOOL_SCHEMA_WARN_TOKENS:
        lines.append(f"└ Budget: OK (≤ {_TOOL_SCHEMA_WARN_TOKENS:,} token warning threshold)")
        return lines, warnings

    lines.append(f"└ ⚠ Above {_TOOL_SCHEMA_WARN_TOKENS:,} token warning threshold")
    warnings.append(
        f"Large tool schema context: ~{token_estimate:,} tokens across {len(tools)} tools"
    )

    sized = sorted(
        ((len(_stable_schema_text(tool)), tool) for tool in tools),
        reverse=True,
        key=lambda item: item[0],
    )
    for byte_count, tool in sized[:_MAX_LISTED_ITEMS]:
        lines.append(f"  └ {_tool_schema_name(tool)}: ~{(byte_count + 3) // 4:,} tokens")
    if len(sized) > _MAX_LISTED_ITEMS:
        lines.append(f"  └ ... {len(sized) - _MAX_LISTED_ITEMS} more")
    lines.append(
        "  Fix: disable rarely used toolsets or enable Tool Search for large MCP/plugin catalogs."
    )
    return lines, warnings


def format_doctor_summary(
    *,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    runtime_tools: list[dict[str, Any]] | None = None,
) -> str:
    """Format a read-only, transport-neutral Hermes diagnostics summary."""

    root = _project_root()
    cwd_path = Path(cwd or os.getcwd())
    search_line, search_warning = _ripgrep_line()
    config_lines, config_warnings = _configuration_lines()
    security_lines, security_warnings = _security_lines()
    context_lines, context_warnings = _context_warning_lines(cwd_path)
    tool_lines, tool_warnings = _tool_schema_warning_lines(runtime_tools)

    warnings = [
        *( [search_warning] if search_warning else [] ),
        *config_warnings,
        *security_warnings,
        *context_warnings,
        *tool_warnings,
    ]

    lines = [
        "Hermes Doctor diagnostics",
        "",
        "Diagnostics",
        f"└ Currently running: Hermes Agent ({_hermes_version()})",
        f"└ Installation type: {_installation_type(root)}",
        f"└ Path: {_display_path(root)}",
        f"└ Invoked: {_invoked_binary(root)}",
        f"└ Python: {platform.python_version()} ({_display_path(sys.executable)})",
        f"└ Hermes home: {_hermes_home_display()}",
        *_provider_model_lines(provider, model, session_id),
        search_line,
        "",
        "Updates",
        "└ Auto-updates: use `/update` in Hermes or run `hermes update` from a terminal",
        "└ Full diagnostics and optional fixes: `hermes doctor` (terminal)",
        "",
        *config_lines,
        "",
        *security_lines,
        "",
        *context_lines,
        "",
        *tool_lines,
    ]

    if warnings:
        lines.extend(["", "Warnings / recommended actions"])
        for warning in warnings[:_MAX_LISTED_ITEMS]:
            lines.append(f"- {warning}")
        if len(warnings) > _MAX_LISTED_ITEMS:
            lines.append(f"- ... {len(warnings) - _MAX_LISTED_ITEMS} more")
    else:
        lines.extend(["", "No blocking issues found by the lightweight slash diagnostics."])

    return "\n".join(str(line) for line in lines)


__all__ = ["format_doctor_summary"]
