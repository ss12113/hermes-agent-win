"""Plaintext conversation export helpers.

This module backs the Claude-Code-compatible ``/export`` command.  Keep it
UI-agnostic: classic CLI, TUI, and future surfaces should all render and write
transcripts through the same functions instead of each inventing a subtly
different export format.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_ROLE_LABELS = {
    "assistant": "Assistant",
    "system": "System",
    "tool": "Tool",
    "user": "User",
}

_WINDOWS_FORBIDDEN_CHARS = '<>:"/\\|?*'
_RESERVED_WINDOWS_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def message_content_to_text(content: Any) -> str:
    """Return a readable text representation for OpenAI-style message content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts = [message_content_part_to_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content and isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return message_content_to_text(content.get("content"))
        if content.get("type") in {"image", "image_url", "input_image"}:
            return image_part_to_text(content)
        return _json_dump(content)
    return str(content)


def message_content_part_to_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return message_content_to_text(part)

    part_type = str(part.get("type") or "").strip()
    if "text" in part and isinstance(part.get("text"), str):
        return str(part["text"])
    if part_type in {"image", "image_url", "input_image"} or "image_url" in part:
        return image_part_to_text(part)
    if part_type in {"tool_result", "function_call_output"}:
        return message_content_to_text(part.get("content") or part.get("output"))
    return _json_dump(part)


def image_part_to_text(part: dict[str, Any]) -> str:
    url = part.get("image_url") or part.get("url") or part.get("source")
    if isinstance(url, dict):
        url = url.get("url") or url.get("media_type") or "image"
    if isinstance(url, str) and url:
        return f"[image: {url}]"
    return "[image]"


def extract_first_prompt(messages: Iterable[dict[str, Any]]) -> str:
    """Return the first user prompt's first line, clipped for filenames."""

    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = message_content_to_text(msg.get("content")).strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip()
        return first_line[:50].rstrip()
    return ""


def sanitize_export_filename(text: str) -> str:
    """Make a short, portable filename slug while preserving readable Unicode."""

    slug = text.strip().lower()
    for char in _WINDOWS_FORBIDDEN_CHARS:
        slug = slug.replace(char, "-")
    slug = re.sub(r"\s+", "-", slug, flags=re.UNICODE)
    slug = re.sub(r"-+", "-", slug).strip("-. ")
    slug = "".join(ch for ch in slug if ch.isprintable())
    if len(slug) > 60:
        slug = slug[:60].rstrip("-. ")
    if not slug or slug in _RESERVED_WINDOWS_NAMES:
        return "conversation"
    return slug


def resolve_export_path(
    filename: str = "",
    *,
    cwd: str | Path | None = None,
    messages: Iterable[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> Path:
    """Resolve the destination for a plaintext transcript export.

    Explicit filenames are resolved against ``cwd`` (matching Claude Code's
    command-line behavior).  A bare ``/export`` writes under Hermes home so it
    does not litter the current project workspace.
    """

    raw = (filename or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        return ensure_txt_suffix(path)

    stamp = (now or datetime.now()).strftime("%Y-%m-%d-%H%M%S")
    prompt = extract_first_prompt(messages or [])
    slug = sanitize_export_filename(prompt) if prompt else ""
    basename = f"{stamp}-{slug}.txt" if slug else f"conversation-{stamp}.txt"
    return get_hermes_home() / "sessions" / "exports" / basename


def ensure_txt_suffix(path: Path) -> Path:
    if path.name.endswith(".txt"):
        return path
    if path.suffix:
        return path.with_suffix(".txt")
    return path.with_name(f"{path.name}.txt")


def render_messages_plain_text(
    messages: Iterable[dict[str, Any]],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render messages as a readable plaintext transcript."""

    ts = (generated_at or datetime.now()).isoformat(timespec="seconds")
    lines = ["# Hermes conversation export", "", f"Generated: {ts}", ""]

    for index, msg in enumerate(messages, start=1):
        role = str(msg.get("role") or "message").lower()
        label = _ROLE_LABELS.get(role, role.title())
        name = msg.get("name")
        if name:
            label = f"{label}: {name}"

        lines.append(f"## {index}. {label}")

        content = message_content_to_text(msg.get("content")).strip()
        if content:
            lines.append(content)

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            if content:
                lines.append("")
            lines.extend(_render_tool_calls(tool_calls))

        if not content and not tool_calls:
            lines.append("(empty)")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_conversation_to_file(
    messages: Iterable[dict[str, Any]],
    filename: str = "",
    *,
    cwd: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write a plaintext transcript export and return the absolute path."""

    message_list = [dict(msg) for msg in messages]
    path = resolve_export_path(filename, cwd=cwd, messages=message_list, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_messages_plain_text(message_list, generated_at=now)
    path.write_text(rendered, encoding="utf-8")
    return path.resolve()


def _render_tool_calls(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, list):
        return [f"Tool calls: {_json_dump(tool_calls)}"]

    lines = ["Tool calls:"]
    for call in tool_calls:
        if not isinstance(call, dict):
            lines.append(f"- {call}")
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or call.get("id") or "tool"
        arguments = function.get("arguments") or call.get("arguments") or ""
        if not isinstance(arguments, str):
            arguments = _json_dump(arguments)
        arguments = arguments.strip()
        if len(arguments) > 500:
            arguments = f"{arguments[:500].rstrip()}…"
        suffix = f" {arguments}" if arguments else ""
        lines.append(f"- {name}{suffix}")
    return lines


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)
