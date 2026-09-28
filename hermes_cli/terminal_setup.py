"""Claude-style terminal setup helpers for Hermes CLI.

The TUI has a TypeScript implementation for ``/terminal-setup``.  This module
keeps the classic Python CLI on the same command surface by installing the
same VS Code-family keybindings when the user explicitly runs the command.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_TERMINALS = ("vscode", "cursor", "windsurf")
TERMINAL_SETUP_USAGE = "Usage: /terminal-setup [auto|vscode|cursor|windsurf]"

_COPY_SEQUENCE = "\u001b[99;13u"
_MULTILINE_SEQUENCE = "\\\r\n"

_TERMINAL_META = {
    "vscode": {"app_name": "Code", "label": "VS Code"},
    "cursor": {"app_name": "Cursor", "label": "Cursor"},
    "windsurf": {"app_name": "Windsurf", "label": "Windsurf"},
}


@dataclass(frozen=True)
class TerminalSetupResult:
    """Result returned by ``configure_terminal_keybindings``."""

    success: bool
    message: str
    requires_restart: bool = False


def detect_vscode_like_terminal(env: Mapping[str, str] | None = None) -> str | None:
    """Detect a VS Code-family integrated terminal from environment variables."""

    env = env or os.environ
    askpass = (env.get("VSCODE_GIT_ASKPASS_MAIN") or "").lower()

    if env.get("CURSOR_TRACE_ID") or "cursor" in askpass:
        return "cursor"
    if "windsurf" in askpass:
        return "windsurf"
    if env.get("TERM_PROGRAM") == "vscode" or env.get("VSCODE_GIT_IPC_HANDLE"):
        return "vscode"
    return None


def is_remote_shell_session(env: Mapping[str, str] | None = None) -> bool:
    """Return True when setup is running inside SSH / remote IDE server shells."""

    env = env or os.environ
    askpass = (env.get("VSCODE_GIT_ASKPASS_MAIN") or "").lower()
    path = (env.get("PATH") or "").lower()

    return bool(
        env.get("SSH_CONNECTION")
        or env.get("SSH_TTY")
        or env.get("SSH_CLIENT")
        or ".vscode-server" in askpass
        or ".cursor-server" in askpass
        or ".windsurf-server" in askpass
        or ".vscode-server" in path
        or ".cursor-server" in path
        or ".windsurf-server" in path
    )


def vscode_style_config_dir(
    app_name: str,
    *,
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    home_dir: str | Path | None = None,
) -> Path | None:
    """Return the User config directory for a VS Code-family app."""

    platform_name = platform_name or sys.platform
    env = env or os.environ
    home = Path(home_dir) if home_dir is not None else Path.home()

    if platform_name == "darwin":
        return home / "Library" / "Application Support" / app_name / "User"
    if platform_name.startswith("win"):
        appdata = env.get("APPDATA")
        return Path(appdata) / app_name / "User" if appdata else None
    return home / ".config" / app_name / "User"


def _strip_json_comments(content: str) -> str:
    """Strip JSONC comments and trailing commas without touching strings."""

    result: list[str] = []
    i = 0
    length = len(content)
    while i < length:
        ch = content[i]
        if ch == '"':
            start = i
            i += 1
            while i < length:
                if content[i] == "\\":
                    i += 2
                    continue
                if content[i] == '"':
                    i += 1
                    break
                i += 1
            result.append(content[start:i])
            continue
        if ch == "/" and i + 1 < length and content[i + 1] == "/":
            newline = content.find("\n", i)
            i = length if newline == -1 else newline
            continue
        if ch == "/" and i + 1 < length and content[i + 1] == "*":
            end = content.find("*/", i + 2)
            i = length if end == -1 else end + 2
            continue
        result.append(ch)
        i += 1

    return re.sub(r",(\s*[}\]])", r"\1", "".join(result))


def _target_bindings(platform_name: str) -> list[dict]:
    base = [
        {
            "key": "shift+enter",
            "command": "workbench.action.terminal.sendSequence",
            "when": "terminalFocus",
            "args": {"text": _MULTILINE_SEQUENCE},
        },
        {
            "key": "ctrl+enter",
            "command": "workbench.action.terminal.sendSequence",
            "when": "terminalFocus",
            "args": {"text": _MULTILINE_SEQUENCE},
        },
        {
            "key": "cmd+enter",
            "command": "workbench.action.terminal.sendSequence",
            "when": "terminalFocus",
            "args": {"text": _MULTILINE_SEQUENCE},
        },
        {
            "key": "cmd+z",
            "command": "workbench.action.terminal.sendSequence",
            "when": "terminalFocus",
            "args": {"text": "\u001b[122;9u"},
        },
        {
            "key": "shift+cmd+z",
            "command": "workbench.action.terminal.sendSequence",
            "when": "terminalFocus",
            "args": {"text": "\u001b[122;10u"},
        },
    ]
    if platform_name == "darwin":
        return [
            {
                "key": "cmd+c",
                "command": "workbench.action.terminal.sendSequence",
                "when": "terminalFocus && terminalTextSelected",
                "args": {"text": _COPY_SEQUENCE},
            },
            *base,
        ]
    return base


def _same_binding(left: Mapping, right: Mapping) -> bool:
    return (
        left.get("key") == right.get("key")
        and left.get("command") == right.get("command")
        and left.get("when") == right.get("when")
        and (left.get("args") or {}).get("text") == (right.get("args") or {}).get("text")
    )


def _merge_keybindings(existing: Iterable[object], target: Iterable[dict]) -> tuple[list[object], int]:
    merged = list(existing)
    added = 0
    for binding in target:
        if any(isinstance(item, Mapping) and _same_binding(item, binding) for item in merged):
            continue
        merged.append(binding)
        added += 1
    return merged, added


def _manual_remote_message(label: str) -> str:
    return (
        f"Cannot install {label} keybindings from a remote shell.\n\n"
        "Open the local app, then run /terminal-setup there, or add these "
        "terminal keybindings locally: Shift+Enter and Ctrl+Enter should send "
        "a backslash followed by Return (\\\\r\\n)."
    )


def configure_terminal_keybindings(
    target: str = "auto",
    *,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    home_dir: str | Path | None = None,
) -> TerminalSetupResult:
    """Install Hermes multiline keybindings for a VS Code-family terminal."""

    raw_target = (target or "auto").strip().lower()
    if raw_target not in {"auto", *SUPPORTED_TERMINALS}:
        return TerminalSetupResult(False, TERMINAL_SETUP_USAGE)

    env = env or os.environ
    platform_name = platform_name or sys.platform
    resolved = detect_vscode_like_terminal(env) if raw_target == "auto" else raw_target
    if resolved is None:
        return TerminalSetupResult(
            False,
            (
                "No supported VS Code-family terminal detected.\n\n"
                f"{TERMINAL_SETUP_USAGE}\n"
                "Supported targets: vscode, cursor, windsurf."
            ),
        )

    meta = _TERMINAL_META[resolved]
    label = meta["label"]
    if is_remote_shell_session(env):
        return TerminalSetupResult(False, _manual_remote_message(label))

    config_dir = vscode_style_config_dir(
        meta["app_name"],
        platform_name=platform_name,
        env=env,
        home_dir=home_dir,
    )
    if config_dir is None:
        return TerminalSetupResult(False, f"Could not locate {label} User config directory.")

    keybindings_path = config_dir / "keybindings.json"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        if keybindings_path.exists():
            raw = keybindings_path.read_text(encoding="utf-8")
            parsed = json.loads(_strip_json_comments(raw) or "[]")
            if not isinstance(parsed, list):
                return TerminalSetupResult(False, f"{keybindings_path} is not a JSON array; leaving it unchanged.")
        else:
            parsed = []

        merged, added = _merge_keybindings(parsed, _target_bindings(platform_name))
        if added:
            keybindings_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return TerminalSetupResult(
                True,
                f"Installed {added} Hermes terminal keybinding(s) in {keybindings_path}.",
                requires_restart=True,
            )
        return TerminalSetupResult(True, f"Hermes terminal keybindings already installed in {keybindings_path}.")
    except json.JSONDecodeError as exc:
        return TerminalSetupResult(False, f"Could not parse {keybindings_path}: {exc}. Leaving it unchanged.")
    except OSError as exc:
        return TerminalSetupResult(False, f"Could not update {keybindings_path}: {exc}")
