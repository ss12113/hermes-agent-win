"""Hermes terminal keybinding summary helpers.

This module is intentionally UI-neutral: the classic CLI uses the predicate to
bind prompt_toolkit keys, while slash-command surfaces can render the same facts
as a read-only help summary.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


_WSL_PROC_PATHS = ("/proc/version", "/proc/sys/kernel/osrelease")


def _env_get(environ: Mapping[str, str] | None, name: str) -> str:
    if environ is None:
        return os.environ.get(name, "")
    value = environ.get(name, "")
    return "" if value is None else str(value)


def should_preserve_ctrl_enter_newline(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    *,
    probe_wsl: bool = True,
) -> bool:
    """Return True when ``Ctrl+Enter``/``Ctrl+J`` should insert a newline.

    Windows Terminal, WSL, SSH sessions, Ghostty, and some modern terminals
    deliver Ctrl+Enter/Ctrl+J as bare LF (``c-j``). In those environments the
    classic CLI must leave ``c-j`` available for newline insertion instead of
    binding it to submit. Plain local POSIX PTYs still bind ``c-j`` to submit so
    thin terminal backends that send LF for Enter remain usable.
    """

    platform_name = sys.platform if platform is None else platform
    if platform_name == "win32":
        return True

    if any(_env_get(environ, v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return True
    if _env_get(environ, "WT_SESSION"):
        return True
    if _env_get(environ, "GHOSTTY_RESOURCES_DIR") or _env_get(environ, "GHOSTTY_BIN_DIR"):
        return True
    if _env_get(environ, "TERM").lower() == "xterm-ghostty":
        return True
    if _env_get(environ, "TERM_PROGRAM").lower() == "ghostty":
        return True
    if "microsoft" in _env_get(environ, "WSL_DISTRO_NAME").lower():
        return True

    if probe_wsl:
        for path in _WSL_PROC_PATHS:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    if "microsoft" in f.read().lower():
                        return True
            except OSError:
                continue
    return False


def format_keybindings_summary(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    *,
    probe_wsl: bool = True,
) -> str:
    """Render a concise, read-only summary of Hermes terminal shortcuts."""

    preserve_ctrl_enter = should_preserve_ctrl_enter_newline(
        environ=environ,
        platform=platform,
        probe_wsl=probe_wsl,
    )
    ctrl_enter_line = (
        "  Ctrl+Enter / Ctrl+J — insert a newline in this terminal profile"
        if preserve_ctrl_enter
        else "  Ctrl+J — submit fallback on plain POSIX PTYs; use Alt+Enter for newline"
    )

    lines = [
        "Keyboard shortcuts",
        "  Enter — submit the current prompt",
        "  Alt+Enter — insert a newline when your terminal forwards it",
        ctrl_enter_line,
        "  Ctrl+G / Alt+G — open the current prompt in $EDITOR",
        "  Tab — complete slash commands, paths, skills, and command arguments",
        "  Up/Down — navigate history when the cursor is at the prompt edge",
        "  Ctrl+C — interrupt the active run or cancel the current prompt/dialog",
        "  Ctrl+Q — force-quit the local CLI/TUI prompt",
        "  Ctrl+L — redraw/repaint the terminal UI",
        "",
        "This is a read-only status view. To change terminal behavior, adjust your",
        "terminal/IDE keybindings or Hermes display settings rather than editing",
        "conversation context.",
    ]
    return "\n".join(lines)
