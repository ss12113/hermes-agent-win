"""Editor-mode helpers for the Claude-compatible /vim command."""

from __future__ import annotations

from typing import Any

EDITOR_MODE_NORMAL = "normal"
EDITOR_MODE_VIM = "vim"
_VALID_EDITOR_MODES = {EDITOR_MODE_NORMAL, EDITOR_MODE_VIM}
_NORMAL_ALIASES = {"", "normal", "emacs", "readline", "default", "off", "false", "0"}
_VIM_ALIASES = {"vim", "vi", "on", "true", "1"}


VIM_USAGE = """Usage: /vim [on|off|status]

Toggles prompt editing mode for the classic Hermes CLI.
- vim:    prompt_toolkit Vim bindings; Escape toggles INSERT/NORMAL modes.
- normal: standard readline/Emacs-style keyboard bindings.

The choice is saved to display.editor_mode in config.yaml and applies to new CLI sessions too.
""".strip()


def normalize_editor_mode(value: Any) -> str:
    """Return Hermes' canonical editor mode (``normal`` or ``vim``)."""
    raw = str(value or "").strip().lower()
    if raw in _VIM_ALIASES:
        return EDITOR_MODE_VIM
    return EDITOR_MODE_NORMAL


def editor_mode_from_command_arg(raw_args: str | None, current_mode: Any) -> tuple[str, str]:
    """Resolve a /vim command argument to (action, mode).

    action is one of ``set``, ``status``, ``help``, or ``error``.  Bare /vim
    mirrors Claude Code and toggles between normal and vim mode.
    """
    arg = (raw_args or "").strip().lower()
    current = normalize_editor_mode(current_mode)
    if not arg:
        return "set", EDITOR_MODE_NORMAL if current == EDITOR_MODE_VIM else EDITOR_MODE_VIM
    first = arg.split(maxsplit=1)[0]
    if first in {"status", "current"}:
        return "status", current
    if first in {"help", "?", "-h", "--help"}:
        return "help", current
    if first in _VIM_ALIASES:
        return "set", EDITOR_MODE_VIM
    if first in _NORMAL_ALIASES:
        return "set", EDITOR_MODE_NORMAL
    return "error", current


def editor_mode_message(mode: str) -> str:
    """Claude-style success message for a saved editor mode."""
    normalized = normalize_editor_mode(mode)
    if normalized == EDITOR_MODE_VIM:
        return "Editor mode set to vim. Use Escape key to toggle between INSERT and NORMAL modes."
    return "Editor mode set to normal. Using standard (readline) keyboard bindings."


def editor_mode_status(mode: str) -> str:
    """Human-readable current editor mode summary."""
    normalized = normalize_editor_mode(mode)
    if normalized == EDITOR_MODE_VIM:
        return "Editor mode: vim — Escape toggles INSERT/NORMAL modes."
    return "Editor mode: normal — standard readline keyboard bindings."


def prompt_toolkit_editing_mode(mode: str):
    """Return prompt_toolkit's EditingMode enum for a Hermes editor mode."""
    from prompt_toolkit.enums import EditingMode

    return EditingMode.VI if normalize_editor_mode(mode) == EDITOR_MODE_VIM else EditingMode.EMACS
