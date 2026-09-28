"""Tests for prompt editor-mode helpers and /vim parity behavior."""

from prompt_toolkit.enums import EditingMode

from hermes_cli.editor_mode import (
    EDITOR_MODE_NORMAL,
    EDITOR_MODE_VIM,
    editor_mode_from_command_arg,
    editor_mode_message,
    editor_mode_status,
    normalize_editor_mode,
    prompt_toolkit_editing_mode,
)


def test_normalize_editor_mode_accepts_vim_and_legacy_normal_aliases():
    assert normalize_editor_mode("vim") == EDITOR_MODE_VIM
    assert normalize_editor_mode("vi") == EDITOR_MODE_VIM
    assert normalize_editor_mode("on") == EDITOR_MODE_VIM
    assert normalize_editor_mode("normal") == EDITOR_MODE_NORMAL
    assert normalize_editor_mode("emacs") == EDITOR_MODE_NORMAL
    assert normalize_editor_mode(None) == EDITOR_MODE_NORMAL


def test_bare_vim_command_toggles_between_normal_and_vim():
    assert editor_mode_from_command_arg("", "normal") == ("set", "vim")
    assert editor_mode_from_command_arg(None, "vim") == ("set", "normal")


def test_vim_command_args_support_status_help_and_explicit_modes():
    assert editor_mode_from_command_arg("status", "vim") == ("status", "vim")
    assert editor_mode_from_command_arg("--help", "normal") == ("help", "normal")
    assert editor_mode_from_command_arg("off", "vim") == ("set", "normal")
    assert editor_mode_from_command_arg("normal", "vim") == ("set", "normal")
    assert editor_mode_from_command_arg("on", "normal") == ("set", "vim")
    assert editor_mode_from_command_arg("nonsense", "vim") == ("error", "vim")


def test_prompt_toolkit_mode_mapping():
    assert prompt_toolkit_editing_mode("vim") is EditingMode.VI
    assert prompt_toolkit_editing_mode("normal") is EditingMode.EMACS


def test_editor_mode_user_messages_match_claude_style():
    assert "Editor mode set to vim" in editor_mode_message("vim")
    assert "Escape key" in editor_mode_message("vim")
    assert "Editor mode set to normal" in editor_mode_message("normal")
    assert "readline" in editor_mode_message("normal")
    assert "Editor mode: vim" in editor_mode_status("vim")
    assert "Editor mode: normal" in editor_mode_status("normal")
