from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from hermes_cli.effort_command import (
    display_effort_value,
    format_effort_status,
    parse_effort_command_arg,
    reasoning_value_for_effort,
)
from hermes_cli.keybindings_summary import (
    format_keybindings_summary,
    should_preserve_ctrl_enter_newline,
)


def test_effort_parser_preserves_distinct_max_wire_value():
    result = parse_effort_command_arg("max")

    assert result.action == "set"
    assert result.reasoning_value == "max"
    assert result.display_value == "max"
    assert "Set effort level to max" in result.message


def test_effort_parser_auto_clears_reasoning_value():
    result = parse_effort_command_arg("auto")

    assert result.action == "clear"
    assert result.reasoning_value == ""
    assert result.display_value == "auto"


def test_effort_parser_status_and_invalid():
    assert parse_effort_command_arg("").action == "status"
    invalid = parse_effort_command_arg("turbo")
    assert invalid.action == "error"
    assert "Valid options" in invalid.message


def test_effort_value_display_and_reasoning_mapping_keeps_xhigh_and_max_distinct():
    assert reasoning_value_for_effort("max") == "max"
    assert reasoning_value_for_effort("xhigh") == "xhigh"
    assert reasoning_value_for_effort("auto") == ""
    assert reasoning_value_for_effort("medium") == "medium"
    assert display_effort_value("xhigh") == "xhigh"
    assert display_effort_value("max") == "max"


def test_effort_status_formats_auto_xhigh_and_max_distinctly():
    assert "auto" in format_effort_status(None)
    assert "Current effort level: xhigh" in format_effort_status({"enabled": True, "effort": "xhigh"})
    assert "Current effort level: max" in format_effort_status({"enabled": True, "effort": "max"})


def test_keybindings_predicate_matches_cli_ctrl_enter_environments():
    assert should_preserve_ctrl_enter_newline({}, platform="win32", probe_wsl=False) is True
    assert should_preserve_ctrl_enter_newline({"SSH_TTY": "/dev/pts/0"}, platform="linux", probe_wsl=False) is True
    assert should_preserve_ctrl_enter_newline({"WT_SESSION": "abc"}, platform="linux", probe_wsl=False) is True
    assert should_preserve_ctrl_enter_newline({}, platform="linux", probe_wsl=False) is False


def test_keybindings_predicate_probes_wsl_proc_marker():
    def _fake_open(path, *args, **kwargs):
        if "/proc/version" in str(path):
            return StringIO("Linux version 5.15.167.4-microsoft-standard-WSL2")
        raise OSError("missing")

    with patch("builtins.open", side_effect=_fake_open):
        assert should_preserve_ctrl_enter_newline({}, platform="linux", probe_wsl=True) is True


def test_keybindings_summary_includes_terminal_shortcuts():
    summary = format_keybindings_summary({"WT_SESSION": "1"}, platform="linux", probe_wsl=False)

    assert summary.splitlines()[0] == "Keyboard shortcuts"
    assert "Enter — submit" in summary
    assert "Ctrl+Enter / Ctrl+J" in summary
    assert "read-only status view" in summary
