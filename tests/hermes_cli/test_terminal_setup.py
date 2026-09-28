from __future__ import annotations

import json

from hermes_cli.terminal_setup import (
    configure_terminal_keybindings,
    detect_vscode_like_terminal,
)


def test_detects_vscode_like_terminal_from_env():
    assert detect_vscode_like_terminal({"TERM_PROGRAM": "vscode"}) == "vscode"
    assert detect_vscode_like_terminal({"CURSOR_TRACE_ID": "abc"}) == "cursor"
    assert detect_vscode_like_terminal({"VSCODE_GIT_ASKPASS_MAIN": "/tmp/Windsurf/git.js"}) == "windsurf"


def test_terminal_setup_installs_keybindings_idempotently(tmp_path):
    result = configure_terminal_keybindings(
        "vscode",
        env={},
        platform_name="linux",
        home_dir=tmp_path,
    )

    assert result.success is True
    assert result.requires_restart is True
    keybindings_path = tmp_path / ".config" / "Code" / "User" / "keybindings.json"
    bindings = json.loads(keybindings_path.read_text(encoding="utf-8"))
    assert any(b["key"] == "shift+enter" for b in bindings)
    assert any(b["key"] == "ctrl+enter" for b in bindings)

    second = configure_terminal_keybindings(
        "vscode",
        env={},
        platform_name="linux",
        home_dir=tmp_path,
    )

    assert second.success is True
    assert second.requires_restart is False
    assert "already installed" in second.message
    assert json.loads(keybindings_path.read_text(encoding="utf-8")) == bindings


def test_terminal_setup_auto_requires_detected_terminal(tmp_path):
    result = configure_terminal_keybindings(
        "auto",
        env={},
        platform_name="linux",
        home_dir=tmp_path,
    )

    assert result.success is False
    assert "No supported VS Code-family terminal detected" in result.message


def test_terminal_setup_refuses_remote_shell(tmp_path):
    result = configure_terminal_keybindings(
        "cursor",
        env={"SSH_CONNECTION": "1 2 3 4"},
        platform_name="linux",
        home_dir=tmp_path,
    )

    assert result.success is False
    assert "remote shell" in result.message
