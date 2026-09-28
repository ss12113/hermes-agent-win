"""Tests for Claude-style permissions/sandbox summaries."""

from hermes_cli.permission_summary import (
    format_permissions_summary,
    format_sandbox_summary,
    permission_state,
    sandbox_state,
)


def test_permission_state_normalizes_config_and_yolo_sources():
    cfg = {
        "approvals": {
            "mode": "smart",
            "timeout": "42",
            "cron_mode": "approve",
            "command_allowlist": ["pytest *", "npm test"],
            "permanent_approved": {"recursive delete": True},
        },
    }

    state = permission_state(cfg, process_yolo=False, session_yolo=True)

    assert state["approval_mode"] == "smart"
    assert state["timeout_seconds"] == 42
    assert state["cron_mode"] == "approve"
    assert state["command_allowlist_count"] == 2
    assert state["permanent_approval_count"] == 1
    assert state["effective_bypass"] is True


def test_permissions_summary_is_redaction_safe_and_mentions_alias():
    cfg = {
        "approvals": {
            "mode": "off",
            "command_allowlist": ["sensitive command text"],
        }
    }

    text = format_permissions_summary(cfg, process_yolo=False, session_yolo=False)

    assert "## Permissions" in text
    assert "`off`" in text
    assert "Command allowlist entries: `1`" in text
    assert "/allowed-tools" in text
    assert "sensitive command text" not in text


def test_sandbox_state_maps_terminal_backend_to_isolation(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    local = sandbox_state({"terminal": {"backend": "local"}}, process_yolo=False, session_yolo=False)
    docker = sandbox_state({"terminal": {"backend": "docker"}}, process_yolo=False, session_yolo=False)
    ssh = sandbox_state({"terminal": {"backend": "ssh"}}, process_yolo=False, session_yolo=False)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    env_modal = sandbox_state({"terminal": {"backend": "local"}}, process_yolo=False, session_yolo=False)

    assert local["sandboxed_backend"] is False
    assert docker["sandboxed_backend"] is True
    assert ssh["sandboxed_backend"] is False
    assert env_modal["terminal_backend"] == "modal"
    assert env_modal["terminal_backend_source"] == "TERMINAL_ENV"
    assert env_modal["sandboxed_backend"] is True


def test_sandbox_summary_mentions_backend_and_permission_posture(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    text = format_sandbox_summary(
        {"terminal": {"backend": "docker"}, "approvals": {"mode": "manual"}},
        process_yolo=False,
        session_yolo=False,
    )

    assert "## Sandbox" in text
    assert "Terminal backend: `docker`" in text
    assert "Terminal backend source: `config.yaml terminal.backend`" in text
    assert "Effective terminal sandbox: `on`" in text
    assert "Dangerous command approval mode: `manual`" in text


def test_sandbox_summary_supports_help_and_status_aliases(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    help_text = format_sandbox_summary(arg="help")
    status_text = format_sandbox_summary(
        {"terminal": {"backend": "docker"}, "approvals": {"mode": "manual"}},
        arg="current",
        process_yolo=False,
        session_yolo=False,
    )

    assert "Usage:" in help_text
    assert "/sandbox-toggle" in help_text
    assert "Terminal backend: `docker`" in status_text


def test_sandbox_exclude_is_read_only_and_does_not_echo_secret_like_pattern():
    text = format_sandbox_summary(arg='exclude "npm run test:* TOKEN=secret-value"')

    assert "Claude-style sandbox exclusion parsed" in text
    assert "No Hermes config was changed" in text
    assert "command_allowlist" in text
    assert "npm run test" not in text
    assert "secret-value" not in text


def test_sandbox_exclude_missing_and_unknown_subcommands_return_usage_guidance():
    missing = format_sandbox_summary(arg="exclude")
    unknown = format_sandbox_summary(arg="toggle-on")

    assert "requires a command pattern" in missing
    assert "unknown `/sandbox` subcommand" in unknown
