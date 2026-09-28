from agent.privacy_summary import collect_privacy_settings, format_privacy_summary


def test_collect_privacy_settings_defaults_secure():
    settings = collect_privacy_settings({}, live_redact_enabled=True, redact_env_value=None)

    assert settings["security.redact_secrets"] is True
    assert settings["live.secret_redaction"] is True
    assert settings["env.HERMES_REDACT_SECRETS.set"] is False
    assert settings["env.HERMES_REDACT_SECRETS.enabled"] is True
    assert settings["privacy.redact_pii"] is False
    assert settings["security.allow_private_urls"] is False
    assert settings["browser.allow_private_urls"] is False
    assert settings["browser.record_sessions"] is False


def test_collect_privacy_settings_normalizes_string_toggles():
    settings = collect_privacy_settings(
        {
            "security": {
                "redact_secrets": "off",
                "allow_private_urls": "yes",
            },
            "privacy": {"redact_pii": "1"},
            "browser": {
                "allow_private_urls": "true",
                "record_sessions": "disabled",
            },
        },
        live_redact_enabled=False,
        redact_env_value="0",
    )

    assert settings["security.redact_secrets"] is False
    assert settings["live.secret_redaction"] is False
    assert settings["env.HERMES_REDACT_SECRETS.set"] is True
    assert settings["env.HERMES_REDACT_SECRETS.enabled"] is False
    assert settings["privacy.redact_pii"] is True
    assert settings["security.allow_private_urls"] is True
    assert settings["browser.allow_private_urls"] is True
    assert settings["browser.record_sessions"] is False


def test_format_privacy_summary_shows_config_values():
    out = format_privacy_summary(
        {
            "security": {
                "redact_secrets": False,
                "allow_private_urls": True,
            },
            "privacy": {"redact_pii": True},
            "browser": {
                "allow_private_urls": True,
                "record_sessions": True,
            },
        },
        live_redact_enabled=False,
        redact_env_value="off",
    )

    assert "Privacy and redaction settings" in out
    assert "security.redact_secrets): disabled" in out
    assert "live import snapshot): disabled" in out
    assert "HERMES_REDACT_SECRETS): set → disabled" in out
    assert "privacy.redact_pii): enabled" in out
    assert "security.allow_private_urls): allowed" in out
    assert "browser.allow_private_urls): allowed" in out
    assert "browser.record_sessions): enabled" in out
    assert "Slack/Discord mention IDs may be preserved" in out


def test_format_privacy_summary_warns_when_redaction_snapshot_differs():
    out = format_privacy_summary(
        {"security": {"redact_secrets": False}},
        live_redact_enabled=True,
    )

    assert "security.redact_secrets): disabled" in out
    assert "live import snapshot): enabled" in out
    assert "Restart Hermes for the config change" in out


def test_format_privacy_summary_mentions_claude_style_aliases():
    out = format_privacy_summary({}, live_redact_enabled=True, redact_env_value=None)

    assert "/privacy-settings" in out
    assert "/privacy_settings" in out


def test_format_privacy_summary_never_prints_raw_env_value():
    out = format_privacy_summary(
        {},
        live_redact_enabled=True,
        redact_env_value="unexpected-sensitive-looking-value",
    )

    assert "HERMES_REDACT_SECRETS): set → disabled" in out
    assert "unexpected-sensitive-looking-value" not in out


def test_collect_privacy_settings_env_signal_matches_import_parser():
    def enabled(value: str) -> bool:
        return collect_privacy_settings({}, redact_env_value=value)[
            "env.HERMES_REDACT_SECRETS.enabled"
        ]

    assert enabled("yes") is True
    assert enabled("on") is True
    assert enabled("true") is True
    assert enabled("1") is True
    assert enabled("") is False
    assert enabled("maybe") is False
