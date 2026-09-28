"""Tests for Claude-compatible quota guidance commands."""

from hermes_cli.rate_limit_options import (
    format_extra_usage_info,
    format_rate_limit_options_info,
)


def test_rate_limit_options_is_offline_guidance():
    text = format_rate_limit_options_info(
        config={"model": {"provider": "nous", "model": "hermes-test"}},
        provider=None,
        model=None,
    )

    assert "Rate limit options" in text
    assert "Provider: nous" in text
    assert "Model: hermes-test" in text
    assert "`/usage`" in text
    assert "informational only" in text


def test_rate_limit_options_redacts_secret_shaped_route_values():
    text = format_rate_limit_options_info(
        config={},
        provider="Bearer should-not-display",
        model="plain-model",
    )

    assert "[REDACTED]" in text
    assert "should-not-display" not in text
    assert "plain-model" in text


def test_extra_usage_is_offline_guidance():
    text = format_extra_usage_info(
        config={"model": {"provider": "openrouter", "model": "claude-compatible"}},
        provider=None,
        model=None,
    )

    assert "Extra usage options" in text
    assert "billing/admin extra-usage flow" in text
    assert "Provider: openrouter" in text
    assert "Model: claude-compatible" in text
    assert "admin requests" in text
    assert "informational only" in text
