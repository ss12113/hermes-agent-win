"""Tests for Claude-compatible /release-notes guidance."""

from hermes_cli.release_notes import format_release_notes_info


def test_release_notes_includes_version_and_official_links():
    text = format_release_notes_info("Hermes Agent v0.test (2099.1.1)")

    assert "Hermes release notes" in text
    assert "Current version: Hermes Agent v0.test (2099.1.1)" in text
    assert "https://github.com/NousResearch/hermes-agent/releases/latest" in text
    assert "https://github.com/NousResearch/hermes-agent/releases" in text
    assert "https://hermes-agent.nousresearch.com/docs/getting-started/updating" in text


def test_release_notes_is_network_free_guidance():
    text = format_release_notes_info("local-version")

    assert "Current version: local-version" in text
    assert "Use `/version`" in text
    assert "`/update`" in text
