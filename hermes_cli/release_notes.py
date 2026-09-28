"""Informational /release-notes command output for Hermes."""

from __future__ import annotations

from hermes_cli import __release_date__, __version__

HERMES_RELEASE_NOTES_URL = "https://github.com/NousResearch/hermes-agent/releases/latest"
HERMES_RELEASES_URL = "https://github.com/NousResearch/hermes-agent/releases"
HERMES_UPDATING_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/getting-started/updating"


def current_version_label() -> str:
    """Return the installed Hermes version label without doing network I/O."""

    return f"Hermes Agent v{__version__} ({__release_date__})"


def format_release_notes_info(version_label: str | None = None) -> str:
    """Return a network-free /release-notes compatibility message.

    Claude Code's /release-notes command renders a bundled/fetched changelog.
    Hermes does not ship a local changelog bundle, so the compatibility command
    points users at the official release-note locations and includes the local
    installed version for context.
    """

    label = version_label or current_version_label()
    return "\n".join(
        [
            "Hermes release notes",
            "",
            f"Current version: {label}",
            f"Latest release notes: {HERMES_RELEASE_NOTES_URL}",
            f"All releases: {HERMES_RELEASES_URL}",
            f"Update guide: {HERMES_UPDATING_DOCS_URL}",
            "",
            "Use `/version` for installed version details and `/update` to update Hermes.",
        ]
    )
