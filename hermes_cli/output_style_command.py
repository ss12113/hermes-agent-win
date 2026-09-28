"""Claude Code /output-style compatibility shim."""

from __future__ import annotations

OUTPUT_STYLE_DEPRECATED_MESSAGE = (
    "/output-style has been deprecated. Use /config to change your output style, "
    "or set it in your settings file. Changes take effect on the next session."
)


def format_output_style_deprecation() -> str:
    """Return the user-facing deprecation message for /output-style."""

    return OUTPUT_STYLE_DEPRECATED_MESSAGE
