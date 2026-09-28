"""Best-effort cleanup for session-boundary, current-context caches.

These registries are populated by runtime skill loading so child processes can
see skill-declared environment variables or credential files. They are not part
of durable conversation state, so a fresh/resumed/branched session should not
silently inherit entries from the previous live conversation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def clear_session_boundary_caches() -> None:
    """Clear current-context skill/runtime caches at a conversation boundary.

    The helper is deliberately fail-open: session switching/resetting must not
    fail because optional runtime cache modules are unavailable or raise while
    being cleared.
    """

    try:
        from tools.env_passthrough import clear_env_passthrough

        clear_env_passthrough()
    except Exception as exc:
        logger.debug(
            "Failed to clear env passthrough at session boundary: %s",
            exc,
            exc_info=True,
        )

    try:
        from tools.credential_files import clear_credential_files

        clear_credential_files()
    except Exception as exc:
        logger.debug(
            "Failed to clear credential files at session boundary: %s",
            exc,
            exc_info=True,
        )
