"""Session-fork mixin for SessionDB: copy one session into a new, marked child.

``fork_session`` is the store-layer answer to "branch this conversation and keep the
original": the parent's live transcript, model route, working directory, resolved system
prompt and per-session state are copied into a NEW session id, and the child row is stamped
with the same branch marker every other branch surface writes
(``model_config._branched_from`` + ``parent_session_id`` — CLI ``/branch``, TUI/Desktop
``session.create``, gateway ``POST /api/sessions/{id}/fork``). The marker is load-bearing:

* ``is_explicit_fork_child`` / ``declared_scope_identity``: ``agent/prompt_cache_scope.py``
  refuses to publish the parent's declared conversation scope for a fork child, and the
  child's compression-lineage root is its own id — so the child's prompt cache starts in its
  own bucket instead of inheriting the parent's key (and never re-keys the parent's).
* ``get_compression_lineage`` returns just the child, so no consumer mistakes a fork for a
  continuation of the parent.
* Listing SQL (``_BRANCH_CHILD_SQL``) keeps marked children picker-visible.

The parent row is never written to: no ``ended_at``/``end_reason`` stamp, no transcript
rewrite. A caller that wants the "branch and move on" REPL behavior ends the parent itself
(CLI ``/branch`` already does).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from hermes_state_ids import new_session_id as _mint_session_id
from hermes_state_messages import _SET_COUNTERS_SQL, _tool_calls_len
from hermes_state_sessions import _parse_model_config

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# The branch marker every branch surface writes. It is only honored when it equals the row's
# ``parent_session_id`` — see ``SessionMessagesMixin._is_explicit_fork_child_row``.
FORK_MARKER_KEY = "_branched_from"

# Lineage markers describing the PARENT's own ancestry. Copied verbatim they would claim the
# child is a branch/delegate/reset of its grandparent (the marker rules bind the value to
# ``parent_session_id``), so they are stripped before the fork marker is written.
_FORK_STRIPPED_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")

# The child's ``sessions`` row. Gateway routing/origin columns (``session_key``, ``chat_id``,
# ``chat_type``, ``thread_id``, ``display_name``, ``origin_json``, ``user_id``) are
# deliberately absent: a branch child must never inherit a peer's routing identity — two live
# rows holding one routing key is the shape reported in #92859 — the same rule
# ``_insert_session_row`` applies to branch/delegate children.
_INSERT_FORK_CHILD_SQL = """INSERT INTO sessions (
    id, source, model, model_config, system_prompt, system_prompt_hash,
    parent_session_id, cwd, git_branch, git_repo_root, profile_name,
    transport_profile, started_at
) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)"""


class SessionForkMixin:
    """Fork a session into a new marked child (see module docstring)."""

    def fork_session(
        self,
        session_id: str,
        *,
        new_session_id: Optional[str] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        model: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        cwd: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy *session_id* into a new child session; returns a summary of the child.

        Returns ``{"session_id", "title", "source", "model", "parent_session_id",
        "message_count", "tool_call_count"}``.

        *title* None derives one in the parent's lineage (``"<parent title> #2"``); *source*,
        *model*, *cwd* and *profile_name* default to the parent's values. *model_config* is
        merged over the parent's per-session state (route keys such as ``provider``/
        ``base_url``/``api_mode`` and any other session kv included), and the fork marker is
        written on top — a caller cannot fork without the marker.

        Raises ValueError when the parent does not exist, *new_session_id* is taken or equals
        the parent's, or the derived/explicit *title* is invalid or already in use. The copy is
        one write transaction; a title write that fails AFTER the child row committed
        compensates by deleting the child (the ``_persist_branch`` pattern), so a failed call
        never leaves a partial or unnamed-but-usable child behind.

        Copy fidelity is row-level: the child's transcript is a byte-exact clone of the
        parent's ACTIVE rows in insertion order — content, ``tool_calls``/``tool_call_id``,
        timestamps, the ``api_content`` prompt-cache sidecar and reasoning sidecars all
        survive. Soft-archived (rewound/compacted-away) rows stay behind with the parent.
        """
        parent_id = str(session_id or "").strip()
        parent = self.get_session(parent_id) if parent_id else None
        if not parent:
            raise ValueError(f"No session {session_id!r}")

        child_id = str(new_session_id or "").strip() or _mint_session_id()
        if child_id == parent_id:
            raise ValueError("A fork must use a new session id, not the parent's")

        # Sanitized BEFORE the row lands: a too-long/blank title fails the call while the store
        # is still untouched.
        base_title = str(parent.get("title") or "").strip() or "fork"
        child_title = self.sanitize_title(
            title if title is not None else self.get_next_title_in_lineage(base_title))

        row = {
            "source": str(source or parent.get("source") or "cli"),
            "model": model if model is not None else parent.get("model"),
            "cwd": cwd if cwd is not None else parent.get("cwd"),
            "git_branch": parent.get("git_branch"),
            "git_repo_root": parent.get("git_repo_root"),
            "transport_profile": parent.get("transport_profile"),
            "profile_name": (profile_name if profile_name is not None else parent.get("profile_name")) or "",
            "system_prompt": parent.get("system_prompt"),
        }
        child_config = self._fork_child_state(parent.get("model_config"), model_config, parent_id)

        def _do(conn):
            # Re-checked inside the transaction: the parent may be gone and the child id may have
            # been taken since the reads above (the callback must stay idempotent under retry).
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (parent_id,)).fetchone() is None:
                raise ValueError(f"No session {parent_id!r}")
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (child_id,)).fetchone() is not None:
                raise ValueError(f"Session {child_id!r} already exists")
            transcript = conn.execute(
                "SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
                (parent_id,)).fetchall()
            message_ids = [int(row_["id"]) for row_ in transcript]
            tool_call_count = sum(_tool_calls_len(row_["tool_calls"], scalar=1) for row_ in transcript)
            conn.execute(_INSERT_FORK_CHILD_SQL, (
                child_id, row["source"], row["model"],
                json.dumps(child_config) if child_config else None,
                self._store_system_prompt(conn, row["system_prompt"]),
                parent_id, row["cwd"], row["git_branch"], row["git_repo_root"],
                row["profile_name"] or self._own_profile_name(), row["transport_profile"],
                time.time()))
            if message_ids:
                # Byte-exact row clone — the same primitive compression uses to continue a
                # conversation across ids: every payload column survives verbatim and the FTS
                # triggers index the child's copies.
                self._clone_message_rows(conn, message_ids, session_id=child_id)
            conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?",
                         (len(message_ids), tool_call_count, child_id))
            return len(message_ids), tool_call_count

        message_count, tool_call_count = self._execute_write(
            _do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

        if child_title:
            try:
                self.set_session_title(child_id, child_title)
            except ValueError:
                # The guarded setter refused (a title conflict): roll the child back so the
                # caller sees one atomic outcome, not a fork whose name never landed.
                self.delete_session(child_id)
                raise

        logger.info("Forked session %s -> %s (%d messages)", parent_id, child_id, message_count)
        return {
            "session_id": child_id,
            "title": child_title,
            "source": row["source"],
            "model": row["model"],
            "parent_session_id": parent_id,
            "message_count": message_count,
            "tool_call_count": tool_call_count,
        }

    @staticmethod
    def _fork_child_state(
        parent_raw: Any, override: Optional[Dict[str, Any]], parent_id: str,
    ) -> Dict[str, Any]:
        """The child's per-session state: parent ``model_config`` minus the parent's own lineage
        markers, plus *override*, plus the fork marker (written last, so it cannot be omitted)."""
        state = _parse_model_config(parent_raw)
        for marker in _FORK_STRIPPED_MARKERS:
            state.pop(marker, None)
        if override:
            state.update(override)
        state[FORK_MARKER_KEY] = parent_id
        return state
