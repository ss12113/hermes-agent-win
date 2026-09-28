"""Invariants for the pre_compaction / post_compaction compression-lifecycle hooks.

The observer pair must be part of the public hook registry with `hermes hooks test`
payloads matching the live call sites, and post_compaction must fire only when a
compaction actually commits.
"""
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.plugins import VALID_HOOKS


def _agent():
    return SimpleNamespace(session_id="sess-1", model="test-model", platform="cli")


def test_compaction_hooks_are_wired_into_the_hook_registry():
    """Both events are valid hook names and ship `hooks test` payloads for the pre contract."""
    from hermes_cli.hooks import _DEFAULT_PAYLOADS

    assert "pre_compaction" in VALID_HOOKS
    assert "post_compaction" in VALID_HOOKS
    assert {"session_id", "model", "platform", "trigger", "message_count",
            "approx_tokens", "in_place", "focus_topic"} <= set(_DEFAULT_PAYLOADS["pre_compaction"])
    assert {"session_id", "model", "platform"} <= set(_DEFAULT_PAYLOADS["post_compaction"])


def test_post_compaction_fires_only_when_a_compaction_commits():
    """The committed edge fires the post hook exactly once; aborts never fire it."""
    from agent.conversation_compression import _CompactionLifecycle

    with patch("hermes_cli.lifecycle.invoke_hook") as invoke:
        committed = _CompactionLifecycle(_agent(), status_emitted=False)
        committed.commit_status = "committed"
        committed.complete()
        assert [c.args[0] for c in invoke.call_args_list] == ["post_compaction"]
        assert invoke.call_args_list[0].kwargs["session_id"] == "sess-1"

    with patch("hermes_cli.lifecycle.invoke_hook") as invoke:
        aborted = _CompactionLifecycle(_agent(), status_emitted=True)
        aborted.complete(force_terminal=True)
        assert invoke.call_args_list == []


def test_compaction_hook_failures_are_isolated():
    """A raising hook dispatch is swallowed so compression is never disturbed."""
    from agent.conversation_compression import _fire_compaction_hook

    with patch("hermes_cli.lifecycle.invoke_hook", side_effect=RuntimeError("boom")):
        _fire_compaction_hook("pre_compaction", _agent(), trigger="auto")  # must not raise
