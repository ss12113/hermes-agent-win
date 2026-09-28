"""Compatibility tests for context-engine session-end host boundaries."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import context_engine as context_engine_module
from run_agent import AIAgent


class _StrictLegacySessionEnd:
    """Engine using a payload-only session-end callback."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def on_session_end(self, messages: list[dict[str, object]]) -> None:
        self.calls.append(messages)


class _CurrentSessionEnd:
    """Engine accepting the current identifiers plus extension metadata."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, object]], dict[str, object]]] = []

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict[str, object]],
        **context: object,
    ) -> None:
        self.calls.append((session_id, messages, dict(context)))


class _PartialSessionEnd:
    """Engine that adopted one optional lifecycle field but not later ones."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], str]] = []

    def on_session_end(
        self,
        messages: list[dict[str, object]],
        *,
        platform: str,
    ) -> None:
        self.calls.append((messages, platform))


class _BodyFailureSessionEnd:
    """Engine whose body raises after proving it was called once."""

    def __init__(self) -> None:
        self.invocations = 0

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict[str, object]],
    ) -> None:
        self.invocations += 1
        raise TypeError("session-end body failure")


class _UninspectableLegacySessionEnd:
    """Strict legacy callback used for the conservative fallback test."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def on_session_end(self, messages: list[dict[str, object]]) -> None:
        self.calls.append(messages)


class _RenamedLegacySessionEnd:
    """Legacy callback whose payload parameter uses a private name."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def on_session_end(self, history: list[dict[str, object]]) -> None:
        self.calls.append(history)


class _SessionIdOnlySessionEnd:
    """Partial current callback that only needs the outgoing session id."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_session_end(self, session_id: str) -> None:
        self.calls.append(session_id)


class _RenamedCurrentSessionEnd:
    """Current callback whose positional parameters use private names."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def on_session_end(self, sid: str, history: list[dict[str, object]]) -> None:
        self.calls.append((sid, history))


class _VarArgsKeywordOnlySessionEnd:
    """Adapter-shaped callback with a variadic prefix and named payload."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], list[dict[str, object]]]] = []

    def on_session_end(
        self,
        *args: object,
        messages: list[dict[str, object]],
    ) -> None:
        self.calls.append((args, messages))


class _OptionalLegacySessionEnd:
    """Legacy payload callback with an optional extension argument."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], object]] = []

    def on_session_end(
        self,
        messages: list[dict[str, object]],
        reason: object = None,
    ) -> None:
        self.calls.append((messages, reason))


class _RenamedOptionalLegacySessionEnd:
    """Legacy payload callback with a renamed optional positional extension."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], object]] = []

    def on_session_end(
        self,
        history: list[dict[str, object]],
        reason: object = None,
    ) -> None:
        self.calls.append((history, reason))


class _ReversedCurrentOptionalSessionEnd:
    """Current callback with reversed named payloads and an optional tail."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], str, object]] = []

    def on_session_end(
        self,
        messages: list[dict[str, object]],
        session_id: str,
        reason: object = None,
    ) -> None:
        self.calls.append((messages, session_id, reason))


def _end_session(engine: object, session_id: str, messages: list[dict[str, object]], **context: object) -> None:
    end_session = getattr(context_engine_module, "end_context_engine_session")
    end_session(engine, session_id, messages, **context)


def test_strict_legacy_session_end_receives_messages_only() -> None:
    engine = _StrictLegacySessionEnd()
    messages = [{"role": "user", "content": "hello"}]

    _end_session(engine, "session-1", messages, platform="cli", model="test-model")

    assert engine.calls == [messages]


def test_current_session_end_receives_ids_and_extension_context() -> None:
    engine = _CurrentSessionEnd()
    messages = [{"role": "assistant", "content": "done"}]

    _end_session(engine, "session-2", messages, platform="gateway", model="test-model")

    assert engine.calls == [
        ("session-2", messages, {"platform": "gateway", "model": "test-model"})
    ]


def test_partial_session_end_receives_only_declared_context() -> None:
    engine = _PartialSessionEnd()
    messages = [{"role": "user", "content": "partial"}]

    _end_session(engine, "session-3", messages, platform="cli", model="newer-field")

    assert engine.calls == [(messages, "cli")]


def test_session_end_body_type_error_is_not_retried() -> None:
    engine = _BodyFailureSessionEnd()

    with pytest.raises(TypeError, match="session-end body failure"):
        _end_session(engine, "session-4", [], platform="cli")

    assert engine.invocations == 1


def test_uninspectable_session_end_uses_payload_only_fallback() -> None:
    engine = _UninspectableLegacySessionEnd()
    messages = [{"role": "user", "content": "opaque"}]

    with patch.object(
        context_engine_module.inspect,
        "signature",
        side_effect=ValueError("opaque callback"),
    ):
        _end_session(engine, "session-5", messages, platform="cli")

    assert engine.calls == [messages]


def test_renamed_legacy_positional_session_end_keeps_payload_compatibility() -> None:
    engine = _RenamedLegacySessionEnd()
    messages = [{"role": "user", "content": "renamed"}]

    _end_session(engine, "session-5a", messages, platform="cli")

    assert engine.calls == [messages]


def test_session_id_only_end_hook_receives_session_id() -> None:
    """A one-argument current hook must not receive the transcript payload."""
    engine = _SessionIdOnlySessionEnd()
    messages = [{"role": "user", "content": "session-id-only"}]

    _end_session(engine, "session-id-only", messages)

    assert engine.calls == ["session-id-only"]


def test_renamed_current_positional_session_end_keeps_identifier_order() -> None:
    engine = _RenamedCurrentSessionEnd()
    messages = [{"role": "assistant", "content": "renamed-current"}]

    _end_session(engine, "session-5b", messages)

    assert engine.calls == [("session-5b", messages)]


def test_variadic_prefix_does_not_hide_required_keyword_only_payload() -> None:
    engine = _VarArgsKeywordOnlySessionEnd()
    messages = [{"role": "user", "content": "keyword-only"}]

    _end_session(engine, "session-5c", messages)

    assert engine.calls == [((), messages)]


def test_optional_legacy_extension_is_not_misclassified_as_current_payload() -> None:
    engine = _OptionalLegacySessionEnd()
    messages = [{"role": "user", "content": "optional-extension"}]

    _end_session(engine, "session-5d", messages)

    assert engine.calls == [(messages, None)]


def test_renamed_optional_legacy_extension_keeps_payload_position() -> None:
    engine = _RenamedOptionalLegacySessionEnd()
    messages = [{"role": "user", "content": "renamed-optional-extension"}]

    _end_session(engine, "session-5e", messages)

    assert engine.calls == [(messages, None)]


def test_reversed_current_signature_omits_optional_tail_without_swapping_payloads() -> None:
    engine = _ReversedCurrentOptionalSessionEnd()
    messages = [{"role": "user", "content": "reversed-current-optional"}]

    _end_session(engine, "session-5f", messages)

    assert engine.calls == [(messages, "session-5f", None)]


def test_transition_delivers_strict_legacy_session_end() -> None:
    engine = _StrictLegacySessionEnd()
    agent = object.__new__(AIAgent)
    agent.session_id = "new-session"
    agent.model = "test-model"
    agent.platform = "cli"
    agent.context_compressor = engine

    messages = [{"role": "user", "content": "old turn"}]
    agent._transition_context_engine_session(
        old_session_id="old-session",
        new_session_id="new-session",
        previous_messages=messages,
    )

    assert engine.calls == [messages]


def test_commit_and_shutdown_deliver_strict_legacy_session_end() -> None:
    engine = _StrictLegacySessionEnd()
    agent = object.__new__(AIAgent)
    agent.session_id = "session-6"
    agent.context_compressor = engine
    agent._memory_manager = None
    messages = [{"role": "assistant", "content": "done"}]

    agent.commit_memory_session(messages)
    agent.shutdown_memory_provider(messages)

    assert engine.calls == [messages, messages]
