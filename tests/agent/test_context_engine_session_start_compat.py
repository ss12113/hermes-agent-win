"""Compatibility tests for context-engine session-start host boundaries."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import context_engine as context_engine_module


class _StrictLegacySessionStart:
    """Third-party hook using the original session-id-only signature."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_session_start(self, session_id: str) -> None:
        self.calls.append(session_id)


class _KwargsSessionStart:
    """Current hook accepting the host's optional context fields."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def on_session_start(self, session_id: str, **context: object) -> None:
        self.calls.append((session_id, dict(context)))


class _PartialSessionStart:
    """Hook that adopted one extension field but not later additions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def on_session_start(self, session_id: str, *, platform: str) -> None:
        self.calls.append((session_id, platform))


class _KeywordOnlySessionStart:
    """Hook whose complete host contract is keyword-only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def on_session_start(self, *, session_id: str, platform: str) -> None:
        self.calls.append((session_id, platform))


class _KeywordBagOnlySessionStart:
    """Adapter accepting the complete host contract through ``**kwargs``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def on_session_start(self, **context: object) -> None:
        self.calls.append(dict(context))


class _MixedKeywordSessionStart:
    """Optional positional prefix plus an explicit keyword-only session ID."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def on_session_start(
        self,
        prefix: str = "plugin-default-prefix",
        *,
        session_id: str,
        platform: str,
        **context: object,
    ) -> None:
        self.calls.append((prefix, session_id, {"platform": platform, **context}))


class _PositionalOnlySessionStart:
    """Current hook exposing one host context field positionally only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def on_session_start(self, session_id: str, platform: str, /) -> None:
        self.calls.append((session_id, platform))


class _PositionalOnlyOptionalTailSessionStart:
    """Positional-only extension fields may have defaults and ``**kwargs``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, object]]] = []

    def on_session_start(
        self,
        session_id: str,
        platform: str = "plugin-default-platform",
        model: str = "plugin-default-model",
        /,
        **context: object,
    ) -> None:
        self.calls.append((session_id, platform, model, dict(context)))


class _BodyFailureSessionStart:
    """Hook whose implementation raises after recording the invocation."""

    def __init__(self) -> None:
        self.invocations = 0

    def on_session_start(self, session_id: str, **context: object) -> None:
        self.invocations += 1
        raise TypeError("session-start body failure")


class _PositionalOnlyBodyFailureSessionStart:
    """Positional-only hook whose body fails after one invocation."""

    def __init__(self) -> None:
        self.invocations = 0

    def on_session_start(self, session_id: str, platform: str, /) -> None:
        self.invocations += 1
        raise TypeError("positional session-start body failure")


def _start_session(engine: object, session_id: str, **context: object) -> None:
    start_session = getattr(
        context_engine_module,
        "start_context_engine_session",
    )
    start_session(engine, session_id, **context)


def test_strict_legacy_session_start_omits_extension_context() -> None:
    """New optional context must not disable a legacy engine's start hook."""
    engine = _StrictLegacySessionStart()

    _start_session(
        engine,
        "session-1",
        hermes_home="/tmp/hermes-test-home",
        platform="cli",
        model="test-model",
    )

    assert engine.calls == ["session-1"]


def test_kwargs_session_start_receives_all_extension_context() -> None:
    """Extension-friendly engines retain the complete current host contract."""
    engine = _KwargsSessionStart()

    _start_session(
        engine,
        "session-2",
        hermes_home="/tmp/hermes-test-home",
        platform="gateway",
        model="test-model",
    )

    assert engine.calls == [
        (
            "session-2",
            {
                "hermes_home": "/tmp/hermes-test-home",
                "platform": "gateway",
                "model": "test-model",
            },
        )
    ]


def test_partial_session_start_receives_only_declared_extension_fields() -> None:
    """A partially upgraded hook is not broken by newer host context keys."""
    engine = _PartialSessionStart()

    _start_session(
        engine,
        "session-3",
        platform="cli",
        model="newer-host-field",
    )

    assert engine.calls == [("session-3", "cli")]


def test_keyword_only_session_start_receives_named_contract() -> None:
    """Keyword-only adapters still receive the canonical session identifier."""
    engine = _KeywordOnlySessionStart()

    _start_session(
        engine,
        "keyword-session",
        platform="gateway",
        model="ignored-newer-field",
    )

    assert engine.calls == [("keyword-session", "gateway")]


def test_kwargs_only_session_start_receives_complete_named_contract() -> None:
    """A kwargs-only adapter receives the identifier without a fake arg slot."""
    engine = _KeywordBagOnlySessionStart()

    _start_session(
        engine,
        "kwargs-session",
        platform="gateway",
        model="test-model",
    )

    assert engine.calls == [
        {
            "session_id": "kwargs-session",
            "platform": "gateway",
            "model": "test-model",
        }
    ]


def test_mixed_session_start_keeps_prefix_default_and_names_session_id() -> None:
    """An explicit keyword-only session ID must not be shifted into a prefix."""
    engine = _MixedKeywordSessionStart()

    _start_session(
        engine,
        "mixed-session",
        platform="gateway",
        model="test-model",
    )

    assert engine.calls == [
        (
            "plugin-default-prefix",
            "mixed-session",
            {"platform": "gateway", "model": "test-model"},
        )
    ]


def test_positional_only_session_start_receives_declared_context() -> None:
    """Python positional-only spellings retain the current host contract."""
    engine = _PositionalOnlySessionStart()

    _start_session(
        engine,
        "session-positional",
        platform="gateway",
        model="unused-newer-field",
    )

    assert engine.calls == [("session-positional", "gateway")]


def test_positional_only_session_start_preserves_defaults_when_skipping_fields() -> None:
    """Later positional-only fields remain deliverable without kwarg leakage."""
    engine = _PositionalOnlyOptionalTailSessionStart()

    _start_session(
        engine,
        "session-positional-default",
        model="host-model",
        trace_id="trace-1",
    )

    assert engine.calls == [
        (
            "session-positional-default",
            "plugin-default-platform",
            "host-model",
            {"trace_id": "trace-1"},
        )
    ]


def test_session_start_body_type_error_is_not_retried() -> None:
    """A plugin body failure is propagated after exactly one invocation."""
    engine = _BodyFailureSessionStart()

    with pytest.raises(TypeError, match="session-start body failure"):
        _start_session(engine, "session-4", platform="cli")

    assert engine.invocations == 1


def test_positional_only_session_start_body_type_error_is_not_retried() -> None:
    """The positional-only adapter also invokes a failing body exactly once."""
    engine = _PositionalOnlyBodyFailureSessionStart()

    with pytest.raises(TypeError, match="positional session-start body failure"):
        _start_session(engine, "session-positional-failure", platform="cli")

    assert engine.invocations == 1


def test_uninspectable_session_start_uses_conservative_legacy_call() -> None:
    """Unknown callback signatures receive no optional extension keywords."""
    engine = _StrictLegacySessionStart()

    with patch.object(
        context_engine_module.inspect,
        "signature",
        side_effect=ValueError("opaque callback"),
    ):
        _start_session(engine, "session-5", platform="cli")

    assert engine.calls == ["session-5"]
