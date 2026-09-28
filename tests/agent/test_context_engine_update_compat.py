"""Compatibility tests for context-engine model updates at host boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngine, update_context_engine_model


class _LegacyContextEngine(ContextEngine):
    """Third-party engine using the pre-``api_mode`` update signature."""

    def __init__(self) -> None:
        super().__init__()
        self.update_calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "legacy-update-model"

    def update_from_response(self, usage: dict) -> None:
        return None

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        return False

    def compress(
        self,
        messages: list[dict],
        current_tokens: int | None = None,
    ) -> list[dict]:
        return messages

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.update_calls.append(
            {
                "model": model,
                "context_length": context_length,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
            }
        )
        super().update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )


class _ModernContextEngine(_LegacyContextEngine):
    """Engine implementing the current update signature."""

    def __init__(self) -> None:
        super().__init__()
        self.api_modes: list[str] = []

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.api_modes.append(api_mode)
        super().update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )


class _BodyTypeErrorContextEngine(_ModernContextEngine):
    """Engine whose callback itself raises ``TypeError`` after one side effect."""

    def __init__(self) -> None:
        super().__init__()
        self.invocations = 0

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.invocations += 1
        raise TypeError("engine body failure")


class _KwargsContextEngine(_LegacyContextEngine):
    """Engine that accepts contract extensions through ``**kwargs``."""

    def __init__(self) -> None:
        super().__init__()
        self.extra_kwargs: list[dict[str, object]] = []

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        **kwargs: object,
    ) -> None:
        self.extra_kwargs.append(dict(kwargs))
        super().update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )


class _UninspectableLegacyUpdate:
    """Strict legacy callback whose advertised signature cannot be inspected."""

    __signature__ = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.calls.append((model, context_length))


class _PositionalOnlyLegacyUpdate:
    """Legacy update callback that exposes the contract positionally only."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: object = "",
        provider: str = "",
        /,
    ) -> None:
        self.calls.append((model, context_length, base_url, api_key, provider))


class _PositionalOnlyCurrentUpdate:
    """Current update callback with positional-only ``api_mode`` support."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: object = "",
        provider: str = "",
        api_mode: str = "",
        /,
    ) -> None:
        self.calls.append(
            (model, context_length, base_url, api_key, provider, api_mode)
        )


def test_current_update_model_receives_api_mode() -> None:
    """The compatibility adapter preserves the current engine contract."""
    engine = _ModernContextEngine()

    update_context_engine_model(
        engine,
        model="current-model",
        context_length=128_000,
        api_mode="anthropic_messages",
    )

    assert engine.api_modes == ["anthropic_messages"]
    assert engine.update_calls[0]["model"] == "current-model"


def test_positional_only_legacy_update_model_receives_contract_values() -> None:
    engine = _LegacyContextEngine()
    update = _PositionalOnlyLegacyUpdate()
    engine.update_model = update  # type: ignore[method-assign]

    update_context_engine_model(
        engine,
        model="positional-legacy-model",
        context_length=64_000,
        base_url="https://example.invalid/v1",
        api_key="[REDACTED]",
        provider="test-provider",
        api_mode="chat_completions",
    )

    assert update.calls == [
        (
            "positional-legacy-model",
            64_000,
            "https://example.invalid/v1",
            "[REDACTED]",
            "test-provider",
        )
    ]


def test_positional_only_current_update_model_receives_api_mode() -> None:
    engine = _LegacyContextEngine()
    update = _PositionalOnlyCurrentUpdate()
    engine.update_model = update  # type: ignore[method-assign]

    update_context_engine_model(
        engine,
        model="positional-current-model",
        context_length=64_000,
        api_mode="anthropic_messages",
    )

    assert update.calls == [
        (
            "positional-current-model",
            64_000,
            "",
            "",
            "",
            "anthropic_messages",
        )
    ]


def test_update_model_body_type_error_is_not_retried_or_swallowed() -> None:
    """Signature compatibility must not turn callback failures into retries."""
    engine = _BodyTypeErrorContextEngine()

    with pytest.raises(TypeError, match="engine body failure"):
        update_context_engine_model(
            engine,
            model="current-model",
            context_length=128_000,
            api_mode="chat_completions",
        )

    assert engine.invocations == 1


def test_kwargs_update_model_receives_api_mode() -> None:
    """Extension-friendly legacy engines receive the new keyword."""
    engine = _KwargsContextEngine()

    update_context_engine_model(
        engine,
        model="kwargs-model",
        context_length=96_000,
        api_mode="anthropic_messages",
    )

    assert engine.extra_kwargs == [{"api_mode": "anthropic_messages"}]


def test_legacy_update_model_preserves_callable_api_key_identity() -> None:
    """The adapter must not coerce rotating-token provider callables."""
    engine = _LegacyContextEngine()

    def token_provider() -> str:
        return "rotating-test-token"

    update_context_engine_model(
        engine,
        model="callable-key-model",
        context_length=96_000,
        api_key=token_provider,
        api_mode="chat_completions",
    )

    assert engine.update_calls[0]["api_key"] is token_provider


def test_uninspectable_legacy_update_model_uses_conservative_contract() -> None:
    """Unknown signatures omit the optional extension-only ``api_mode`` keyword."""
    engine = _LegacyContextEngine()
    update = _UninspectableLegacyUpdate()
    engine.update_model = update  # type: ignore[method-assign]

    update_context_engine_model(
        engine,
        model="legacy-model",
        context_length=64_000,
        api_mode="chat_completions",
    )

    assert update.calls == [("legacy-model", 64_000)]


def _make_agent(
    engine: ContextEngine,
    *,
    fallback_model: dict[str, str] | None = None,
    api_mode: str | None = None,
):
    from plugins.context_engine import PreparedContextEngine
    from run_agent import AIAgent

    candidate = PreparedContextEngine(
        engine=engine,
        engine_name=engine.name,
        commands=(),
    )
    config = {"context": {"engine": engine.name}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "plugins.context_engine.prepare_context_engine",
            return_value=candidate,
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=204_800,
        ),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key-redacted",
            base_url="https://example.invalid/v1",
            api_mode=api_mode,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )


def _successful_response(content: str = "recovered") -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_agent_init_supports_legacy_update_model_signature() -> None:
    """A pre-api_mode repository engine remains usable during agent startup."""
    engine = _LegacyContextEngine()

    agent = _make_agent(engine)

    assert agent.context_compressor is engine
    assert engine.update_calls == [
        {
            "model": agent.model,
            "context_length": 204_800,
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key-redacted",
            "provider": agent.provider,
        }
    ]


def test_model_switch_supports_legacy_update_model_signature() -> None:
    """Live model switches use the same compatibility boundary as startup."""
    engine = _LegacyContextEngine()
    agent = _make_agent(engine)
    engine.update_calls.clear()

    with (
        patch.object(agent, "_create_openai_client", return_value=MagicMock()),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=131_072,
        ),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
    ):
        agent.switch_model(
            "next-model",
            agent.provider,
            api_key="next-test-key",
            base_url=agent.base_url,
            api_mode="chat_completions",
        )

    assert engine.update_calls == [
        {
            "model": "next-model",
            "context_length": 131_072,
            "base_url": "https://example.invalid/v1",
            "api_key": "next-test-key",
            "provider": agent.provider,
        }
    ]


def test_fallback_supports_legacy_update_model_signature() -> None:
    """Fallback activation does not discard an otherwise healthy legacy engine."""
    engine = _LegacyContextEngine()
    agent = _make_agent(
        engine,
        fallback_model={"provider": "openrouter", "model": "fallback-model"},
    )
    engine.update_calls.clear()
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-test-key"
    fallback_client.base_url = "https://fallback.example.invalid/v1"

    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, None),
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=96_000,
        ),
        patch("hermes_cli.config.load_config", return_value={}),
    ):
        activated = agent._try_activate_fallback()

    assert activated is True
    assert engine.update_calls == [
        {
            "model": "fallback-model",
            "context_length": 96_000,
            "base_url": "https://fallback.example.invalid/v1",
            "api_key": "fallback-test-key",
            "provider": "openrouter",
        }
    ]


def test_primary_runtime_restore_supports_legacy_update_model_signature() -> None:
    """Turn-boundary restoration keeps legacy context engines active."""
    engine = _LegacyContextEngine()
    agent = _make_agent(engine)
    expected_runtime = dict(agent._primary_runtime)
    engine.update_calls.clear()
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent.context_compressor.context_length = 1

    with patch.object(
        agent,
        "_create_openai_client",
        return_value=MagicMock(),
    ):
        restored = agent._restore_primary_runtime()

    assert restored is True
    assert agent._fallback_activated is False
    assert engine.update_calls == [
        {
            "model": expected_runtime["compressor_model"],
            "context_length": expected_runtime["compressor_context_length"],
            "base_url": expected_runtime["compressor_base_url"],
            "api_key": expected_runtime["compressor_api_key"],
            "provider": expected_runtime["compressor_provider"],
        }
    ]


@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_primary_runtime_restore_preserves_current_engine_api_mode(
    legacy_snapshot: bool,
) -> None:
    """Restoring a fallback turn re-applies the primary engine's API mode."""
    engine = _ModernContextEngine()
    agent = _make_agent(engine, api_mode="chat_completions")
    if legacy_snapshot:
        agent._primary_runtime.pop("compressor_api_mode")
    engine.api_modes.clear()
    engine.update_calls.clear()
    agent._fallback_activated = True
    agent._fallback_index = 1

    with patch.object(
        agent,
        "_create_openai_client",
        return_value=MagicMock(),
    ):
        restored = agent._restore_primary_runtime()

    assert restored is True
    assert engine.api_modes == ["chat_completions"]


@pytest.mark.parametrize(
    ("status_code", "error_message"),
    [
        pytest.param(
            429,
            "Extra usage is required for long context requests.",
            id="long-context-tier",
        ),
        pytest.param(
            400,
            "prompt is too long: 233153 tokens > 200000 maximum",
            id="provider-reported-limit",
        ),
    ],
)
def test_context_error_recovery_supports_legacy_update_model_signature(
    status_code: int,
    error_message: str,
) -> None:
    """Both runtime context-limit recovery paths use the compatibility adapter."""
    engine = _LegacyContextEngine()
    agent = _make_agent(engine, api_mode="chat_completions")
    agent.model = "claude-sonnet-4.6"
    agent.provider = "anthropic"
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = True
    agent.save_trajectories = False
    engine.context_length = 1_000_000
    engine.threshold_tokens = 500_000
    engine.update_calls.clear()

    error = Exception(error_message)
    error.status_code = status_code
    agent.client.chat.completions.create.side_effect = [
        error,
        _successful_response(),
    ]

    with (
        patch("agent.conversation_loop.time.sleep"),
        patch.object(
            agent,
            "_compress_context",
            return_value=([{"role": "user", "content": "hello"}], "compressed"),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "hello",
            conversation_history=[
                {"role": "user", "content": "previous"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    assert result["completed"] is True
    assert [call["context_length"] for call in engine.update_calls] == [200_000]


def test_lmstudio_preload_supports_legacy_update_model_signature() -> None:
    """A successful local preload updates legacy engines instead of failing silently."""
    engine = _LegacyContextEngine()
    agent = _make_agent(engine)
    engine.update_calls.clear()
    agent.model = "local-model"
    agent.provider = "lmstudio"

    # Patch the defining sibling: the ``hermes_cli.models`` re-export is a
    # plugin-compat pointer scheduled for removal (see COMPAT_MANIFEST.md).
    with patch(
        "hermes_cli.models_local.ensure_lmstudio_model_loaded",
        return_value=160_000,
    ):
        agent._ensure_lmstudio_runtime_loaded()

    assert engine.update_calls == [
        {
            "model": "local-model",
            "context_length": 160_000,
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key-redacted",
            "provider": "lmstudio",
        }
    ]
