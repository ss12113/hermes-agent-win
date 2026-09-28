from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str = "/btw Which file was it?") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_db = None
    runner._provider_routing = {}
    runner._run_in_executor_with_context = AsyncMock(side_effect=lambda fn: fn())
    runner.session_store = MagicMock()
    runner._async_session_store = MagicMock()
    runner._async_session_store._store = runner.session_store
    runner._async_session_store.get_or_create_session = AsyncMock(
        return_value=SimpleNamespace(session_id="session-1")
    )
    return runner


@pytest.mark.asyncio
async def test_btw_without_question_shows_side_question_usage():
    runner = _make_runner()

    result = await runner._handle_btw_command(_make_event("/btw"))

    assert result == "Usage: /btw <question>"


@pytest.mark.asyncio
async def test_btw_uses_current_context_and_live_route_without_mutating_history():
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    history = [
        {"role": "user", "content": "The file was config/runtime.yaml."},
        {"role": "assistant", "content": "Noted."},
    ]
    snapshot = [dict(message) for message in history]
    agent = SimpleNamespace(
        provider="custom:example",
        model="gpt-test",
        base_url="https://example.invalid/v1",
        api_key="[REDACTED]",
        api_mode="codex_responses",
        _session_messages=history,
        session_id="session-1",
    )
    runner._running_agents[session_key] = agent

    with patch(
        "agent.side_question.answer_side_question",
        return_value="config/runtime.yaml",
    ) as answer:
        result = await runner._handle_btw_command(event)

    assert result == "💬 BTW\n\nconfig/runtime.yaml"
    assert history == snapshot
    answer.assert_called_once()
    kwargs = answer.call_args.kwargs
    assert kwargs["question"] == "Which file was it?"
    assert kwargs["messages"] == snapshot
    assert kwargs["main_runtime"]["provider"] == "custom:example"
    assert kwargs["main_runtime"]["model"] == "gpt-test"


@pytest.mark.asyncio
async def test_btw_redacts_provider_errors_before_display():
    runner = _make_runner()
    event = _make_event()
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=(
            "gpt-test",
            {
                "provider": "custom:example",
                "model": "gpt-test",
                "base_url": "https://example.invalid/v1",
                "api_key": "[REDACTED]",
                "api_mode": "codex_responses",
            },
        )
    )

    with patch(
        "gateway.run._load_gateway_config",
        return_value={},
    ), patch(
        "agent.side_question.answer_side_question",
        side_effect=RuntimeError("Authorization header: [REDACTED]"),
    ):
        result = await runner._handle_btw_command(event)

    assert result.startswith("❌ BTW failed:")
    assert "[REDACTED]" in result


@pytest.mark.asyncio
async def test_btw_loads_persisted_context_for_cold_gateway_session():
    runner = _make_runner()
    event = _make_event()
    async_store = cast(Any, runner._async_session_store)
    persisted = [
        {"role": "user", "content": "The cold-session file was config/cold.yaml."},
        {"role": "assistant", "content": "Noted."},
    ]
    async_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="cold-session-1"
    )
    session_db = AsyncMock()
    session_db.get_messages_as_conversation.return_value = persisted
    runner._session_db = session_db
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("gpt-test", {"provider": "custom:example", "model": "gpt-test"})
    )

    with patch("gateway.run._load_gateway_config", return_value={}), patch(
        "agent.side_question.answer_side_question",
        return_value="config/cold.yaml",
    ) as answer:
        result = await runner._handle_btw_command(event)

    assert result == "💬 BTW\n\nconfig/cold.yaml"
    async_store.get_or_create_session.assert_awaited_once_with(event.source)
    session_db.get_messages_as_conversation.assert_awaited_once_with("cold-session-1")
    assert answer.call_args.kwargs["messages"] == persisted


@pytest.mark.asyncio
async def test_btw_rejects_duplicate_concurrent_call_for_same_session():
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    runner._running_agents[session_key] = SimpleNamespace(
        provider="custom:example",
        model="gpt-test",
        _session_messages=[{"role": "user", "content": "context"}],
        session_id="session-1",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_executor(fn):
        started.set()
        await release.wait()
        return fn()

    runner._run_in_executor_with_context = AsyncMock(side_effect=delayed_executor)
    with patch(
        "agent.side_question.answer_side_question",
        return_value="answer",
    ) as answer:
        first = asyncio.create_task(runner._handle_btw_command(event))
        await started.wait()
        second = asyncio.create_task(runner._handle_btw_command(event))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second)

    assert results.count("A /btw side question is already running for this session.") == 1
    assert results.count("💬 BTW\n\nanswer") == 1
    answer.assert_called_once()
