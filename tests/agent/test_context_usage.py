from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.context_usage import estimate_live_context_usage


def _agent(**kwargs):
    defaults = {
        "session_total_tokens": 0,
        "context_compressor": SimpleNamespace(
            last_prompt_tokens=0,
            context_length=200_000,
        ),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_reported_prompt_tokens_win_over_estimate():
    agent = _agent(
        context_compressor=SimpleNamespace(last_prompt_tokens=42_000, context_length=200_000),
        _session_messages=[{"role": "user", "content": "short"}],
        _cached_system_prompt="system prompt",
        tools=[{"function": {"name": "terminal", "description": "tool"}}],
    )

    usage = estimate_live_context_usage(agent)

    assert usage.tokens == 42_000
    assert usage.context_length == 200_000
    assert usage.source == "reported"
    assert usage.estimated is False


def test_estimates_messages_system_prompt_and_tools_when_report_is_missing():
    agent = _agent(
        _session_messages=[{"role": "user", "content": "abcd"}],
        _cached_system_prompt="efgh",
        tools=[{"function": {"name": "terminal", "description": "ijkl"}}],
    )

    usage = estimate_live_context_usage(agent)

    # 1 token for message text + 1 for system prompt + a non-zero schema cost.
    assert usage.tokens > 2
    assert usage.context_length == 200_000
    assert usage.source == "estimated"
    assert usage.estimated is True


def test_negative_post_compression_sentinel_uses_fallback_messages():
    agent = _agent(
        context_compressor=SimpleNamespace(last_prompt_tokens=-1, context_length=100_000),
    )

    usage = estimate_live_context_usage(
        agent,
        fallback_messages=[{"role": "user", "content": "abcd"}],
    )

    assert usage.tokens > 0
    assert usage.context_length == 100_000
    assert usage.source == "estimated"


def test_ignores_magicmock_auto_attributes_and_falls_back_to_session_total():
    agent = MagicMock()
    agent.session_total_tokens = 12_345
    agent.context_compressor.last_prompt_tokens = 0
    agent.context_compressor.context_length = 200_000

    usage = estimate_live_context_usage(agent)

    assert usage.tokens == 12_345
    assert usage.context_length == 200_000
    assert usage.source == "session_total"
    assert usage.estimated is True
