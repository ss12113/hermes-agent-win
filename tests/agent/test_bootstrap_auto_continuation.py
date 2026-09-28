"""Anchored-bootstrap auto-continuation behavior contract.

The Minimal-condition first round (zero tools + minimal system prompt, see
``agent/tool_bootstrap.py``) anchors the "We need…" trajectory but cannot
execute anything by construction. Ending the turn on that prose reply would
strand the user's task — the session only gets its full catalog on their
NEXT message. The conversation loop must therefore promote and continue
automatically inside the same turn:

1. Round 1 (anchored, zero tools) produces a text reply; the loop emits it
   as an interim assistant message and re-prompts with the synthetic
   continuation nudge instead of returning it as ``final_response``.
2. Round 2 runs with the FULL tool catalog and its reply is the final
   response the user sees.
3. The auto-continuation fires exactly once per session (promotion is
   monotonic); a subsequent turn must not re-trigger it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def bootstrap_agent():
    """AIAgent with a mocked OpenAI client and an enabled zero-tools config."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="deepseek-v4-pro",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        # Non-empty catalog so the zero-tools branch can anchor. The
        # feature is fail-open on an empty catalog (never lock out tools).
        agent.tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patch",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        # Inject the config directly (skips config.yaml) — mirrors the
        # production zero-anchored-standard settings.
        agent._anchored_tool_bootstrap_config = {
            "enabled": True,
            "models": ["deepseek-v4-pro"],
            "bootstrap_zero_tools": True,
            "promote_on": "either",
        }
        return agent


def _run_bootstrap_turn(agent, round1_text, round2_text):
    """Drive one turn through the anchored round-1 -> full-tools round-2 flow."""
    from tests.agent.test_run_agent import _mock_response

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=round1_text, finish_reason="stop"),
        _mock_response(content=round2_text, finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_emit_interim_assistant_message") as emit,
    ):
        result = agent.run_conversation("检查一下服务状态")

    create = agent.client.chat.completions.create
    requests = [call.kwargs for call in create.call_args_list]
    return result, requests, emit


class TestBootstrapAutoContinuation:
    def test_round1_interim_round2_final_and_two_api_calls(self, bootstrap_agent):
        result, requests, emit = _run_bootstrap_turn(
            bootstrap_agent, "好的，开始处理。", "处理完成。"
        )

        # Round 2's reply is what the user receives as the turn's answer.
        assert result["final_response"] == "处理完成。"
        assert result.get("completed") is not False
        # Exactly one extra round: anchored round 1 + full-tool round 2.
        assert len(requests) == 2
        # The round-1 prose was surfaced as an interim, not as the final.
        emit.assert_called_once()
        assert emit.call_args[0][0].get("content") == "好的，开始处理。"

    def test_round1_zero_tools_round2_full_catalog(self, bootstrap_agent):
        _, requests, _ = _run_bootstrap_turn(
            bootstrap_agent, "好的，开始处理。", "处理完成。"
        )

        round1_tools = requests[0].get("tools")
        round2_tools = requests[1].get("tools")
        assert not round1_tools, (
            "round 1 must expose the anchored zero-tool surface, "
            f"got {round1_tools!r}"
        )
        assert len(round2_tools) == 2, (
            "round 2 must expose the full catalog after promotion, "
            f"got {round2_tools!r}"
        )

    def test_auto_continuation_fires_exactly_once_per_session(
        self, bootstrap_agent
    ):
        _, requests, _ = _run_bootstrap_turn(
            bootstrap_agent, "好的，开始处理。", "处理完成。"
        )
        assert bootstrap_agent._anchored_tool_bootstrap_auto_continued is True

        # Second turn in the same session: promoted, single round, no
        # continuation flag re-trigger (monotonic promotion latch).
        from tests.agent.test_run_agent import _mock_response

        bootstrap_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="又一个回答。", finish_reason="stop"),
        ]
        with (
            patch.object(
                bootstrap_agent, "_flush_messages_to_session_db", return_value=True
            ),
            patch.object(bootstrap_agent, "_persist_session"),
            patch.object(bootstrap_agent, "_save_trajectory"),
            patch.object(bootstrap_agent, "_cleanup_task_resources"),
            patch.object(bootstrap_agent, "_emit_interim_assistant_message") as emit2,
        ):
            result2 = bootstrap_agent.run_conversation("再来一次")

        assert result2["final_response"] == "又一个回答。"
        assert bootstrap_agent.client.chat.completions.create.call_count == 3
        emit2.assert_not_called()
