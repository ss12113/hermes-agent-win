"""Regression tests for the one-shot no-tool stop guard.

``hermes -z`` runs a single task conversation; the run must not end before
the agent has executed at least one tool. Some models (observed:
deepseek-v4-flash, 2026-08-16) open with a narration-only message and stop
with ``finish_reason=stop`` and zero tool calls — a clean exit that ends
the whole run with the task unstarted. The guard (agent/oneshot_task_guard.py)
re-prompts, bounded to 3 consecutive stalls. Interactive chat / gateway
sessions (no ``_oneshot_mode`` flag) are unaffected, as are one-shot turns
that already executed a tool.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors the dropped-tool-call
    recovery fixture) so we can stage text-only stop responses."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
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
        return agent


def _text_stop_response(content: str):
    """A clean text-only finish_reason=stop response with no tool calls."""
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="chatcmpl-textstop",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )


class TestOneshotNoToolGuard:
    def test_text_stop_before_any_tool_call_nudges(self, loop_agent):
        """In one-shot mode, a text-only stop before any tool call must
        re-prompt instead of exiting with the task unstarted."""
        loop_agent._oneshot_mode = True
        loop_agent.client.chat.completions.create.side_effect = [
            _text_stop_response("我先了解工作区结构，再按流程推进。"),
            _text_stop_response("still narrating"),
            _text_stop_response("still narrating"),
            _text_stop_response("final text answer"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("fix the project")

        # 3 bounded nudges then a clean final exit = 4 API calls total.
        assert loop_agent.client.chat.completions.create.call_count == 4, (
            "A text-only stop with zero tool calls must trigger bounded "
            "re-prompts, not exit after one call."
        )

        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "tool" in (last_user.get("content") or "").lower(), (
            "The nudge must explicitly tell the model to call a tool."
        )
        assert "final text answer" in result["final_response"]

    def test_clean_stop_unaffected_outside_oneshot(self, loop_agent):
        """Interactive/gateway sessions never set _oneshot_mode: a genuine
        text-only stop must exit normally on the first turn."""
        loop_agent.client.chat.completions.create.side_effect = [
            _text_stop_response("Here is your answer."),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "Without the one-shot flag a clean text stop must not re-prompt."
        )
        assert "Here is your answer." in result["final_response"]

    def test_nudge_pair_is_ephemeral_scaffolding(self, loop_agent):
        """The re-prompt pair must be flagged so persistence never writes it
        to the durable transcript."""
        from agent.session_persistence import _is_ephemeral_scaffolding

        loop_agent._oneshot_mode = True
        loop_agent.client.chat.completions.create.side_effect = [
            _text_stop_response("我先了解一下。"),
            _text_stop_response("still narrating"),
            _text_stop_response("still narrating"),
            _text_stop_response("final text answer"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("fix the project")

        assert result["completed"] is True
        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_oneshot_start_nudge")
        ]
        assert not leftover, (
            "The nudge pair must be stripped at finalization, not kept in "
            "the returned transcript."
        )
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_oneshot_start_nudge": True}
        ), (
            "_oneshot_start_nudge messages must be classified as ephemeral "
            "scaffolding so they are never persisted."
        )

    def test_guard_off_when_tool_was_executed(self):
        """has_executed_tool_call must detect an earlier assistant tool call,
        so the guard stays silent once real work has started."""
        from agent.oneshot_task_guard import has_executed_tool_call

        assert not has_executed_tool_call(None)
        assert not has_executed_tool_call([])
        assert not has_executed_tool_call([
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "ok, working"},
        ])
        assert has_executed_tool_call([
            {"role": "user", "content": "do the task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file"}}],
            },
        ])

    def test_nudge_builder_bounds_and_toggle(self):
        """The nudge builder must respect the attempt bound and the
        HERMES_ONESHOT_START_NUDGE=0 kill switch."""
        from agent.oneshot_task_guard import build_oneshot_start_nudge

        assert build_oneshot_start_nudge(attempts=0) is not None
        assert build_oneshot_start_nudge(attempts=2) is not None
        assert build_oneshot_start_nudge(attempts=3) is None
        with patch.dict("os.environ", {"HERMES_ONESHOT_START_NUDGE": "0"}):
            assert build_oneshot_start_nudge(attempts=0) is None
