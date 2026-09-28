from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = SimpleNamespace(
        _session_messages=[
            {"role": "user", "content": "The file is config/runtime.yaml."},
            {"role": "assistant", "content": "I noted it."},
        ],
        provider="custom:example",
        model="gpt-test",
        base_url="https://example.invalid/v1",
        api_key="[REDACTED]",
        api_mode="chat_completions",
    )
    cli.conversation_history = list(cli.agent._session_messages)
    cli.config = {}
    cli.session_id = "cli-btw-session"
    cli._pending_input = MagicMock()
    cli._background_tasks = {}
    cli._app = None
    cli._agent_running = False
    cli.final_response_markdown = "auto"
    cli.bell_on_complete = False
    cli._ensure_runtime_credentials = MagicMock(return_value=True)
    cli._resolve_turn_agent_config = MagicMock(
        return_value={
            "model": "gpt-test",
            "runtime": {
                "provider": "custom:example",
                "base_url": "https://example.invalid/v1",
                "api_key": "[REDACTED]",
                "api_mode": "chat_completions",
            },
        }
    )
    return cli


def test_btw_empty_argument_reports_usage_without_starting_worker():
    cli = _make_cli()

    with patch("cli._cprint") as print_line:
        cli._handle_btw_command("/btw")

    print_line.assert_any_call("  Usage: /btw <question>")
    assert cli._background_tasks == {}


def test_btw_snapshots_context_and_does_not_append_parent_history():
    cli = _make_cli()
    original = [dict(message) for message in cli.agent._session_messages]

    with patch("cli._cprint"), patch(
        "agent.side_question.answer_side_question", return_value="config/runtime.yaml"
    ) as answer, patch("cli.ChatConsole") as console:
        console.return_value.print = MagicMock()
        cli._handle_btw_command("/btw Which file was it?")
        for thread in list(cli._background_tasks.values()):
            thread.join(timeout=10)

    assert cli.agent._session_messages == original
    assert cli.conversation_history == original
    answer.assert_called_once()
    kwargs = answer.call_args.kwargs
    assert kwargs["question"] == "Which file was it?"
    assert kwargs["messages"] == original
    assert kwargs["main_runtime"]["model"] == "gpt-test"
    assert kwargs["main_runtime"]["provider"] == "custom:example"
    assert cli._background_tasks == {}


def test_process_command_dispatches_btw_to_dedicated_handler():
    cli = _make_cli()

    with patch.object(cli, "_handle_btw_command") as handler:
        assert cli.process_command("/btw Which file was it?") is True

    handler.assert_called_once_with("/btw Which file was it?")


def test_btw_skill_extension_keeps_precedence_over_builtin_handler():
    cli = _make_cli()
    skill = {
        "/btw": {
            "name": "project-btw",
            "description": "Project-local side workflow",
        }
    }

    with patch("cli._ensure_skill_commands", return_value=skill), patch(
        "cli.get_skill_bundles", return_value={}
    ), patch("cli._get_plugin_cmd_handler_names", return_value=set()), patch(
        "cli.build_skill_invocation_message", return_value="PROJECT BTW PROMPT"
    ), patch.object(cli, "_handle_btw_command") as builtin:
        assert cli.process_command("/btw use the project workflow") is True

    builtin.assert_not_called()
    cli._pending_input.put.assert_called_once_with("PROJECT BTW PROMPT")


def test_btw_is_the_only_side_question_inline_busy_command():
    cli = _make_cli()
    cli._agent_running = True

    assert cli._should_handle_btw_command_inline("/btw what file?") is True
    assert cli._should_handle_btw_command_inline("/background run tests") is False

    cli._agent_running = False
    assert cli._should_handle_btw_command_inline("/btw what file?") is False
