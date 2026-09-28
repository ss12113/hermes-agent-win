from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    mod = importlib.import_module("tui_gateway.server")
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


def _call(server, method, **params):
    return server._methods[method](1, params)


def _session(server, sid="sid-btw"):
    session = {
        "session_key": "tui-btw-session",
        "history": [
            {"role": "user", "content": "Use config/runtime.yaml."},
            {"role": "assistant", "content": "Noted."},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": True,
        "agent": SimpleNamespace(
            provider="custom:example",
            model="gpt-test",
            base_url="https://example.invalid/v1",
            api_key="[REDACTED]",
            api_mode="chat_completions",
            _session_messages=None,
        ),
    }
    server._sessions[sid] = session
    return sid, session


def test_command_dispatch_btw_returns_usage_without_calling_model(server):
    sid, _ = _session(server)

    result = _call(server, "command.dispatch", name="btw", arg="", session_id=sid)

    assert result["result"]["type"] == "exec"
    assert result["result"]["output"] == "Usage: /btw <question>"


def test_command_dispatch_btw_uses_snapshot_and_preserves_session(server):
    sid, session = _session(server)
    before = [dict(message) for message in session["history"]]

    with patch(
        "agent.side_question.answer_side_question", return_value="config/runtime.yaml"
    ) as answer:
        result = _call(
            server,
            "command.dispatch",
            name="btw",
            arg="Which file was it?",
            session_id=sid,
        )

    assert result["result"] == {"type": "exec", "output": "💬 BTW\n\nconfig/runtime.yaml"}
    assert session["history"] == before
    answer.assert_called_once()
    kwargs = answer.call_args.kwargs
    assert kwargs["question"] == "Which file was it?"
    assert kwargs["messages"] == before
    assert kwargs["main_runtime"]["model"] == "gpt-test"


def test_command_dispatch_btw_rejects_a_concurrent_side_question(server):
    sid, session = _session(server)
    lock = threading.Lock()
    lock.acquire()
    session["_btw_lock"] = lock

    result = _call(
        server,
        "command.dispatch",
        name="btw",
        arg="Can I ask another?",
        session_id=sid,
    )

    assert "already running" in result["result"]["output"]
    lock.release()


def test_slash_exec_routes_btw_in_process_instead_of_the_worker(server):
    sid, _ = _session(server)

    with patch.object(
        server,
        "_run_side_question_for_session",
        return_value="💬 BTW\n\nconfig/runtime.yaml",
    ) as answer:
        result = _call(
            server,
            "slash.exec",
            command="/btw Which file was it?",
            session_id=sid,
        )

    assert result["result"] == {
        "type": "exec",
        "output": "💬 BTW\n\nconfig/runtime.yaml",
    }
    answer.assert_called_once_with(server._sessions[sid], "Which file was it?")
    assert "btw" in server._IN_PROCESS_COMMANDS
    assert "command.dispatch" in server._LONG_HANDLERS


def test_command_dispatch_extension_bundle_wins_over_builtin_btw(server):
    sid, _ = _session(server)

    with patch(
        "agent.skill_bundles.resolve_bundle_command_key", return_value="/btw"
    ), patch(
        "agent.skill_bundles.build_bundle_invocation_message",
        return_value=("PROJECT BTW BUNDLE", ["project-btw"], []),
    ), patch(
        "agent.skill_bundles.get_skill_bundles",
        return_value={"/btw": {"name": "project-btw"}},
    ), patch.object(server, "_run_side_question_for_session") as builtin:
        result = _call(
            server,
            "command.dispatch",
            name="btw",
            arg="use the project bundle",
            session_id=sid,
        )

    assert result["result"]["type"] == "send"
    assert result["result"]["message"] == "PROJECT BTW BUNDLE"
    builtin.assert_not_called()
