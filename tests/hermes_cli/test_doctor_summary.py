from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli import doctor_summary


def _tool(name: str, description: str = "short") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_tool_schema_context_ok_for_small_runtime_tools(monkeypatch):
    monkeypatch.setattr(doctor_summary, "_TOOL_SCHEMA_WARN_TOKENS", 10_000)

    lines, warnings = doctor_summary._tool_schema_warning_lines([_tool("read_file")])

    assert warnings == []
    assert "Tool Schema Context" in lines
    assert any("1 tools" in line for line in lines)
    assert any("Budget: OK" in line for line in lines)


def test_tool_schema_context_warns_and_lists_largest_tools(monkeypatch):
    monkeypatch.setattr(doctor_summary, "_TOOL_SCHEMA_WARN_TOKENS", 10)

    lines, warnings = doctor_summary._tool_schema_warning_lines(
        [
            _tool("tiny", "x"),
            _tool("huge_tool", "y" * 400),
        ]
    )

    rendered = "\n".join(lines)
    assert "Above 10 token warning threshold" in rendered
    assert "huge_tool" in rendered
    assert "Fix: disable rarely used toolsets" in rendered
    assert len(warnings) == 1
    assert warnings[0].startswith("Large tool schema context:")
    assert "2 tools" in warnings[0]


def test_format_doctor_summary_includes_runtime_tool_schema_section(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_summary, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(doctor_summary, "_hermes_version", lambda: "test-version")
    monkeypatch.setattr(doctor_summary, "_invoked_binary", lambda _root: "hermes")
    monkeypatch.setattr(doctor_summary, "_hermes_home_display", lambda: "~/.hermes")
    monkeypatch.setattr(doctor_summary, "_ripgrep_line", lambda: ("└ Search: OK", None))
    monkeypatch.setattr(doctor_summary, "_configuration_lines", lambda: (["Configuration", "└ Structure: OK"], []))
    monkeypatch.setattr(doctor_summary, "_security_lines", lambda: (["Security Advisories", "└ None"], []))
    monkeypatch.setattr(doctor_summary, "_context_warning_lines", lambda _cwd: (["Context Usage Warnings", "└ None"], []))

    output = doctor_summary.format_doctor_summary(
        session_id="sess-1",
        provider="custom",
        model="model-x",
        cwd=tmp_path,
        runtime_tools=[_tool("terminal")],
    )

    assert "Hermes Doctor diagnostics" in output
    assert "Runtime model: custom/model-x" in output
    assert "Tool Schema Context" in output
    assert "Runtime schemas: 1 tools" in output
    assert "No blocking issues" in output


def test_tui_doctor_dispatch_passes_live_agent_tool_schemas(monkeypatch, tmp_path):
    from tui_gateway import server

    captured = {}

    def fake_format_doctor_summary(**kwargs):
        captured.update(kwargs)
        return "doctor-output"

    monkeypatch.setattr(doctor_summary, "format_doctor_summary", fake_format_doctor_summary)

    sid = "doctor-session"
    tools = [_tool("terminal")]
    agent = SimpleNamespace(provider="custom", model="model-x", tools=tools)
    server._sessions[sid] = {
        "session_key": "session-key",
        "cwd": str(tmp_path),
        "agent": agent,
    }
    try:
        response = server._methods["command.dispatch"](
            "r-doctor", {"name": "doctor", "session_id": sid}
        )
    finally:
        server._sessions.pop(sid, None)

    assert response["result"] == {"type": "exec", "output": "doctor-output"}
    assert captured["session_id"] == "session-key"
    assert captured["provider"] == "custom"
    assert captured["model"] == "model-x"
    assert captured["cwd"] == str(tmp_path)
    assert captured["runtime_tools"] is tools


@pytest.mark.asyncio
async def test_gateway_doctor_handler_passes_live_agent_tool_schemas(monkeypatch):
    from gateway.slash_commands import GatewaySlashCommandsMixin

    captured = {}

    def fake_format_doctor_summary(**kwargs):
        captured.update(kwargs)
        return "gateway-doctor-output"

    monkeypatch.setattr(doctor_summary, "format_doctor_summary", fake_format_doctor_summary)

    class Runner(GatewaySlashCommandsMixin):
        pass

    tools = [_tool("terminal")]
    agent = SimpleNamespace(provider="custom", model="model-y", tools=tools)
    runner = Runner()
    runner.async_session_store = AsyncMock()
    runner.async_session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="gw-session",
        session_key="gw-key",
    )
    runner._running_agents = {"gw-key": agent}
    runner._agent_cache = {}

    output = await runner._handle_doctor_command(SimpleNamespace(source="source"))

    assert output == "gateway-doctor-output"
    assert captured["session_id"] == "gw-session"
    assert captured["provider"] == "custom"
    assert captured["model"] == "model-y"
    assert captured["runtime_tools"] is tools
