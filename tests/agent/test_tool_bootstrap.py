"""Tests for agent.tool_bootstrap (dsh-anchored-standard port)."""

from types import SimpleNamespace

import pytest

from agent.tool_bootstrap import (
    _AGENT_ACTIVE_REQUEST_ATTR,
    _AGENT_CFG_ATTR,
    _AGENT_PROMOTED_ATTR,
    apply_tool_bootstrap,
)

FULL_CATALOG = [
    {"type": "function", "name": "terminal", "description": "run shell"},
    {"type": "function", "name": "read_file", "description": "read file"},
    {"type": "function", "name": "write_file", "description": "write file"},
    {"type": "function", "name": "web_search", "description": "search web"},
    {"type": "function", "name": "patch", "description": "edit file"},
]

# Real Hermes registrations use the OpenAI nested layout
# {"type": "function", "function": {"name": ...}}.
NESTED_CATALOG = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "run shell",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "read file",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "write file",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search web",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "edit file",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _agent(model="deepseek-v4-pro", cfg=None):
    agent = SimpleNamespace(model=model)
    setattr(agent, _AGENT_CFG_ATTR, cfg)
    return agent


def _messages(*markers):
    """Build history markers: 'user', 'toolcall' (assistant w/ tool_calls),
    'toolresult' (role=tool)."""
    out = []
    for m in markers:
        if m == "user":
            out.append({"role": "user", "content": "hi"})
        elif m == "toolcall":
            out.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]})
        elif m == "toolresult":
            out.append({"role": "tool", "tool_call_id": "c1", "content": "ok"})
        else:  # pragma: no cover
            raise ValueError(m)
    return out


@pytest.mark.parametrize(
    "cfg",
    [
        None,  # attribute missing entirely → treat as disabled
        {"enabled": False},
        {"enabled": False, "models": ["deepseek-v4-pro"], "bootstrap_tools": ["terminal", "read_file"]},
        {},
    ],
)
def test_disabled_returns_catalog_unchanged(cfg):
    agent = _agent(cfg=cfg)
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == FULL_CATALOG


def test_model_not_matching_returns_catalog_unchanged():
    cfg = {"enabled": True, "models": ["deepseek-v4-pro"], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(model="gpt-5.6-sol", cfg=cfg)
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == FULL_CATALOG


def test_empty_models_matches_every_model():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    for model in ("deepseek-v4-pro", "deepseek-v4-flash", "gpt-5.6-sol"):
        agent = _agent(model=model, cfg=cfg)
        out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
        assert [t["name"] for t in out] == ["terminal", "read_file"]


def test_model_substring_match():
    cfg = {"enabled": True, "models": ["deepseek-v4-pro"], "bootstrap_tools": ["terminal", "read_file"]}
    # Upstream often names the checkpoint variant deepseek-v4-pro-0813.
    agent = _agent(model="deepseek-v4-pro-0813", cfg=cfg)
    out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert [t["name"] for t in out] == ["terminal", "read_file"]


def test_first_request_anchors_to_bootstrap_subset():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert [t["name"] for t in out] == ["terminal", "read_file"]
    # Schemas are the same objects (request-local filter, no mutation).
    assert out[0] is FULL_CATALOG[0]
    assert out[1] is FULL_CATALOG[1]


def test_after_assistant_tool_call_full_catalog_restored():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user", "toolcall", "toolresult"))
    assert out == FULL_CATALOG
    assert getattr(agent, _AGENT_PROMOTED_ATTR) is True


def test_promotion_latch_survives_compacted_history():
    """A compressed transcript must not re-anchor an already promoted agent."""
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("toolresult")) == FULL_CATALOG
    # Simulate compaction dropping the old tool messages from the API view.
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == FULL_CATALOG


def test_tool_result_alone_also_promotes():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("toolresult"))
    assert out == FULL_CATALOG


def test_sessions_promote_independently():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    promoted = apply_tool_bootstrap(_agent(cfg=cfg), FULL_CATALOG, _messages("toolcall"))
    fresh = apply_tool_bootstrap(_agent(cfg=cfg), FULL_CATALOG, _messages("user"))
    assert promoted == FULL_CATALOG
    assert [t["name"] for t in fresh] == ["terminal", "read_file"]


def test_bootstrap_tools_absent_degrades_to_full_catalog():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    catalog = [{"name": "web_search"}, {"name": "write_file"}]
    assert apply_tool_bootstrap(agent, catalog, _messages("user")) == catalog


def test_partially_present_bootstrap_keeps_only_available():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    catalog = [{"name": "terminal"}, {"name": "web_search"}, {"name": "write_file"}]
    out = apply_tool_bootstrap(agent, catalog, _messages("user"))
    assert [t["name"] for t in out] == ["terminal"]


def test_empty_bootstrap_config_returns_catalog_unchanged():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": []}
    agent = _agent(cfg=cfg)
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == FULL_CATALOG


def test_empty_catalog_returns_empty_catalog():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    assert apply_tool_bootstrap(agent, [], _messages("user")) == []


def test_none_catalog_fails_open():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    assert apply_tool_bootstrap(agent, None, _messages("user")) is None


def test_none_history_still_anchors_first_request():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    out = apply_tool_bootstrap(agent, FULL_CATALOG, None)
    assert out is not None
    assert [t["name"] for t in out] == ["terminal", "read_file"]


def test_non_dict_entries_are_skipped_safely():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    catalog = [{"name": "terminal"}, "garbage", {"name": "read_file"}]
    out = apply_tool_bootstrap(agent, catalog, _messages("user"))
    assert [t["name"] for t in out] == ["terminal", "read_file"]


def test_nested_openai_schema_anchors_and_promotes():
    """Real Hermes registrations use {"type": "function",
    "function": {"name": ...}} — the filter must read the nested name."""
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    first = apply_tool_bootstrap(agent, NESTED_CATALOG, _messages("user"))
    assert [t["function"]["name"] for t in first] == ["terminal", "read_file"]
    # Schemas are preserved by identity (request-local filter, no copy).
    assert first[0] is NESTED_CATALOG[0]

    promoted = apply_tool_bootstrap(
        agent, NESTED_CATALOG,
        _messages("user", "toolcall", "toolresult"),
    )
    assert promoted == NESTED_CATALOG


def test_nested_schema_partial_availability():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    catalog = [NESTED_CATALOG[0], NESTED_CATALOG[3], NESTED_CATALOG[1]]
    out = apply_tool_bootstrap(agent, catalog, _messages("user"))
    assert [t["function"]["name"] for t in out] == ["terminal", "read_file"]


def test_responses_transport_receives_filtered_then_full_catalog():
    """The production codex_responses conversion sees the same two phases."""
    from agent.transports.codex import ResponsesApiTransport

    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = _agent(cfg=cfg)
    transport = ResponsesApiTransport()
    first = apply_tool_bootstrap(agent, NESTED_CATALOG, _messages("user"))
    first_wire = transport.build_kwargs(
        model="deepseek-v4-pro",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}],
        tools=first,
        reasoning_config={"effort": "medium", "enabled": True},
        base_url="http://127.0.0.1:8080/v1",
        provider="custom",
    )
    assert [tool["name"] for tool in first_wire["tools"]] == ["terminal", "read_file"]

    promoted = apply_tool_bootstrap(agent, NESTED_CATALOG, _messages("toolresult"))
    promoted_wire = transport.build_kwargs(
        model="deepseek-v4-pro",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}],
        tools=promoted,
        reasoning_config={"effort": "medium", "enabled": True},
        base_url="http://127.0.0.1:8080/v1",
        provider="custom",
    )
    assert [tool["name"] for tool in promoted_wire["tools"]] == [
        "terminal", "read_file", "write_file", "web_search", "patch",
    ]


# ── Subagent exemption (upstream issue #15) ────────────────────────────────


def test_subagent_returns_full_catalog_even_on_first_request():
    cfg = {"enabled": True, "models": [], "bootstrap_tools": ["terminal", "read_file"]}
    agent = SimpleNamespace(model="deepseek-v4-pro", is_subagent=True)
    setattr(agent, _AGENT_CFG_ATTR, cfg)
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == FULL_CATALOG


# ── First-request context strip (upstream issue #6/#12) ────────────────────

from agent.tool_bootstrap import (
    bootstrap_max_output_tokens,
    strip_bootstrap_context,
)


def _ctx_agent(model="deepseek-v4-pro", is_subagent=False):
    agent = SimpleNamespace(model=model, is_subagent=is_subagent)
    setattr(agent, _AGENT_CFG_ATTR, {
        "enabled": True,
        "models": ["deepseek-v4-pro"],
        "bootstrap_tools": ["terminal", "patch"],
        "bootstrap_max_tokens": 0,
        "suppressed_context_sources": ["agent-instructions", "skill-catalog"],
    })
    agent._bootstrap_context_files_prompt = "## AGENTS.md\n\n# 工作区上下文"
    agent._bootstrap_skills_prompt = "## Skills (mandatory)\n\n<available_skills>\nx\n</available_skills>"
    return agent


SP = (
    "身份指令\n\n"
    "## AGENTS.md\n\n# 工作区上下文\n\n"
    "## Skills (mandatory)\n\n<available_skills>\nx\n</available_skills>\n\n"
    "MEMORY (your personal notes)"
)


def test_strip_bootstrap_context_removes_both_blocks():
    agent = _ctx_agent()
    out = strip_bootstrap_context(agent, SP)
    assert "AGENTS.md" not in out
    assert "available_skills" not in out
    assert "身份指令" in out
    assert "MEMORY" in out


def test_strip_bootstrap_context_preserves_non_matching_model():
    agent = _ctx_agent(model="gpt-5.6-sol")
    assert strip_bootstrap_context(agent, SP) == SP


def test_strip_bootstrap_context_preserves_subagent():
    agent = _ctx_agent(is_subagent=True)
    assert strip_bootstrap_context(agent, SP) == SP


def test_strip_bootstrap_context_preserves_promoted():
    agent = _ctx_agent()
    setattr(agent, _AGENT_PROMOTED_ATTR, True)
    assert strip_bootstrap_context(agent, SP) == SP


def test_strip_bootstrap_context_empty_sources_disables():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["suppressed_context_sources"] = []
    assert strip_bootstrap_context(agent, SP) == SP


def test_strip_bootstrap_context_missing_block_fails_open():
    agent = _ctx_agent()
    agent._bootstrap_context_files_prompt = ""
    agent._bootstrap_skills_prompt = ""
    assert strip_bootstrap_context(agent, SP) == SP


# ── Full Minimal-condition port: zero tools + minimal prompt replacement ──


def test_zero_tools_first_request_returns_empty_catalog():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_zero_tools"] = True
    agent._anchored_tool_bootstrap_config["promote_on"] = "either"
    out = apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert out == []


def test_zero_tools_promotes_on_first_assistant_message():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_zero_tools"] = True
    agent._anchored_tool_bootstrap_config["promote_on"] = "either"
    # First request: no assistant history → zero tools.
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user")) == []
    # Second request: history now carries the plain assistant message from
    # request #1 → promoted, full catalog restored.
    history = _messages("user") + [{"role": "assistant", "content": "OK"}]
    assert apply_tool_bootstrap(agent, FULL_CATALOG, history) == FULL_CATALOG


def test_active_request_flag_set_on_zero_tools_anchoring():
    """The conversation loop consumes the per-request flag to auto-continue
    after the anchored round-1 reply."""
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_zero_tools"] = True
    agent._anchored_tool_bootstrap_config["promote_on"] = "either"
    apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert getattr(agent, _AGENT_ACTIVE_REQUEST_ATTR) is True


def test_active_request_flag_cleared_when_not_anchoring():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_zero_tools"] = True
    agent._anchored_tool_bootstrap_config["promote_on"] = "either"
    # Stale True from a previous anchored request...
    setattr(agent, _AGENT_ACTIVE_REQUEST_ATTR, True)
    # ...must not survive a request that does NOT anchor (promoted session).
    history = _messages("user") + [{"role": "assistant", "content": "OK"}]
    apply_tool_bootstrap(agent, FULL_CATALOG, history)
    assert getattr(agent, _AGENT_ACTIVE_REQUEST_ATTR) is False


def test_active_request_flag_set_for_subset_anchoring():
    agent = _ctx_agent()  # bootstrap_tools subset, no zero-tools key
    apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert getattr(agent, _AGENT_ACTIVE_REQUEST_ATTR) is True


def test_active_request_flag_cleared_when_disabled():
    agent = _agent(cfg={"enabled": False})
    setattr(agent, _AGENT_ACTIVE_REQUEST_ATTR, True)
    apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user"))
    assert getattr(agent, _AGENT_ACTIVE_REQUEST_ATTR) is False


def test_zero_tools_without_promote_on_either_stays_anchored():
    """Default promote_on=tool-call cannot promote a zero-tools first round,
    so a plain assistant reply keeps the session anchored (documented
    footgun; zero-tools deployments must set promote_on: either)."""
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_zero_tools"] = True
    history = _messages("user") + [{"role": "assistant", "content": "OK"}]
    assert apply_tool_bootstrap(agent, FULL_CATALOG, history) == []


def test_promote_on_either_also_promotes_tool_call():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["promote_on"] = "either"
    assert apply_tool_bootstrap(agent, FULL_CATALOG, _messages("user", "toolcall")) == FULL_CATALOG


def test_strip_bootstrap_context_replaces_with_minimal_prompt():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_minimal_system_prompt"] = (
        "You are Hermes Agent. Reply in Simplified Chinese."
    )
    out = strip_bootstrap_context(agent, SP)
    assert out == "You are Hermes Agent. Reply in Simplified Chinese."


def test_strip_bootstrap_context_minimal_prompt_respects_promotion():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_minimal_system_prompt"] = "MIN"
    setattr(agent, _AGENT_PROMOTED_ATTR, True)
    assert strip_bootstrap_context(agent, SP) == SP


# ── User-message runtime-injection strip (memory-context tail) ─────────────


def _memory_messages():
    return [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "只回复OK\n\n<memory-context>\n[System note: recalled memory]\n- user prefers OK.\n</memory-context>"},
    ]


def test_strip_bootstrap_user_injections_removes_memory_tail():
    from agent.tool_bootstrap import strip_bootstrap_user_injections

    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["suppressed_context_sources"] = [
        "agent-instructions", "skill-catalog", "memory-context",
    ]
    msgs = _memory_messages()
    out = strip_bootstrap_user_injections(agent, msgs)
    assert out is not msgs  # request-local copy
    assert "<memory-context>" not in out[-1]["content"]
    assert out[-1]["content"] == "只回复OK"
    assert out[0] is msgs[0]  # untouched message kept as-is


def test_strip_bootstrap_user_injections_disabled_without_source():
    from agent.tool_bootstrap import strip_bootstrap_user_injections

    agent = _ctx_agent()  # default suppressed sources lack memory-context
    msgs = _memory_messages()
    assert strip_bootstrap_user_injections(agent, msgs) is msgs


def test_strip_bootstrap_user_injections_skips_promoted():
    from agent.tool_bootstrap import strip_bootstrap_user_injections

    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["suppressed_context_sources"] = [
        "agent-instructions", "skill-catalog", "memory-context",
    ]
    setattr(agent, _AGENT_PROMOTED_ATTR, True)
    msgs = _memory_messages()
    assert strip_bootstrap_user_injections(agent, msgs) is msgs


# ── First-request output cap (upstream issue #6/#11) ───────────────────────


def test_bootstrap_max_output_tokens_default_disabled():
    agent = _ctx_agent()  # bootstrap_max_tokens = 0
    assert bootstrap_max_output_tokens(agent) is None


def test_bootstrap_max_output_tokens_configured():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_max_tokens"] = 1024
    assert bootstrap_max_output_tokens(agent) == 1024


def test_bootstrap_max_output_tokens_promoted_returns_none():
    agent = _ctx_agent()
    agent._anchored_tool_bootstrap_config["bootstrap_max_tokens"] = 1024
    setattr(agent, _AGENT_PROMOTED_ATTR, True)
    assert bootstrap_max_output_tokens(agent) is None


def test_bootstrap_max_output_tokens_non_matching_model_returns_none():
    agent = _ctx_agent(model="gpt-5.6-sol")
    agent._anchored_tool_bootstrap_config["bootstrap_max_tokens"] = 1024
    assert bootstrap_max_output_tokens(agent) is None
